#include "raylib.h"
#include "raymath.h" // Required for Vector math functions
#include <cmath>     // For std::pow, std::log, std::fmod
#include <thread>
#include <vector>
#include <atomic>    // Required for std::atomic

constexpr int screenWidth = 1280;
constexpr int screenHeight = 720;

// Global Mandelbrot parameters
int maxIterations = 100;
Vector2 viewCenter = {-0.7, 0.0};
double viewWidthComplex = 3.5;
constexpr double initialViewWidthComplex = 3.5;

// Interaction state
double zoomFactor = 1.1;
bool isPanning = false;
Vector2 panStartMouse = {0, 0};
Vector2 panStartCenter = {0, 0};

RenderTexture2D mandelbrotTexture;
bool needsRedraw = true;

// Atomic counter for render generations to handle interruptions
std::atomic<unsigned int> g_currentRenderGeneration(0);

// Calculates iterations and final complex value for a point in the complex plane
// Returns iteration count. Stores final complex value components in zx_out, zy_out.
int CalculateMandelbrot(const double cx, const double cy, const int maxIter, double &zx_out, double &zy_out) {
    // Check if the point is in the main cardioid
    if (const double q = (cx - 0.25) * (cx - 0.25) + cy * cy; q * (q + (cx - 0.25)) < 0.25 * cy * cy) {
        zx_out = 0;
        zy_out = 0;
        return maxIter;
    }
    // Check if the point is in the period-2 bulb
    if ((cx + 1.0) * (cx + 1.0) + cy * cy < 0.0625) {
        zx_out = 0;
        zy_out = 0;
        return maxIter;
    }

    double zx = 0.0;
    double zy = 0.0;
    double zx2 = 0.0;
    double zy2 = 0.0;
    int iter = 0;

    while (zx2 + zy2 < 4.0 && iter < maxIter) {
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
    }
    zx_out = zx;
    zy_out = zy;
    return iter;
}

const double LOG2 = std::log(2.0);

Color GetMandelbrotColor(int iterations, int currentMaxIterations, double zReal, double zImag) {
    if (iterations >= currentMaxIterations) {
        return BLACK;
    }
    const double log_zn = std::log(zReal * zReal + zImag * zImag) / 2.0;
    const double nu = std::log(log_zn / LOG2) / LOG2;
    const double smoothIter = static_cast<double>(iterations) + 1.0 - nu;
    const auto hue = static_cast<float>(std::fmod(smoothIter * 0.03, 1.0) * 360.0);
    constexpr float saturation = 0.85f;
    constexpr float value = 0.75f;
    return ColorFromHSV(hue, saturation, value);
}

Vector2 MapPixelToComplex(const Vector2 pixelPos, const Vector2 currentViewCenter, const double currentComplexWidth,
                          const int currentScreenWidth, const int currentScreenHeight) {
    const double scale = currentComplexWidth / currentScreenWidth;
    const double cx = currentViewCenter.x + (pixelPos.x - currentScreenWidth / 2.0) * scale;
    const double cy = currentViewCenter.y - (pixelPos.y - currentScreenHeight / 2.0) * scale;
    return {static_cast<float>(cx), static_cast<float>(cy)};
}

void UpdateMandelbrotTexture(const RenderTexture2D &targetTexture, const Vector2 centerToRender,
                             const double complexWidthToRender, const int iterLimit) {
    // Capture the generation number for this specific render call
    const unsigned int capturedRenderGeneration = g_currentRenderGeneration.load(std::memory_order_acquire);

    const int texWidth = targetTexture.texture.width;
    const int texHeight = targetTexture.texture.height;
    std::vector<Color> pixelColors(static_cast<size_t>(texWidth) * texHeight); // Consider reusing this buffer
    unsigned int nThreadsUnsigned = std::thread::hardware_concurrency();
    if (nThreadsUnsigned == 0) nThreadsUnsigned = 1; // Fallback to 1 thread
    const int nThreads = static_cast<int>(nThreadsUnsigned);
    std::vector<std::thread> threads;
    threads.reserve(nThreadsUnsigned);

    const double scale = complexWidthToRender / texWidth; // Define scale for lambda capture

    auto computeRows = [&](const int startY, const int endY) {
        for (int y = startY; y < endY; ++y) {
            // Check if this render task is still current before processing a row
            if (g_currentRenderGeneration.load(std::memory_order_acquire) != capturedRenderGeneration) {
                return; // Abandon this outdated render task
            }
            for (int x = 0; x < texWidth; ++x) {
                const double cx = centerToRender.x + (x - texWidth / 2.0) * scale;
                const double cy = centerToRender.y - (y - texHeight / 2.0) * scale;
                double finalZReal, finalZImag;
                const int iterations = CalculateMandelbrot(cx, cy, iterLimit, finalZReal, finalZImag);
                pixelColors[static_cast<size_t>(y) * texWidth + x] = GetMandelbrotColor(
                    iterations, iterLimit, finalZReal, finalZImag);
            }
        }
    };

    const int chunkSize = texHeight / nThreads;
    const int remainingRows = texHeight % nThreads;
    int currentY = 0;
    for (int i = 0; i < nThreads; ++i) {
        if (const int rowsForThread = chunkSize + (i < remainingRows ? 1 : 0); rowsForThread > 0) {
            threads.emplace_back(computeRows, currentY, currentY + rowsForThread);
            currentY += rowsForThread;
        }
    }
    for (auto &t : threads) {
        if (t.joinable()) t.join();
    }

    // After threads are joined, check if this render is still the most current one
    if (g_currentRenderGeneration.load(std::memory_order_acquire) == capturedRenderGeneration) {
        UpdateTexture(targetTexture.texture, pixelColors.data());
    }
    // If not the current generation, do nothing; the main loop will handle `needsRedraw`.
}

int main() {
    InitWindow(screenWidth, screenHeight, "Mandelbrot Viewer");
    SetTargetFPS(60); // Adjust FPS as needed, higher FPS might make interruptions more noticeable

    mandelbrotTexture = LoadRenderTexture(screenWidth, screenHeight);
    needsRedraw = true; // Render on the first frame

    while (!WindowShouldClose()) {
        const float wheelMove = GetMouseWheelMove();
        const Vector2 currentMousePos = GetMousePosition();
        bool interactionOccurred = false;

        if (wheelMove != 0) {
            const auto [xVal, yVal] = MapPixelToComplex(currentMousePos, viewCenter, viewWidthComplex, screenWidth, screenHeight);

            viewWidthComplex *= std::pow(zoomFactor, -wheelMove);
            maxIterations = static_cast<int>(100.0 + 150.0 * std::log(initialViewWidthComplex / viewWidthComplex));
            if (maxIterations < 100) maxIterations = 100;

            const Vector2 mouseComplexPosAfterZoom = MapPixelToComplex(currentMousePos, viewCenter, viewWidthComplex, screenWidth, screenHeight);
            viewCenter.x += (xVal - mouseComplexPosAfterZoom.x);
            viewCenter.y -= (yVal - mouseComplexPosAfterZoom.y); // Corrected y-coordinate adjustment for complex plane

            interactionOccurred = true;
        }

        const double currentScale = viewWidthComplex / screenWidth;
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            isPanning = true;
            panStartMouse = currentMousePos;
            panStartCenter = viewCenter;
            // No need to set interactionOccurred here, as continuous panning will handle it
        }

        if (isPanning) {
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                Vector2 mouseDelta = Vector2Subtract(currentMousePos, panStartMouse);
                // Check if there's actual movement to avoid redundant redraws
                if (mouseDelta.x != 0 || mouseDelta.y != 0) {
                    viewCenter.x = static_cast<float>(panStartCenter.x - mouseDelta.x * currentScale);
                    viewCenter.y = static_cast<float>(panStartCenter.y - mouseDelta.y * currentScale); // Corrected y-coordinate adjustment
                    interactionOccurred = true;
                }
            }
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                isPanning = false;
                // interactionOccurred might have been set if there was movement.
                // If only a click, no redraw was triggered yet.
            }
        }
        
        if(interactionOccurred){
            g_currentRenderGeneration.fetch_add(1, std::memory_order_release); // Increment generation
            needsRedraw = true;
        }


        // --- Mandelbrot Texture Update ---
        if (needsRedraw) {
            unsigned int generationBeforeUpdate = g_currentRenderGeneration.load(std::memory_order_acquire);
            UpdateMandelbrotTexture(mandelbrotTexture, viewCenter, viewWidthComplex, maxIterations);
            
            // If the generation hasn't changed during the update process, the render is valid.
            if (g_currentRenderGeneration.load(std::memory_order_acquire) == generationBeforeUpdate) {
                 needsRedraw = false;
            }
            // Otherwise, needsRedraw remains true, and the loop will attempt to render the newer generation.
        }

        // --- Drawing ---
        BeginDrawing();
        ClearBackground(RAYWHITE);

        const auto texViewWidth = static_cast<float>(mandelbrotTexture.texture.width);
        const auto texViewHeight = static_cast<float>(mandelbrotTexture.texture.height);
        const Rectangle sourceRec = {0.0f, 0.0f, texViewWidth, -texViewHeight}; 
        constexpr Vector2 textureDrawPosition = {0.0f, 0.0f};
        DrawTextureRec(mandelbrotTexture.texture, sourceRec, textureDrawPosition, WHITE);

        // --- UI ---
        DrawRectangle(5, 5, 330, 85, Fade(SKYBLUE, 0.7f));
        DrawRectangleLines(5, 5, 330, 85, BLUE);
        DrawText("Mandelbrot Viewer", 15, 15, 20, BLUE);
        DrawText(TextFormat("Center: (%.5f, %.5f)", viewCenter.x, viewCenter.y), 15, 40, 10, DARKBLUE);
        DrawText(TextFormat("Width: %.3e", viewWidthComplex), 15, 55, 10, DARKBLUE);
        DrawText(TextFormat("Iterations: %d", maxIterations), 15, 70, 10, DARKBLUE);

        DrawFPS(screenWidth - 80, 10);
        EndDrawing();
    }

    UnloadRenderTexture(mandelbrotTexture);
    CloseWindow();
    return 0;
}