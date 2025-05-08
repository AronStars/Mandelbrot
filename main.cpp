#include "raylib.h"
#include "raymath.h" // Required for Vector math functions
#include <cmath>     // For std::pow, std::log, std::fmod
#include <thread>
#include <vector>
#include <atomic>    // Required for std::atomic

constexpr int screenWidth = 1280;
constexpr int screenHeight = 720;

constexpr int LOW_RES_DOWNSAMPLE_FACTOR = 2; // Use 2 for half-resolution, 4 for quarter, etc.
constexpr int lowResScreenWidth = screenWidth / LOW_RES_DOWNSAMPLE_FACTOR;
constexpr int lowResScreenHeight = screenHeight / LOW_RES_DOWNSAMPLE_FACTOR;

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
bool isLowResPanningActive = false; // True if currently panning and using low-res preview

RenderTexture2D mandelbrotTexture;
RenderTexture2D lowResMandelbrotTexture; // For low-resolution panning preview
bool needsRedraw = true;

std::atomic<unsigned int> g_currentRenderGeneration(0);

int CalculateMandelbrot(const double cx, const double cy, const int maxIter, double &zx_out, double &zy_out) {
    if (const double q = (cx - 0.25) * (cx - 0.25) + cy * cy; q * (q + (cx - 0.25)) < 0.25 * cy * cy) {
        zx_out = 0;
        zy_out = 0;
        return maxIter;
    }
    if ((cx + 1.0) * (cx + 1.0) + cy * cy < 0.0625) {
        zx_out = 0;
        zy_out = 0;
        return maxIter;
    }
    double zx = 0.0, zy = 0.0, zx2 = 0.0, zy2 = 0.0;
    int iter = 0;
    while (zx2 + zy2 < 4.0 && iter < maxIter) {
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
    }
    zx_out = zx; zy_out = zy; return iter;
}

const double LOG2 = std::log(2.0);

Color GetMandelbrotColor(const int iterations, const int currentMaxIterations, const double zReal, const double zImag) {
    if (iterations >= currentMaxIterations) return BLACK;
    const double log_zn = std::log(zReal * zReal + zImag * zImag) / 2.0;
    const double nu = std::log(log_zn / LOG2) / LOG2;
    const double smoothIter = static_cast<double>(iterations) + 1.0 - nu;
    const auto hue = static_cast<float>(std::fmod(smoothIter * 0.03, 1.0) * 360.0);
    constexpr float saturation = 0.85f; constexpr float value = 0.75f;
    return ColorFromHSV(hue, saturation, value);
}

Vector2 MapPixelToComplex(const Vector2 pixelPos, const Vector2 currentViewCenter, const double currentComplexWidth,
                          const int currentScreenWidth, const int currentScreenHeight) {
    const double scale = currentComplexWidth / currentScreenWidth;
    const double cx = currentViewCenter.x + (pixelPos.x - currentScreenWidth / 2.0) * scale;
    const double cy = currentViewCenter.y - (pixelPos.y - currentScreenHeight / 2.0) * scale;
    return {static_cast<float>(cx), static_cast<float>(cy)};
}

void UpdateMandelbrotTexture(const RenderTexture2D &targetTexture,
                             const Vector2 centerToRender, const double complexWidthToRender, const int iterLimit) {
    const unsigned int capturedRenderGeneration = g_currentRenderGeneration.load(std::memory_order_acquire);
    const int texWidth = targetTexture.texture.width;
    const int texHeight = targetTexture.texture.height;
    std::vector<Color> pixelColors(static_cast<size_t>(texWidth) * texHeight);
    unsigned int nThreadsUnsigned = std::thread::hardware_concurrency();
    if (nThreadsUnsigned == 0) nThreadsUnsigned = 1;
    const int nThreads = static_cast<int>(nThreadsUnsigned);
    std::vector<std::thread> threads;
    threads.reserve(nThreadsUnsigned);
    const double scale = complexWidthToRender / texWidth;

    auto computeRows = [&](const int startY, const int endY) {
        for (int y = startY; y < endY; ++y) {
            if (g_currentRenderGeneration.load(std::memory_order_acquire) != capturedRenderGeneration) return;
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
    for (auto &t : threads) if (t.joinable()) t.join();

    if (g_currentRenderGeneration.load(std::memory_order_acquire) == capturedRenderGeneration) {
        UpdateTexture(targetTexture.texture, pixelColors.data());
    }
}

int main() {
    InitWindow(screenWidth, screenHeight, "Mandelbrot Viewer");
    SetTargetFPS(60); // Higher FPS can make interaction feel smoother

    mandelbrotTexture = LoadRenderTexture(screenWidth, screenHeight);
    lowResMandelbrotTexture = LoadRenderTexture(lowResScreenWidth, lowResScreenHeight);
    needsRedraw = true; 

    while (!WindowShouldClose()) {
        const float wheelMove = GetMouseWheelMove();
        const Vector2 currentMousePos = GetMousePosition();
        bool interactionOccurred = false;

        if (wheelMove != 0) {
            const auto [xVal, yVal] = MapPixelToComplex(currentMousePos, viewCenter, viewWidthComplex, screenWidth, screenHeight);
            viewWidthComplex *= std::pow(zoomFactor, -wheelMove);
            maxIterations = static_cast<int>(100.0 + 150.0 * std::log(initialViewWidthComplex / viewWidthComplex));
            if (maxIterations < 100) maxIterations = 100;
            const auto [x, y] = MapPixelToComplex(currentMousePos, viewCenter, viewWidthComplex, screenWidth, screenHeight);
            viewCenter.x += (xVal - x);
            viewCenter.y -= (yVal - y);

            isLowResPanningActive = false; // Zooming always requests full-res
            interactionOccurred = true;
        }

        const double currentScale = viewWidthComplex / screenWidth;
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            isPanning = true;
            isLowResPanningActive = true; // Start panning with low res
            panStartMouse = currentMousePos;
            panStartCenter = viewCenter;
            interactionOccurred = true; // Trigger initial low-res draw
        }

        if (isPanning) {
            if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                Vector2 mouseDelta = Vector2Subtract(currentMousePos, panStartMouse);
                if (mouseDelta.x != 0 || mouseDelta.y != 0) {
                    viewCenter.x = static_cast<float>(panStartCenter.x - mouseDelta.x * currentScale);
                    viewCenter.y = static_cast<float>(panStartCenter.y - mouseDelta.y * currentScale);
                    // panStartMouse = currentMousePos; // Optional: for continuous feeling pan from new pos
                    // panStartCenter = viewCenter;
                    isLowResPanningActive = true; // Ensure still in low-res mode
                    interactionOccurred = true;
                }
            }
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                isPanning = false;
                isLowResPanningActive = false; // Stop low-res mode, request full-res
                interactionOccurred = true; // Trigger full-res redraw
            }
        }

        if (interactionOccurred) {
            g_currentRenderGeneration.fetch_add(1, std::memory_order_release);
            needsRedraw = true;
        }

        if (needsRedraw) {
            const unsigned int generationBeforeUpdate = g_currentRenderGeneration.load(std::memory_order_acquire);

            RenderTexture2D currentTargetTexture = isLowResPanningActive ? lowResMandelbrotTexture : mandelbrotTexture;
            UpdateMandelbrotTexture(currentTargetTexture, viewCenter, viewWidthComplex, maxIterations);
            
            if (g_currentRenderGeneration.load(std::memory_order_acquire) == generationBeforeUpdate) {
                 needsRedraw = false;
            }
        }

        BeginDrawing();
        ClearBackground(RAYWHITE);

        if (isLowResPanningActive) {
            const Rectangle lowResSourceRec = {
                0.0f, 0.0f, static_cast<float>(lowResMandelbrotTexture.texture.width),
                -static_cast<float>(lowResMandelbrotTexture.texture.height)
            };
            constexpr Rectangle screenDestRec = {
                0.0f, 0.0f, static_cast<float>(screenWidth), static_cast<float>(screenHeight)
            };
            DrawTexturePro(lowResMandelbrotTexture.texture, lowResSourceRec, screenDestRec, {0, 0}, 0.0f, WHITE);
        } else {
            const Rectangle sourceRec = {
                0.0f, 0.0f, static_cast<float>(mandelbrotTexture.texture.width),
                -static_cast<float>(mandelbrotTexture.texture.height)
            };
            constexpr Vector2 textureDrawPosition = {0.0f, 0.0f};
            DrawTextureRec(mandelbrotTexture.texture, sourceRec, textureDrawPosition, WHITE);
        }

        DrawRectangle(5, 5, 220, 85, Fade(SKYBLUE, 0.7f));
        DrawRectangleLines(5, 5, 220, 85, BLUE);
        DrawText("Mandelbrot Viewer", 15, 15, 20, BLUE);
        DrawText(TextFormat("Center: (%.5f, %.5f)", viewCenter.x, viewCenter.y), 15, 40, 10, DARKBLUE);
        DrawText(TextFormat("Width: %.3e", viewWidthComplex), 15, 55, 10, DARKBLUE);
        DrawText(TextFormat("Iterations: %d", maxIterations), 15, 70, 10, DARKBLUE);
        DrawFPS(screenWidth - 80, 10);
        EndDrawing();
    }

    UnloadRenderTexture(mandelbrotTexture);
    UnloadRenderTexture(lowResMandelbrotTexture);
    CloseWindow();
    return 0;
}