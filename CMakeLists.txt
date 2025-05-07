cmake_minimum_required(VERSION 3.21) # Using 3.21 as a widely available modern version
project(Mandelbrot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG "5.0"
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build Raylib as a static library" FORCE)
set(RAYLIB_BUILD_EXAMPLES OFF CACHE BOOL "Do not build Raylib examples" FORCE)

FetchContent_MakeAvailable(raylib)

add_executable(Mandelbrot WIN32 main.cpp)

target_link_libraries(Mandelbrot PRIVATE raylib)

if (MINGW)
    set_property(TARGET Mandelbrot APPEND_STRING PROPERTY LINK_FLAGS "-static-libgcc -static-libstdc++")
elseif (MSVC)
    set_property(TARGET Mandelbrot PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug Release MinSizeRel RelWithDebInfo)" FORCE)
endif ()

# --- Installation Setup ---
install(TARGETS Mandelbrot
        RUNTIME DESTINATION bin  # Installs the .exe to a 'bin' subfolder
        LIBRARY DESTINATION lib  # For any .dll or .so files directly produced by this target
        ARCHIVE DESTINATION lib  # For .lib files
)

# This section uses BundleUtilities to gather dependencies on Windows
# This section uses BundleUtilities to gather dependencies on Windows
if (WIN32 AND CMAKE_HOST_WIN32) # Ensure this runs on Windows hosts
    install(CODE [===[
        message(STATUS "Running BundleUtilities to fixup bundle at ${CMAKE_INSTALL_PREFIX}/bin/Mandelbrot.exe")
        include(BundleUtilities)
        set(BU_VERBOSE TRUE) # Set to FALSE to make it less chatty

        # List of directories where BundleUtilities should look for DLLs.
        set(BU_LIBRARY_DIRS "C:/Users/arong/AppData/Local/Programs/CLion/bin/mingw/bin") # Path to CLion's MinGW

        fixup_bundle(
            "${CMAKE_INSTALL_PREFIX}/bin/Mandelbrot.exe"  # The installed executable
            ""                                           # No plugins to copy for this app
            "${BU_LIBRARY_DIRS}"                         # Directories to search for libraries
        )
        message(STATUS "BundleUtilities fixup_bundle complete.")
    ]===] COMPONENT Runtime)
endif ()