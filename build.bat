@echo off
REM Build script for Ball Tracker C++ application on Windows
REM This script automates the CMake build process for Visual Studio

echo === Ball Tracker Build Script (Windows) ===

REM Create build directory if it doesn't exist
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

REM Configure with CMake for Visual Studio
echo Configuring with CMake for Visual Studio...
cmake .. -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

REM Build the application
echo Building application...
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo === Build Complete ===
echo Executable location: build\bin\Release\ball_tracker.exe
echo.
echo To run the application:
echo   .\build\bin\Release\ball_tracker.exe
echo.
echo To see debug output:
echo   .\build\bin\Release\ball_tracker.exe 2^>debug.log
echo.
pause