@echo off
REM ============================================================
REM  ROV Simulator — Windows .exe Builder
REM  
REM  This script creates a standalone ROV_Simulator.exe that
REM  anyone can run without installing Python or any packages.
REM
REM  REQUIREMENTS:
REM    - Windows 10/11
REM    - Python 3.10+ installed (python.org or Microsoft Store)
REM    - Internet connection (to download packages)
REM
REM  USAGE:
REM    1. Double-click this file, OR
REM    2. Open Command Prompt in this folder and run:
REM         build_windows_exe.bat
REM
REM  OUTPUT:
REM    dist\ROV_Simulator\ROV_Simulator.exe
REM    (Copy the entire ROV_Simulator folder to share it)
REM ============================================================

REM Keep the window open on ANY exit (error or success)
REM by using cmd /k to force a persistent prompt
if "%~1"=="" (
    cmd /k "%~f0" RUNNING
    exit /b
)

echo.
echo ============================================================
echo   ROV Simulator — Windows .exe Builder
echo ============================================================
echo.

REM Show Python version so user can see something is happening
echo Checking for Python...
python --version 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    echo Press any key to close...
    pause >nul
    exit /b 1
)
echo.

echo [1/4] Creating virtual environment...
echo         (this may take a moment)
if exist build_venv rmdir /s /q build_venv
python -m venv build_venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo Press any key to close...
    pause >nul
    exit /b 1
)
call build_venv\Scripts\activate.bat
echo         Done.
echo.

echo [2/4] Installing dependencies...
echo         Installing: pyinstaller pybullet numpy opencv-python Pillow
echo         (this will take a few minutes — downloading packages)
echo.
pip install --upgrade pip
pip install pyinstaller pybullet numpy opencv-python Pillow
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Press any key to close...
    pause >nul
    exit /b 1
)
echo.

echo [3/4] Building .exe with PyInstaller...
echo         (this will take 1-3 minutes)
echo.
pyinstaller rov_sim.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed.
    echo Check the output above for details.
    echo Press any key to close...
    pause >nul
    exit /b 1
)
echo.

echo [4/4] Verifying output...
if exist "dist\ROV_Simulator\ROV_Simulator.exe" (
    echo.
    echo ============================================================
    echo   BUILD SUCCESSFUL!
    echo ============================================================
    echo.
    echo   Your standalone simulator is at:
    echo     dist\ROV_Simulator\ROV_Simulator.exe
    echo.
    echo   To share it:
    echo     1. Copy the entire "dist\ROV_Simulator" folder
    echo     2. Send it as a .zip to anyone with Windows
    echo     3. They just double-click ROV_Simulator.exe
    echo.
    echo   No Python installation needed on the target machine!
    echo ============================================================
) else (
    echo.
    echo ERROR: Build completed but .exe not found.
    echo Check the dist\ folder manually.
)

echo.
echo This window will stay open. You can close it manually.
echo.