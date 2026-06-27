@echo off
setlocal EnableDelayedExpansion
title UniverSR Super-Resolution - Installer
color 0A

echo.
echo ============================================================
echo  UniverSR Super-Resolution - Dependency Installer
echo  Requires: NVIDIA GPU with CUDA
echo ============================================================
echo.

set ROOT_DIR=%~dp0..
set VENV_DIR=%ROOT_DIR%\venvs\venv_universr
set REQ_FILE=%~dp0universr_requirements.txt

:: ----------------------------------------------------------------
:: [1/8] Check Python
:: ----------------------------------------------------------------
echo [1/8] Checking Python...
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3 not found in PATH.
    echo         Download from: https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('py -3 --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
echo [OK] Found Python !PY_VER!
if !PY_MINOR! LSS 10 (
    echo [ERROR] Python 3.10 or newer is required. Found !PY_VER!
    pause
    exit /b 1
)
if !PY_MINOR! GTR 12 (
    echo [WARNING] Python !PY_VER! is newer than tested ^(3.10-3.12^). Proceeding anyway.
)
echo.

:: ----------------------------------------------------------------
:: [2/8] Check required files
:: ----------------------------------------------------------------
echo [2/8] Checking required files...
if not exist "%REQ_FILE%" (
    echo [ERROR] universr_requirements.txt not found at: %REQ_FILE%
    pause
    exit /b 1
)
echo [OK] universr_requirements.txt found.
echo.

:: ----------------------------------------------------------------
:: [3/8] Create virtual environment
:: ----------------------------------------------------------------
echo [3/8] Setting up virtual environment...
if exist "%VENV_DIR%" (
    echo [INFO] Removing existing venv_universr...
    rmdir /s /q "%VENV_DIR%"
)
py -3 -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created.
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.
echo.

:: ----------------------------------------------------------------
:: [4/8] Detect NVIDIA GPU and select CUDA version
:: ----------------------------------------------------------------
echo [4/8] Detecting NVIDIA GPU...
set GPU_FOUND=0
set DRIVER_MAJOR=0

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set GPU_FOUND=1
    nvidia-smi --query-gpu=driver_version --format=csv,noheader > "%TEMP%\universr_gpu_driver.txt" 2>nul
    for /f "tokens=1 delims=." %%a in (%TEMP%\universr_gpu_driver.txt) do (
        set DRIVER_MAJOR=%%a
        goto :USR_DRIVER_DONE
    )
)
:USR_DRIVER_DONE
del "%TEMP%\universr_gpu_driver.txt" >nul 2>&1

if !GPU_FOUND! EQU 0 (
    echo [ERROR] No NVIDIA GPU detected.
    echo         UniverSR requires an NVIDIA GPU with CUDA.
    echo         CPU-only mode is not supported.
    pause
    exit /b 1
)

echo [OK] NVIDIA GPU detected. Driver major version: !DRIVER_MAJOR!

if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
    goto :USR_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
    goto :USR_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
    goto :USR_START_INSTALL
)

echo [ERROR] NVIDIA driver version !DRIVER_MAJOR! is too old ^(minimum: 450^).
echo         Please update your GPU drivers from: https://www.nvidia.com/drivers
pause
exit /b 1

:USR_START_INSTALL
echo [INFO] Selected CUDA variant: !TORCH_VARIANT!
echo.

:: ----------------------------------------------------------------
:: [5/8] Upgrade pip and install PyTorch CUDA
:: ----------------------------------------------------------------
echo [5/8] Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel --quiet
echo.

echo Installing PyTorch ^(!TORCH_VARIANT!^)...
python -m pip install torch torchaudio --index-url !TORCH_URL!
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)
echo [OK] PyTorch installed.
echo.

:: ----------------------------------------------------------------
:: [6/8] Install requirements.txt
:: ----------------------------------------------------------------
echo [6/8] Installing dependencies from universr_requirements.txt...
python -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
    echo [WARNING] Some packages may have failed. Check the output above.
) else (
    echo [OK] All requirements installed.
)
echo.

echo Restoring PyTorch CUDA ^(requirements.txt may have overwritten it^)...
python -m pip install torch torchaudio torchvision --index-url !TORCH_URL! --force-reinstall --no-deps
if errorlevel 1 (
    echo [ERROR] PyTorch restore failed.
    pause
    exit /b 1
)
echo [OK] PyTorch CUDA restored: !TORCH_VARIANT!
echo.

:: ----------------------------------------------------------------
:: [7/8] Install UniverSR from GitHub
:: ----------------------------------------------------------------
echo [7/8] Installing UniverSR from GitHub...

echo      Installing git+https://github.com/woongzip1/UniverSR.git ...
python -m pip install "git+https://github.com/woongzip1/UniverSR.git"
if errorlevel 1 (
    echo [ERROR] UniverSR installation from GitHub failed.
    echo         Make sure git is installed and you have internet access.
    echo         Download git from: https://git-scm.com/downloads
    pause
    exit /b 1
)
echo [OK] UniverSR installed.
echo.

echo Restoring PyTorch CUDA after UniverSR install...
python -m pip install torch torchaudio torchvision --index-url !TORCH_URL! --force-reinstall --no-deps --quiet
if errorlevel 1 (
    echo [WARNING] PyTorch CUDA re-restore failed. Check GPU functionality.
) else (
    echo [OK] PyTorch CUDA confirmed: !TORCH_VARIANT!
)
echo.

:: ----------------------------------------------------------------
:: [8/8] Verify installation
:: ----------------------------------------------------------------
echo [8/8] Verifying installation...
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__, '- import OK')"
python -c "from universr import UniverSR; print('universr: import OK')"
if errorlevel 1 (
    echo [ERROR] universr failed to import.
    pause
    exit /b 1
)
python -c "import PyQt6; print('PyQt6: import OK')"
echo.

echo ============================================================
echo  UniverSR installation complete!
echo.
echo  PyTorch variant : !TORCH_VARIANT!
echo  Venv location   : %VENV_DIR%
echo.
echo  Next steps:
echo    1. Run start.bat from the application root
echo    2. Select "UniverSR Super-Resolution" from the menu
echo    3. Drop an audio or video file and click "Enhance Audio"
echo.
echo  Models are downloaded automatically on first run.
echo  Available models:
echo    - woongzip1/universr-audio  (general audio, recommended)
echo    - woongzip1/universr-speech (speech only)
echo.
echo  Note: ffmpeg is required for video files and MP3/AAC input
echo        Download: https://ffmpeg.org/download.html
echo ============================================================
echo.
pause
