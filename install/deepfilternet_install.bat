@echo off
setlocal EnableDelayedExpansion
title DeepFilterNet - Installer
color 0A

echo.
echo ============================================================
echo  DeepFilterNet Noise Suppression - Dependency Installer
echo  DeepFilterNet2 / DeepFilterNet3
echo  Supports: NVIDIA GPU (CUDA) or CPU
echo ============================================================
echo.

set "ROOT_DIR=%~dp0.."
set "VENV_DIR=%ROOT_DIR%\venvs\venv_deepfilternet"
set "REQ_FILE=%~dp0deepfilternet_requirements.txt"

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
if not exist "!REQ_FILE!" (
    echo [ERROR] deepfilternet_requirements.txt not found at: !REQ_FILE!
    pause
    exit /b 1
)
echo [OK] deepfilternet_requirements.txt found.
echo.

:: ----------------------------------------------------------------
:: [3/8] Create virtual environment
:: ----------------------------------------------------------------
echo [3/8] Setting up virtual environment...
if exist "!VENV_DIR!" (
    echo [INFO] Removing existing venv_deepfilternet...
    rmdir /s /q "!VENV_DIR!"
)
py -3 -m venv "!VENV_DIR!"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created.
call "!VENV_DIR!\Scripts\activate.bat"
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
set "DF_GPU_TMP=!TEMP!\df_gpu_driver.txt"

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set GPU_FOUND=1
    nvidia-smi --query-gpu=driver_version --format=csv,noheader > "!DF_GPU_TMP!" 2>nul
    for /f "usebackq tokens=1 delims=." %%a in ("!DF_GPU_TMP!") do (
        set DRIVER_MAJOR=%%a
        goto :DF_DRIVER_DONE
    )
)
:DF_DRIVER_DONE
if exist "!DF_GPU_TMP!" del "!DF_GPU_TMP!" >nul 2>&1

if !GPU_FOUND! EQU 0 (
    echo [WARNING] No NVIDIA GPU detected. Installing CPU-only PyTorch.
    set TORCH_URL=https://download.pytorch.org/whl/cpu
    set TORCH_VARIANT=CPU only
    goto :DF_START_INSTALL
)

echo [OK] NVIDIA GPU detected. Driver major version: !DRIVER_MAJOR!

if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
    goto :DF_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
    goto :DF_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
    goto :DF_START_INSTALL
)

echo [WARNING] NVIDIA driver version !DRIVER_MAJOR! is too old ^(minimum: 450^).
echo           Falling back to CPU-only installation.
echo           Please update your GPU drivers from: https://www.nvidia.com/drivers
set TORCH_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU only

:DF_START_INSTALL
echo [INFO] Selected PyTorch variant: !TORCH_VARIANT!
echo.

:: ----------------------------------------------------------------
:: [5/8] Install PyTorch and requirements
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

echo Installing dependencies from deepfilternet_requirements.txt...
python -m pip install -r "!REQ_FILE!"
if errorlevel 1 (
    echo [WARNING] Some packages may have failed. Check the output above.
) else (
    echo [OK] All requirements installed.
)
echo.

echo Restoring PyTorch ^(!TORCH_VARIANT!^)...
python -m pip install torch torchaudio torchvision --index-url !TORCH_URL! --force-reinstall --no-deps
if errorlevel 1 (
    echo [ERROR] PyTorch restore failed.
    pause
    exit /b 1
)
echo [OK] PyTorch restored: !TORCH_VARIANT!
echo.

:: ----------------------------------------------------------------
:: [6/8] Install Rust (if missing)
:: ----------------------------------------------------------------
echo [6/8] Checking for Rust toolchain...
cargo --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Rust not found. Installing Rust ^(required to compile deepfilterlib^)...
    echo         This will download ~150 MB and may take several minutes.
    set RUSTUP_URL=https://win.rustup.rs/x86_64
    set "RUSTUP_INIT=!TEMP!\rustup-init.exe"
    echo Downloading rustup-init...
    powershell -Command "Invoke-WebRequest -Uri '!RUSTUP_URL!' -OutFile '!RUSTUP_INIT!' -ErrorAction SilentlyContinue"
    if errorlevel 1 (
        echo [ERROR] Failed to download rustup-init. Please install Rust manually from https://rustup.rs/
        pause
        exit /b 1
    )
    echo Installing Rust ^(default settings^)...
    "!RUSTUP_INIT!" -y
    if errorlevel 1 (
        echo [ERROR] Rust installation failed.
        pause
        exit /b 1
    )
    echo [OK] Rust installed.
    REM Add cargo to PATH for this session
    set "PATH=!USERPROFILE!\.cargo\bin;!PATH!"
) else (
    echo [OK] Rust found.
)
echo.

:: ----------------------------------------------------------------
:: [7/8] Install deepfilternet
:: ----------------------------------------------------------------
echo [7/8] Installing deepfilternet...
python -m pip install deepfilternet
if errorlevel 1 (
    echo [ERROR] deepfilternet installation failed.
    pause
    exit /b 1
)
echo [OK] deepfilternet installed.
echo.

:: ----------------------------------------------------------------
:: [8/8] Verify installation
:: ----------------------------------------------------------------
echo [8/8] Verifying installation...
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__, '- import OK')"
python -c "from df import enhance, init_df; print('deepfilternet (df): import OK')"
if errorlevel 1 (
    echo [ERROR] deepfilternet failed to import.
    pause
    exit /b 1
)
python -c "import PyQt6; print('PyQt6: import OK')"
echo.

echo ============================================================
echo  DeepFilterNet installation complete!
echo.
echo  PyTorch variant : !TORCH_VARIANT!
echo  Venv location   : !VENV_DIR!
echo.
echo  Next steps:
echo    1. Run start.bat from the application root
echo    2. Select "venv_deepfilternet" from the menu
echo    3. Click "Download Model" to fetch DeepFilterNet2 ^& 3 ^(~2 MB^)
echo    4. Drop an audio or video file and click "Enhance Audio"
echo.
echo  Note: ffmpeg is required for video files and MP3/AAC input
echo        Download: https://ffmpeg.org/download.html
echo ============================================================
echo.
pause