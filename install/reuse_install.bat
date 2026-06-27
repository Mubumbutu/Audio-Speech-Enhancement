@echo off
setlocal EnableDelayedExpansion
title RE-USE Speech Enhancement - Installer
color 0A

echo.
echo ============================================================
echo  RE-USE Speech Enhancement - Dependency Installer
echo  NVIDIA RE-USE : Multilingual Universal Speech Enhancement
echo  Requires: NVIDIA GPU with CUDA
echo ============================================================
echo.

set ROOT_DIR=%~dp0..
set VENV_DIR=%ROOT_DIR%\venvs\venv_reuse
set FIX_DIR=%ROOT_DIR%\fix
set REQ_FILE=%~dp0reuse_requirements.txt

:: ----------------------------------------------------------------
:: [1/7] Check Python
:: ----------------------------------------------------------------
echo [1/7] Checking Python...
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
:: [2/7] Check required files
:: ----------------------------------------------------------------
echo [2/7] Checking required files...
if not exist "%REQ_FILE%" (
    echo [ERROR] reuse_requirements.txt not found at: %REQ_FILE%
    pause
    exit /b 1
)
if not exist "%FIX_DIR%\mamba_ssm_shim.py" (
    echo [ERROR] mamba_ssm_shim.py not found at: %FIX_DIR%\mamba_ssm_shim.py
    pause
    exit /b 1
)
echo [OK] All required files found.
echo.

:: ----------------------------------------------------------------
:: [3/7] Create virtual environment
:: ----------------------------------------------------------------
echo [3/7] Setting up virtual environment...
if exist "%VENV_DIR%" (
    echo [INFO] Removing existing venv_reuse...
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
:: [4/7] Detect NVIDIA GPU and select CUDA version
:: ----------------------------------------------------------------
echo [4/7] Detecting NVIDIA GPU...
set GPU_FOUND=0
set DRIVER_MAJOR=0

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    set GPU_FOUND=1
    nvidia-smi --query-gpu=driver_version --format=csv,noheader > "%TEMP%\reuse_gpu_driver.txt" 2>nul
    for /f "tokens=1 delims=." %%a in (%TEMP%\reuse_gpu_driver.txt) do (
        set DRIVER_MAJOR=%%a
        goto :REUSE_DRIVER_DONE
    )
)
:REUSE_DRIVER_DONE
del "%TEMP%\reuse_gpu_driver.txt" >nul 2>&1

if !GPU_FOUND! EQU 0 (
    echo [ERROR] No NVIDIA GPU detected.
    echo         RE-USE requires an NVIDIA GPU with CUDA.
    echo         CPU-only mode is not supported.
    pause
    exit /b 1
)

echo [OK] NVIDIA GPU detected. Driver major version: !DRIVER_MAJOR!

if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
    goto :REUSE_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
    goto :REUSE_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
    goto :REUSE_START_INSTALL
)

echo [ERROR] NVIDIA driver version !DRIVER_MAJOR! is too old ^(minimum: 450^).
echo         Please update your GPU drivers from: https://www.nvidia.com/drivers
pause
exit /b 1

:REUSE_START_INSTALL
echo [INFO] Selected CUDA variant: !TORCH_VARIANT!
echo.

:: ----------------------------------------------------------------
:: [5/7] Install PyTorch CUDA and requirements
:: ----------------------------------------------------------------
echo [5/7] Upgrading pip and setuptools...
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

echo Installing dependencies from reuse_requirements.txt...
python -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
    echo [WARNING] Some packages may have failed. Check the output above.
) else (
    echo [OK] All requirements installed.
)
echo.

echo Ensuring safetensors ^(huggingface_hub fix^)...
python -m pip install "safetensors>=0.4.3" --quiet
if errorlevel 1 (
    echo [ERROR] safetensors installation failed.
    pause
    exit /b 1
)
echo [OK] safetensors installed.
echo.

echo Restoring PyTorch CUDA ^(demucs overwrite protection^)...
python -m pip install torch torchaudio torchvision --index-url !TORCH_URL! --force-reinstall --no-deps
if errorlevel 1 (
    echo [ERROR] PyTorch restore failed.
    pause
    exit /b 1
)
echo [OK] PyTorch CUDA restored: !TORCH_VARIANT!
echo.

:: ----------------------------------------------------------------
:: [6/7] Install mamba-ssm pure-PyTorch shim
:: ----------------------------------------------------------------
echo [6/7] Installing mamba-ssm pure-PyTorch shim...
echo        ^(No CUDA Toolkit or MSVC required^)

set SHIM_DIR=%VENV_DIR%\Lib\site-packages\mamba_ssm
if not exist "!SHIM_DIR!" mkdir "!SHIM_DIR!"

copy /Y "%FIX_DIR%\mamba_ssm_shim.py" "!SHIM_DIR!\__init__.py" >nul
if errorlevel 1 (
    echo [ERROR] Could not copy shim to !SHIM_DIR!
    pause
    exit /b 1
)
echo [OK] mamba_ssm shim installed.
echo.

:: ----------------------------------------------------------------
:: [7/7] Verify installation
:: ----------------------------------------------------------------
echo [7/7] Verifying installation...
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import mamba_ssm; print('mamba_ssm shim: import OK')"
if errorlevel 1 (
    echo [ERROR] mamba_ssm shim failed to import.
    pause
    exit /b 1
)
python -c "import safetensors; print('safetensors:', safetensors.__version__)"
if errorlevel 1 (
    echo [ERROR] safetensors failed to import.
    pause
    exit /b 1
)
python -c "import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__)"
if errorlevel 1 (
    echo [ERROR] huggingface_hub failed to import.
    pause
    exit /b 1
)
python -c "import PyQt6; print('PyQt6: import OK')"
echo.

echo ============================================================
echo  RE-USE installation complete!
echo.
echo  PyTorch variant : !TORCH_VARIANT!
echo  Venv location   : %VENV_DIR%
echo.
echo  Next steps:
echo    1. Run start.bat from the application root
echo    2. Select "RE-USE Speech Enhancement" from the menu
echo    3. Click "Download Model" to fetch nvidia/RE-USE ^(~100 MB^)
echo    4. Drop an audio or video file and click "Enhance Audio"
echo.
echo  Note: ffmpeg is required for video files and MP3/AAC input
echo        Download: https://ffmpeg.org/download.html
echo ============================================================
echo.
pause
