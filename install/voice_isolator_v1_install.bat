@echo off
setlocal EnableDelayedExpansion
title Inworld Voice Isolator v1 - Installer
color 0A
echo.
echo ============================================================
echo  Inworld Voice Isolator v1 - Dependency Installer
echo  Apache-2.0 48 kHz voice isolation for noisy recordings
echo  Offline model ^(64M, best quality^) + realtime model ^(58M^)
echo  Runs on NVIDIA CUDA or CPU, ~490 MB weights
echo ============================================================
echo.
set ROOT_DIR=%~dp0..
set VENV_DIR=%ROOT_DIR%\venvs\venv_voice_isolator_v1
set REQ_FILE=%~dp0voice_isolator_v1_requirements.txt
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
echo [ERROR] voice_isolator_v1_requirements.txt not found at: %REQ_FILE%
pause
exit /b 1
)
echo [OK] voice_isolator_v1_requirements.txt found.
echo.
:: ----------------------------------------------------------------
:: [3/7] Create virtual environment
:: ----------------------------------------------------------------
echo [3/7] Setting up virtual environment...
if exist "%VENV_DIR%" (
echo [INFO] Removing existing venv_voice_isolator_v1...
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
:: [4/7] Detect NVIDIA GPU and select PyTorch variant
:: ----------------------------------------------------------------
echo [4/7] Detecting NVIDIA GPU...
set GPU_FOUND=0
set DRIVER_MAJOR=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
set GPU_FOUND=1
nvidia-smi --query-gpu=driver_version --format=csv,noheader > "%TEMP%\voice_isolator_gpu_driver.txt" 2>nul
for /f "tokens=1 delims=." %%a in (%TEMP%\voice_isolator_gpu_driver.txt) do (
set DRIVER_MAJOR=%%a
goto :VI_DRIVER_DONE
)
)
:VI_DRIVER_DONE
del "%TEMP%\voice_isolator_gpu_driver.txt" >nul 2>&1
if !GPU_FOUND! EQU 0 (
echo [INFO] No NVIDIA GPU detected. Installing CPU-only PyTorch.
echo        Voice Isolator runs on CPU as well, though slower.
set TORCH_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU only
goto :VI_START_INSTALL
)
echo [OK] NVIDIA GPU detected. Driver major version: !DRIVER_MAJOR!
if !DRIVER_MAJOR! GEQ 550 (
set TORCH_URL=https://download.pytorch.org/whl/cu124
set TORCH_VARIANT=GPU CUDA 12.4
goto :VI_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 525 (
set TORCH_URL=https://download.pytorch.org/whl/cu121
set TORCH_VARIANT=GPU CUDA 12.1
goto :VI_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 450 (
set TORCH_URL=https://download.pytorch.org/whl/cu118
set TORCH_VARIANT=GPU CUDA 11.8
goto :VI_START_INSTALL
)
echo [WARNING] NVIDIA driver version !DRIVER_MAJOR! is too old ^(minimum: 450^).
echo           Falling back to CPU-only installation.
echo           Please update your GPU drivers from: https://www.nvidia.com/drivers
set TORCH_URL=https://download.pytorch.org/whl/cpu
set TORCH_VARIANT=CPU only
:VI_START_INSTALL
echo [INFO] Selected PyTorch variant: !TORCH_VARIANT!
echo.
:: ----------------------------------------------------------------
:: [5/7] Upgrade pip and install PyTorch + requirements
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
echo Installing dependencies from voice_isolator_v1_requirements.txt...
python -m pip install -r "%REQ_FILE%"
if errorlevel 1 (
echo [WARNING] Some packages may have failed. Check the output above.
) else (
echo [OK] All requirements installed.
)
echo.
echo Restoring PyTorch ^(!TORCH_VARIANT!^)...
python -m pip install torch torchaudio --index-url !TORCH_URL! --force-reinstall --no-deps
if errorlevel 1 (
echo [ERROR] PyTorch restore failed.
pause
exit /b 1
)
echo [OK] PyTorch restored: !TORCH_VARIANT!
echo.
:: ----------------------------------------------------------------
:: [6/7] Install Inworld Voice Isolator
:: ----------------------------------------------------------------
echo [6/7] Installing Inworld Voice Isolator...
echo        NOTE: The [fast] extra ^(CUDA Mamba kernels^) is Linux-only.
echo              On Windows the built-in pure-PyTorch scan is used.
python -m pip install inworld-voice-isolation
if errorlevel 1 (
echo [ERROR] inworld-voice-isolation installation failed.
echo         Check your internet access and Python version ^(3.10+ required^).
pause
exit /b 1
)
echo [OK] inworld-voice-isolation installed.
echo.
echo Restoring PyTorch ^(!TORCH_VARIANT!^) after package install...
python -m pip install torch torchaudio --index-url !TORCH_URL! --force-reinstall --no-deps --quiet
if errorlevel 1 (
echo [WARNING] PyTorch re-restore failed. Check GPU functionality.
) else (
echo [OK] PyTorch confirmed: !TORCH_VARIANT!
)
echo.
:: ----------------------------------------------------------------
:: [7/7] Verify installation
:: ----------------------------------------------------------------
echo [7/7] Verifying installation...
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__, '- import OK')"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import soundfile; print('soundfile: import OK')"
python -c "import safetensors; print('safetensors: import OK')"
python -c "import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__, '- import OK')"
python -c "import PyQt6; print('PyQt6: import OK')"
python -c "from inworld import voice_isolation; print('voice_isolation:', voice_isolation.__version__, '- import OK')"
if errorlevel 1 (
echo [ERROR] inworld.voice_isolation failed to import.
pause
exit /b 1
)
echo [OK] Inworld Voice Isolator: package installed successfully.
echo [INFO] Model weights will be downloaded on first use via "Download Model" button.
echo.
echo ============================================================
echo  Inworld Voice Isolator v1 installation complete!
echo.
echo  PyTorch variant : !TORCH_VARIANT!
echo  Venv location   : %VENV_DIR%
echo.
echo  Next steps:
echo    1. Run start.bat from the application root
echo    2. Select "Inworld Voice Isolator v1" from the menu
echo    3. Click "Download Model" to fetch the weights
echo       ^(~490 MB total: offline 256 MB + realtime 233 MB^)
echo    4. Drop an audio file and click "Process Queue"
echo.
echo  Modes:
echo    Offline   - voice-isolator-v1, best quality for recordings
echo    Streaming - voice-isolator-v1-realtime, look-ahead 0/20/40 ms
echo.
echo  Note: Output is 48 kHz float WAV ^(or keep original sample rate^).
echo  Note: Denoising only - it does not restore lost bandwidth.
echo  Note: "Save removed background" writes input minus voice as WAV.
echo  Note: On GPUs older than Ampere ^(GTX 10xx / RTX 20xx^) set
echo        Precision to fp32 in the app if bf16 causes issues.
echo  Note: Optional machine check after install: voice-isolate-doctor
echo  Note: ffmpeg is required for video files and MP3/AAC input
echo        Download: https://ffmpeg.org/download.html
echo ============================================================
echo.
pause