@echo off
setlocal EnableDelayedExpansion
title AudioSR Super-Resolution - Installer
color 0A

echo.
echo ============================================================
echo  AudioSR Super-Resolution - Dependency Installer
echo  Requires: NVIDIA GPU with CUDA
echo ============================================================
echo.

set ROOT_DIR=%~dp0..
set VENV_DIR=%ROOT_DIR%\venvs\venv_audiosr
set REQ_FILE=%~dp0audiosr_requirements.txt

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
    echo [ERROR] audiosr_requirements.txt not found at: %REQ_FILE%
    pause
    exit /b 1
)
echo [OK] audiosr_requirements.txt found.
echo.

:: ----------------------------------------------------------------
:: [3/8] Create virtual environment
:: ----------------------------------------------------------------
echo [3/8] Setting up virtual environment...
if exist "%VENV_DIR%" (
    echo [INFO] Removing existing venv_audiosr...
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
    nvidia-smi --query-gpu=driver_version --format=csv,noheader > "%TEMP%\audiosr_gpu_driver.txt" 2>nul
    for /f "tokens=1 delims=." %%a in (%TEMP%\audiosr_gpu_driver.txt) do (
        set DRIVER_MAJOR=%%a
        goto :ASR_DRIVER_DONE
    )
)
:ASR_DRIVER_DONE
del "%TEMP%\audiosr_gpu_driver.txt" >nul 2>&1

if !GPU_FOUND! EQU 0 (
    echo [ERROR] No NVIDIA GPU detected.
    echo         AudioSR requires an NVIDIA GPU with CUDA.
    echo         CPU-only mode is not supported.
    pause
    exit /b 1
)

echo [OK] NVIDIA GPU detected. Driver major version: !DRIVER_MAJOR!

if !DRIVER_MAJOR! GEQ 550 (
    set TORCH_URL=https://download.pytorch.org/whl/cu124
    set TORCH_VARIANT=GPU CUDA 12.4
    goto :ASR_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 525 (
    set TORCH_URL=https://download.pytorch.org/whl/cu121
    set TORCH_VARIANT=GPU CUDA 12.1
    goto :ASR_START_INSTALL
)
if !DRIVER_MAJOR! GEQ 450 (
    set TORCH_URL=https://download.pytorch.org/whl/cu118
    set TORCH_VARIANT=GPU CUDA 11.8
    goto :ASR_START_INSTALL
)

echo [ERROR] NVIDIA driver version !DRIVER_MAJOR! is too old ^(minimum: 450^).
echo         Please update your GPU drivers from: https://www.nvidia.com/drivers
pause
exit /b 1

:ASR_START_INSTALL
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
echo [6/8] Installing dependencies from audiosr_requirements.txt...
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
:: [7/8] Install audiosr and its fragile pinned dependencies
::
::  Problem 1: audiosr metadata falsely requires numpy<=1.23.5
::             -> install with --no-deps, then add deps manually
::
::  Problem 2: audiosr pins transformers==4.30.2, which requires
::             tokenizers<0.14. Those old tokenizers versions have
::             NO pre-built wheel for Python 3.12 on Windows, so pip
::             tries to build from source -> fails (needs Rust compiler).
::
::  Fix: install tokenizers==0.15.2 (first cp312-compatible binary wheel)
::       and transformers==4.35.2 (first version that accepts tokenizers>=0.15).
::       audiosr only uses RobertaTokenizer / AutoTokenizer - the API is
::       identical across these versions so it works without modification.
::
::  Problem 3: timm/phonemizer/etc can pull in CPU torch from PyPI.
::             -> restore PyTorch CUDA again after this block.
:: ----------------------------------------------------------------
echo [7/8] Installing audiosr and pinned dependencies...

echo      Step 1/4 : Installing tokenizers==0.15.2 ^(pre-built binary wheel^)...
python -m pip install tokenizers==0.15.2 --only-binary :all: --quiet
if errorlevel 1 (
    echo [WARNING] tokenizers==0.15.2 failed. Trying 0.15.1 ...
    python -m pip install tokenizers==0.15.1 --only-binary :all: --quiet
    if errorlevel 1 (
        echo [ERROR] Could not install tokenizers. audiosr may not work.
    ) else (
        echo [OK] tokenizers 0.15.1 installed.
    )
) else (
    echo [OK] tokenizers 0.15.2 installed.
)

:: 4.35.2 is the first transformers release that supports tokenizers>=0.15.
:: Every transformers 4.x release hard-requires huggingface_hub<1.0 in its
:: own internal version check - it raises ImportError at import time
:: otherwise. Install --no-deps so pip cannot touch huggingface-hub or
:: tokenizers here; the correct huggingface_hub version is force-set below.
echo      Step 2/4 : Installing transformers==4.35.2 ^(--no-deps^)...
python -m pip install transformers==4.35.2 --no-deps --quiet
if errorlevel 1 (
    echo [WARNING] transformers==4.35.2 failed. Trying 4.36.2 ...
    python -m pip install transformers==4.36.2 --no-deps --quiet
    if errorlevel 1 (
        echo [ERROR] transformers installation failed. audiosr may not work.
    ) else (
        echo [OK] transformers 4.36.2 installed.
    )
) else (
    echo [OK] transformers 4.35.2 installed.
)

:: Installed WITHOUT --no-deps on purpose: several of these packages have
:: real runtime dependencies that audiosr needs at import time
:: (phonemizer -> segments, progressbar2 -> python-utils, etc.).
echo      Step 3/4 : Installing remaining audiosr runtime deps...
python -m pip install ^
    tqdm ^
    scipy ^
    pandas ^
    unidecode ^
    phonemizer ^
    ftfy ^
    timm ^
    progressbar2 ^
    python-utils ^
    torchlibrosa>=0.0.9 ^
    regex ^
    sacremoses ^
    sentencepiece ^
    --quiet
if errorlevel 1 (
    echo [WARNING] Some audiosr runtime deps may have failed. Check the output above.
) else (
    echo [OK] audiosr runtime deps installed.
)

:: transformers 4.x hard-requires huggingface_hub<1.0 at import time.
:: Force-pin hub BELOW 1.0 here using --no-deps so pip does not try to
:: "fix" this by reinstalling/downgrading other already-installed packages.
python -m pip install "huggingface_hub>=0.23.0,<1.0" --upgrade --no-deps --quiet
if errorlevel 1 (
    echo [WARNING] Could not re-pin huggingface_hub. Check version manually.
) else (
    echo [OK] huggingface_hub pinned to ^<1.0 for transformers compatibility.
)

echo      Step 4/4 : Installing audiosr==0.0.7 ^(--no-deps^)...
python -m pip install audiosr==0.0.7 --no-deps
if errorlevel 1 (
    echo [ERROR] audiosr installation failed.
    pause
    exit /b 1
)
echo [OK] audiosr installed.
echo.

echo Restoring PyTorch CUDA after audiosr deps...
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
python -c "import torchvision; print('torchvision:', torchvision.__version__, '- import OK')"
python -c "import importlib.metadata as m; print('huggingface_hub:', m.version('huggingface_hub'))"
python -c "import transformers; print('transformers:', transformers.__version__, '- import OK')"
if errorlevel 1 (
    echo [ERROR] transformers failed to import - huggingface_hub pin did not take effect.
    pause
    exit /b 1
)
python -c "import audiosr; print('audiosr: import OK')"
if errorlevel 1 (
    echo [ERROR] audiosr failed to import.
    pause
    exit /b 1
)
python -c "import PyQt6; print('PyQt6: import OK')"
echo.

echo ============================================================
echo  AudioSR installation complete!
echo.
echo  PyTorch variant : !TORCH_VARIANT!
echo  Venv location   : %VENV_DIR%
echo.
echo  Next steps:
echo    1. Run start.bat from the application root
echo    2. Select "AudioSR Super-Resolution" from the menu
echo    3. Drop an audio or video file and click "Enhance Audio"
echo.
echo  Note: ffmpeg is required for video files and MP3/AAC input
echo        Download: https://ffmpeg.org/download.html
echo ============================================================
echo.
pause
