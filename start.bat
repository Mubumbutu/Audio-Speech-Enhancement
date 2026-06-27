@echo off
setlocal EnableDelayedExpansion
title Audio Enhancement Studio - Launcher
color 0A

set ROOT_DIR=%~dp0
set VENVS_DIR=%ROOT_DIR%venvs
set APP_SCRIPT=%ROOT_DIR%app\app.py

set LABEL_venv_reuse=RE-USE Speech Enhancement
set LABEL_venv_audiosr=AudioSR Super-Resolution
set LABEL_venv_deepfilternet=DeepFilterNet Noise Suppression
set LABEL_venv_flashsr=FlashSR Super-Resolution
set LABEL_venv_universr=UniverSR Super-Resolution
set LABEL_venv_lavasr=LavaSR Super-Resolution
set LABEL_venv_novasr=NovaSR Super-Resolution
set LABEL_venv_flowhigh=FlowHigh Super-Resolution
set LABEL_venv_clearervoice=ClearerVoice Studio
set LABEL_venv_voicefixer=VoiceFixer Speech Restoration

if not exist "%VENVS_DIR%" (
    echo [ERROR] No venvs folder found at: %VENVS_DIR%
    echo         Run install.bat first.
    pause
    exit /b 1
)

set COUNT=0
for /d %%D in ("%VENVS_DIR%\*") do (
    if exist "%%D\Scripts\activate.bat" (
        set /a COUNT+=1
        set "VENV_!COUNT!=%%~fD"
        set "NAME_!COUNT!=%%~nxD"
    )
)

if !COUNT! EQU 0 (
    echo [ERROR] No valid virtual environments found in: %VENVS_DIR%
    echo         Run install.bat first.
    pause
    exit /b 1
)

:MENU
cls
echo.
echo ============================================================
echo  Audio Enhancement Studio - Launcher
echo ============================================================
echo.

for /l %%i in (1,1,!COUNT!) do (
    set "RAW_NAME=!NAME_%%i!"
    set "LABEL=!LABEL_%%i!"
    call set "LBL_VALUE=%%LABEL_!RAW_NAME!%%"
    if "!LBL_VALUE!"=="" (
        echo  [%%i] !RAW_NAME!
    ) else (
        echo  [%%i] !LBL_VALUE!
    )
)
echo  [0] Exit
echo.
set /p CHOICE="Select an environment to launch: "

if "%CHOICE%"=="0" exit /b 0

set VALID=0
for /l %%i in (1,1,!COUNT!) do (
    if "%CHOICE%"=="%%i" set VALID=1
)
if !VALID! EQU 0 (
    echo.
    echo [ERROR] Invalid choice.
    echo.
    pause
    goto :MENU
)

call set "SELECTED_VENV=%%VENV_%CHOICE%%%"

echo.
echo [INFO] Activating: !SELECTED_VENV!
call "!SELECTED_VENV!\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [INFO] Launching application...
echo.
python "%APP_SCRIPT%"
set APP_EXIT=%errorlevel%

if not "%APP_EXIT%"=="0" (
    echo.
    echo [ERROR] Application exited with code %APP_EXIT%.
    pause
)

exit /b %APP_EXIT%
