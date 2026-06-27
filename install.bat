@echo off
setlocal EnableDelayedExpansion
title Audio Enhancement Studio - Installer
color 0A

:MENU
cls
echo.
echo ============================================================
echo  Audio Enhancement Studio - Installer
echo  Each model is installed into its own isolated environment.
echo ============================================================
echo.
echo  [1] RE-USE Speech Enhancement
echo  [2] AudioSR Super-Resolution
echo  [3] DeepFilterNet Noise Suppression
echo  [4] FlashSR Super-Resolution
echo  [5] UniverSR Super-Resolution
echo  [6] NovaSR Super-Resolution
echo  [7] LavaSR Super-Resolution
echo  [8] FlowHigh Super-Resolution
echo  [9] ClearerVoice Studio
echo  [10] VoiceFixer Speech Restoration
echo  [0] Exit
echo.
set /p CHOICE="Select a model to install: "

if "%CHOICE%"=="1" goto :INSTALL_REUSE
if "%CHOICE%"=="2" goto :INSTALL_AUDIOSR
if "%CHOICE%"=="3" goto :INSTALL_DEEPFILTERNET
if "%CHOICE%"=="4" goto :INSTALL_FLASHSR
if "%CHOICE%"=="5" goto :INSTALL_UNIVERSR
if "%CHOICE%"=="6" goto :INSTALL_NOVASR
if "%CHOICE%"=="7" goto :INSTALL_LAVASR
if "%CHOICE%"=="8" goto :INSTALL_FLOWHIGH
if "%CHOICE%"=="9" goto :INSTALL_CLEARERVOICE
if /i "%CHOICE%"=="10" goto :INSTALL_VOICEFIXER
if "%CHOICE%"=="0" goto :END

echo.
echo [ERROR] Invalid choice.
echo.
pause
goto :MENU

:INSTALL_REUSE
echo.
call "%~dp0install\reuse_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] RE-USE installation failed.
) else (
    echo.
    echo [OK] RE-USE installation finished.
)
goto :ASK_AGAIN

:INSTALL_AUDIOSR
echo.
call "%~dp0install\audiosr_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] AudioSR installation failed.
) else (
    echo.
    echo [OK] AudioSR installation finished.
)
goto :ASK_AGAIN

:INSTALL_DEEPFILTERNET
echo.
call "%~dp0install\deepfilternet_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] DeepFilterNet installation failed.
) else (
    echo.
    echo [OK] DeepFilterNet installation finished.
)
goto :ASK_AGAIN

:INSTALL_FLASHSR
echo.
call "%~dp0install\flashsr_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] FlashSR installation failed.
) else (
    echo.
    echo [OK] FlashSR installation finished.
)
goto :ASK_AGAIN

:INSTALL_UNIVERSR
echo.
call "%~dp0install\universr_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] UniverSR installation failed.
) else (
    echo.
    echo [OK] UniverSR installation finished.
)
goto :ASK_AGAIN

:INSTALL_NOVASR
echo.
call "%~dp0install\novasr_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] NovaSR installation failed.
) else (
    echo.
    echo [OK] NovaSR installation finished.
)
goto :ASK_AGAIN

:INSTALL_LAVASR
echo.
call "%~dp0install\lavasr_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] LavaSR installation failed.
) else (
    echo.
    echo [OK] LavaSR installation finished.
)
goto :ASK_AGAIN

:INSTALL_FLOWHIGH
echo.
call "%~dp0install\flowhigh_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] FlowHigh installation failed.
) else (
    echo.
    echo [OK] FlowHigh installation finished.
)
goto :ASK_AGAIN

:INSTALL_CLEARERVOICE
echo.
call "%~dp0install\clearervoice_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] ClearerVoice installation failed.
) else (
    echo.
    echo [OK] ClearerVoice installation finished.
)
goto :ASK_AGAIN

:INSTALL_VOICEFIXER
echo.
call "%~dp0install\voicefixer_install.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] VoiceFixer installation failed.
) else (
    echo.
    echo [OK] VoiceFixer installation finished.
)
goto :ASK_AGAIN

:ASK_AGAIN
echo.
choice /C YN /M "Install another model"

if errorlevel 2 goto :END
if errorlevel 1 goto :MENU

:END
echo.
echo Done.
pause
exit /b 0
