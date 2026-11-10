@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if exist ".venv_windows\Scripts\python.exe" set PY=.venv_windows\Scripts\python.exe
if "%PY%"=="" (
    echo Todavia no esta instalado el entorno. Abre primero "instalar_en_windows.bat".
    pause
    exit /b 1
)
"%PY%" -m app_escritorio.app
if errorlevel 1 pause
