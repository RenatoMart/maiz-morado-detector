@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Revisando e instalando lo que necesita el proyecto...
echo.
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 instalar_dependencias.py %*
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python instalar_dependencias.py %*
    ) else (
        echo No se encontro Python en este equipo.
        echo Descargalo de https://www.python.org/downloads/ y, al instalarlo,
        echo marca la casilla "Add python.exe to PATH". Luego vuelve a abrir este archivo.
    )
)
echo.
pause
