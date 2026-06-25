@echo off
REM Reset the BGH Red Thread AI companion server's local data.
REM Deletes the "data" folder (chroma_db, world_graph.json) next to this script,
REM so the next launch starts from an empty database.
REM Your Foundry world data is NOT touched - this only clears the companion
REM server's local vector index/graph. Use this to re-test a clean install.

echo Stopping companion server if running...
taskkill /F /IM companion-server.exe >nul 2>&1

set "DATA=%~dp0data"
if exist "%DATA%" (
    echo Deleting "%DATA%" ...
    rmdir /S /Q "%DATA%"
    echo Done. An empty data folder will be created on next launch.
) else (
    echo No data folder found next to this script. Nothing to reset.
)
echo.
pause
