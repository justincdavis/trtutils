@echo off
setlocal enabledelayedexpansion

python3 -m pytest --log-cli-level=WARNING -rP tests/ || exit /b 1

endlocal