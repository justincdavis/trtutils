@echo off
setlocal enabledelayedexpansion

rem Set the path to the tests directory
set "test_directory=.\tests"

rem Loop through all Python files in the tests directory
for %%i in ("%test_directory%\*.py") do (
    echo Running pytest on "%%i"
    python3 -m pytest "%%i"
)

endlocal
