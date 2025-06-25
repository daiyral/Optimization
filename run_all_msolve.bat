@echo off
setlocal

:: Set path to msolve binary and tests directory
set "MSOLVE=C:\msys64\mingw64\bin\msolve.exe"
set "TESTDIR=C:\msys64\mingw64\bin\tests"

echo Running msolve on all .ms files in %TESTDIR%
echo.

:: Loop through all .ms files
for %%F in ("%TESTDIR%\*.ms") do (
    echo Running: %MSOLVE% -f "%%F"
    %MSOLVE% -f "%%F"
    
    if errorlevel 1 (
        echo ERROR running file: %%F
    ) else (
        echo Success: %%F
    )
    
    echo --------------------------------
)

echo Done running all .ms files.
pause
