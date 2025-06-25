@echo off
setlocal enabledelayedexpansion

:: Set path to msolve binary and tests directory
set "MSOLVE=C:\msys64\mingw64\bin\msolve.exe"
set "TESTDIR=C:\msys64\mingw64\bin\tests"
set "LOGFILE=msolve_timing_log.txt"

:: Initialize timing log file
echo Msolve Timing Log - %date% %time% > "%LOGFILE%"
echo ================================================ >> "%LOGFILE%"

echo Running msolve on all .ms files in %TESTDIR%
echo Timing results will be saved to: %LOGFILE%
echo.

:: Get start time for total execution
set "TOTAL_START_TIME=%time%"

:: Initialize file counter
set "FILE_COUNT=0"

:: Loop through all .ms files
for %%F in ("%TESTDIR%\*.ms") do (
    set /a FILE_COUNT+=1
    set "FILENAME=%%~nxF"
    
    echo Running: %MSOLVE% -f "%%F"
    echo File !FILE_COUNT!: !FILENAME!
    
    :: Get start time for this file
    set "START_TIME=!time!"
    
    :: Run msolve
    %MSOLVE% -f "%%F"
    set "EXIT_CODE=!errorlevel!"
    
    :: Get end time for this file
    set "END_TIME=!time!"
    
    :: Calculate elapsed time for this file
    call :calculate_time_diff "!START_TIME!" "!END_TIME!" ELAPSED_TIME
    
    if !EXIT_CODE! EQU 0 (
        echo Success: !FILENAME! ^(Time: !ELAPSED_TIME!^)
        echo !FILENAME!: !ELAPSED_TIME! >> "%LOGFILE%"
    ) else (
        echo ERROR running file: !FILENAME! ^(Time: !ELAPSED_TIME!^)
        echo !FILENAME!: !ELAPSED_TIME! ^(ERROR^) >> "%LOGFILE%"
    )
    
    echo --------------------------------
)

:: Get end time for total execution
set "TOTAL_END_TIME=%time%"

:: Calculate total elapsed time
call :calculate_time_diff "%TOTAL_START_TIME%" "%TOTAL_END_TIME%" TOTAL_ELAPSED

echo.
echo ================================================
echo SUMMARY:
echo Total files processed: %FILE_COUNT%
echo Total execution time: %TOTAL_ELAPSED%
echo Timing log saved to: %LOGFILE%
echo ================================================

:: Append summary to log file
echo. >> "%LOGFILE%"
echo ================================================ >> "%LOGFILE%"
echo SUMMARY: >> "%LOGFILE%"
echo Total files processed: %FILE_COUNT% >> "%LOGFILE%"
echo Total execution time: %TOTAL_ELAPSED% >> "%LOGFILE%"
echo End time: %date% %time% >> "%LOGFILE%"
echo ================================================ >> "%LOGFILE%"

echo Done running all .ms files.
pause
goto :eof

:: Function to calculate time difference
:calculate_time_diff
setlocal
set "start_time=%~1"
set "end_time=%~2"

:: Convert start time to centiseconds
call :time_to_centiseconds "%start_time%" start_cs

:: Convert end time to centiseconds
call :time_to_centiseconds "%end_time%" end_cs

:: Calculate difference
set /a diff_cs=end_cs-start_cs

:: Handle day rollover (if end time is less than start time)
if !diff_cs! LSS 0 (
    set /a diff_cs=diff_cs+8640000
)

:: Convert back to time format
call :centiseconds_to_time !diff_cs! result_time

endlocal & set "%~3=%result_time%"
goto :eof

:: Function to convert time to centiseconds
:time_to_centiseconds
setlocal
set "time_str=%~1"

:: Parse time components
for /f "tokens=1-4 delims=:." %%a in ("%time_str%") do (
    set "hours=%%a"
    set "minutes=%%b"
    set "seconds=%%c"
    set "centisecs=%%d"
)

:: Remove leading spaces/zeros
set /a hours=1%hours%-100
set /a minutes=1%minutes%-100
set /a seconds=1%seconds%-100
if defined centisecs (
    set /a centisecs=1%centisecs%-100
) else (
    set centisecs=0
)

:: Convert to total centiseconds
set /a total_cs=hours*360000+minutes*6000+seconds*100+centisecs

endlocal & set "%~2=%total_cs%"
goto :eof

:: Function to convert centiseconds back to time format
:centiseconds_to_time
setlocal
set /a input_cs=%~1

set /a hours=input_cs/360000
set /a remainder=input_cs%%360000
set /a minutes=remainder/6000
set /a remainder=remainder%%6000
set /a seconds=remainder/100
set /a centisecs=remainder%%100

:: Format with leading zeros
if %hours% LSS 10 set "hours=0%hours%"
if %minutes% LSS 10 set "minutes=0%minutes%"
if %seconds% LSS 10 set "seconds=0%seconds%"
if %centisecs% LSS 10 set "centisecs=0%centisecs%"

set "formatted_time=%hours%:%minutes%:%seconds%.%centisecs%"

endlocal & set "%~2=%formatted_time%"
goto :eof
