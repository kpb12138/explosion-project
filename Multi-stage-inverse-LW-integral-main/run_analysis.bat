@echo off

REM Run constraint efficiency analysis script
echo Running constraint efficiency analysis script...
python dataConstraintAnalysis.py

REM Check if the first script executed successfully
if %ERRORLEVEL% NEQ 0 (
    echo Constraint efficiency analysis script failed!
    pause
    exit /b %ERRORLEVEL%
)

echo Constraint efficiency analysis script completed!

REM Run IDT correlation generation script
echo Running IDT correlation generation script...
python correlation_IDT_gen.py

REM Check if the second script executed successfully
if %ERRORLEVEL% NEQ 0 (
    echo IDT correlation generation script failed!
    pause
    exit /b %ERRORLEVEL%
)

echo IDT correlation generation script completed!

REM All scripts executed successfully
echo All scripts completed successfully!
pause
