@echo off
:: MoodWave Launcher Wrapper
:: Double-click this file to launch the full stack.
:: It runs the PowerShell script with execution policy bypass.
::
:: If you see "conda environment not found" errors, make sure you
:: installed dependencies in the CS330_v2 environment:
::   conda activate CS330_v2
::   pip install -r backend/requirements.txt
::   pip install python-osc websockets opencv-python

echo Launching MoodWave...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_moodwave.ps1"
