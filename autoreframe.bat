@echo off
rem Drag .360 files or a folder onto this file, or run:  autoreframe.bat "C:\path\to\clips" [options]
"%USERPROFILE%\autoreframe-env\Scripts\python.exe" "%~dp0autoreframe.py" %*
echo.
pause
