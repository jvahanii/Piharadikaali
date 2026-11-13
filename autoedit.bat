@echo off
rem Drag .360 files or a session folder onto this file, or run:  autoedit.bat "C:\path\to\clips" [options]
"%USERPROFILE%\autoreframe-env\Scripts\python.exe" "%~dp0autoedit.py" %*
echo.
pause
