@echo off
REM Launch AINet desktop shell
cd /d "%~dp0.."
python -m desktop.app %*
