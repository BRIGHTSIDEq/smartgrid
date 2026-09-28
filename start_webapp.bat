@echo off
rem Запуск веб-приложения Smart Grid: http://127.0.0.1:8050/
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -m webapp --open
