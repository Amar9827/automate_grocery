@echo off
cd /d "%~dp0"

if not exist logs mkdir logs

call "..\\.venv\Scripts\activate.bat"

set PYTHONIOENCODING=utf-8
python zepto_client.py --auto >> logs\checker.log 2>&1 

deactivate
