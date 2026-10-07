@echo off
echo ====================================================================
echo Khoi chay ASR FastAPI Web Demo (Tailwind CSS, Dark/Light Mode)...
echo ====================================================================
call conda activate asr_env
cd /d "%~dp0\.."
start http://127.0.0.1:8000
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
pause
