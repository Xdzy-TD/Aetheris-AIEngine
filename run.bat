@echo off
REM Aetheris — Windows launcher for the full app (FastAPI + Ollama LLM planner).
REM Batch equivalent of deploy/cloud_demo/entrypoint.sh's default (non-GUI) path.
REM Extra args pass straight through to `run.py serve`, e.g.:
REM   run.bat --model qwen2.5:7b

if "%AETHERIS_HOST%"=="" set AETHERIS_HOST=0.0.0.0
if "%AETHERIS_PORT%"=="" set AETHERIS_PORT=8000

where ollama >nul 2>nul
if %ERRORLEVEL%==0 (
    tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
    if not %ERRORLEVEL%==0 start "ollama" /min ollama serve
) else (
    echo [run.bat] ollama not found on PATH — install from https://ollama.com/download
    echo [run.bat] or run with --no-llm for stub routing.
)

python run.py serve --host %AETHERIS_HOST% --port %AETHERIS_PORT% %*
