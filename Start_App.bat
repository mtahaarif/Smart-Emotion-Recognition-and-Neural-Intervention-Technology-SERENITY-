@echo off
cd /d "%~dp0"

echo ========================================
echo   SERENITY AI - STARTUP SCRIPT
echo ========================================
echo.

:: Check if backend folder exists
if not exist "backend" (
    echo ERROR: backend folder not found!
    echo Please organize your files first.
    pause
    exit /b
)

:: Check if frontend folder exists
if not exist "frontend" (
    echo ERROR: frontend folder not found!
    echo Please organize your files first.
    pause
    exit /b
)

echo [1/3] Starting Backend Server...
echo.
set "PYTHON_CMD=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"

:: Edge-optimized defaults (can be overridden by existing environment variables)
set "SERENITY_EDGE_OPTIMIZED_MODE=true"
set "SERENITY_LAZY_RUNTIME_INIT=true"
set "SERENITY_WHISPER_PRELOAD_ENABLED=false"
set "SERENITY_WHISPER_CPU_THREADS=2"
set "SERENITY_CLOUD_LLM_WARMUP_ENABLED=false"
set "SERENITY_TTS_WARMUP_ENABLED=false"
set "SERENITY_SER_TFLITE_THREADS=2"
set "SERENITY_FER_TFLITE_THREADS=2"
set "SERENITY_FER_MAX_FRAME_SIDE=640"
set "SERENITY_STREAM_TOKEN_DELTA=true"
set "SERENITY_STREAM_TTS_SENTENCE_AUDIO=true"
set "SERENITY_STREAM_TTS_FINAL_TEXT_ONLY=false"
set "SERENITY_STREAM_QUEUE_WAIT_SECONDS=0.015"
set "SERENITY_CLOUD_LLM_CONNECT_TIMEOUT_SECONDS=4"
set "SERENITY_CLOUD_LLM_TIMEOUT_SECONDS=12"
set "SERENITY_CLOUD_LLM_FAILURE_THRESHOLD=2"
set "SERENITY_CLOUD_LLM_COOLDOWN_SECONDS=30"

start "Serenity Backend" cmd /k "cd /d \"%~dp0\" && \"%PYTHON_CMD%\" -m uvicorn backend.main:app --host 127.0.0.1 --port 5000"

timeout /t 3 /nobreak >nul

echo [2/3] Starting Frontend Development Server...
echo.
start "Serenity Frontend" cmd /k "cd frontend && npm run dev"

echo [3/3] Waiting for servers to initialize...
timeout /t 8 /nobreak >nul

echo.
echo ========================================
echo   LAUNCHING APPLICATION
echo ========================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:5173
echo.

start http://localhost:5173

echo.
echo Application launched successfully!
echo Close this window after shutting down servers.
echo.
pause
