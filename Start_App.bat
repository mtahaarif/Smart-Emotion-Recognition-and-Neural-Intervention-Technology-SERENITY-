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
start "Serenity Backend" cmd /k "cd backend && python app.py"

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
