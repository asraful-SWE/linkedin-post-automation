@echo off
REM ============================================================
REM LinkedIn Auto Posting Bot - Quick Start Script (Windows)
REM ============================================================

echo.
echo ======================================================
echo 🤖 LinkedIn Auto Posting Telegram Bot - Quick Start
echo ======================================================
echo.

cd /d e:\LinkedInAutoPosting\linkedin_ai_poster

REM Check if virtual environment exists
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo.
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

echo ✅ Virtual environment activated
echo.

REM Check dependencies
echo 📋 Checking dependencies...
python -c "import telegram; print('✅ python-telegram-bot installed')" 2>nul || (
    echo ❌ Dependencies missing, installing...
    pip install -r requirements.txt
)

echo.
echo ============================================================
echo ✨ Everything is ready! Choose what to do:
echo ============================================================
echo.
echo Option 1: Start Backend API
echo   Type: 1
echo.
echo Option 2: Start Telegram Bot
echo   Type: 2
echo.
echo Option 3: Start Both (recommended - opens 2 windows)
echo   Type: 3
echo.
echo Option 4: Exit
echo   Type: 4
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting Backend API...
    echo 📍 Access at: http://localhost:8000
    echo 📖 API Docs at: http://localhost:8000/docs
    echo.
    python start.py
) else if "%choice%"=="2" (
    echo.
    echo 🤖 Starting Telegram Bot...
    echo 📍 Bot: @borno_posting_bot
    echo 📍 URL: t.me/borno_posting_bot
    echo.
    python start_telegram_bot.py
) else if "%choice%"=="3" (
    echo.
    echo 🚀 Starting Both Services...
    echo.
    echo Opening Backend API in new window...
    start cmd /k "cd /d e:\LinkedInAutoPosting\linkedin_ai_poster && call venv\Scripts\activate.bat && python start.py"
    timeout /t 2 /nobreak
    echo.
    echo Opening Telegram Bot in new window...
    start cmd /k "cd /d e:\LinkedInAutoPosting\linkedin_ai_poster && call venv\Scripts\activate.bat && python start_telegram_bot.py"
    echo.
    echo ✅ Both services started in separate windows!
    echo.
    echo 📊 Backend API: http://localhost:8000
    echo 🤖 Telegram Bot: @borno_posting_bot
    echo.
    pause
) else if "%choice%"=="4" (
    echo. 
    echo Goodbye! 👋
    exit /b 0
) else (
    echo ❌ Invalid choice. Exiting...
    exit /b 1
)

pause
