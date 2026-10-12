@echo off
setlocal

echo =======================================================
echo   ML Pipeline Telegram Bot - Setup
echo =======================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo [4/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Package installation failed. See above for details.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo   Setup complete!
echo =======================================================
echo.
echo NEXT STEPS:
echo   1. Copy .env.example to .env:
echo      copy .env.example .env
echo.
echo   2. Edit .env and fill in:
echo      TELEGRAM_BOT_TOKEN = your Telegram bot token
echo      GROQ_API_KEY       = your Groq API key
echo.
echo   3. Run the bot:
echo      venv\Scripts\activate
echo      python main.py
echo.
echo =======================================================
pause
