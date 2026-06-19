#!/bin/bash
# ============================================================
# LinkedIn Auto Posting Bot - Quick Start Script (Linux/Mac)
# ============================================================

echo ""
echo "======================================================"
echo "🤖 LinkedIn Auto Posting Telegram Bot - Quick Start"
echo "======================================================"
echo ""

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Check dependencies
echo "📋 Checking dependencies..."
python3 -c "import telegram; print('✅ python-telegram-bot installed')" 2>/dev/null || {
    echo "❌ Dependencies missing, installing..."
    pip install -r requirements.txt
}

echo ""
echo "============================================================"
echo "✨ Everything is ready! Choose what to do:"
echo "============================================================"
echo ""
echo "Option 1: Start Backend API"
echo "  Type: 1"
echo ""
echo "Option 2: Start Telegram Bot"
echo "  Type: 2"
echo ""
echo "Option 3: Start Both (recommended - opens 2 terminals)"
echo "  Type: 3"
echo ""
echo "Option 4: Exit"
echo "  Type: 4"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Backend API..."
        echo "📍 Access at: http://localhost:8000"
        echo "📖 API Docs at: http://localhost:8000/docs"
        echo ""
        python3 start.py
        ;;
    2)
        echo ""
        echo "🤖 Starting Telegram Bot..."
        echo "📍 Bot: @borno_posting_bot"
        echo "📍 URL: t.me/borno_posting_bot"
        echo ""
        python3 start_telegram_bot.py
        ;;
    3)
        echo ""
        echo "🚀 Starting Both Services..."
        echo ""
        
        # Check which terminal emulator to use
        if command -v gnome-terminal &> /dev/null; then
            echo "Opening Backend API in new window..."
            gnome-terminal -- bash -c "cd '$(pwd)' && source venv/bin/activate && python3 start.py; exec bash"
            sleep 2
            echo "Opening Telegram Bot in new window..."
            gnome-terminal -- bash -c "cd '$(pwd)' && source venv/bin/activate && python3 start_telegram_bot.py; exec bash"
        elif command -v xterm &> /dev/null; then
            echo "Opening Backend API in new window..."
            xterm -e "cd '$(pwd)' && source venv/bin/activate && python3 start.py" &
            sleep 2
            echo "Opening Telegram Bot in new window..."
            xterm -e "cd '$(pwd)' && source venv/bin/activate && python3 start_telegram_bot.py" &
        elif command -v osascript &> /dev/null; then
            # macOS
            echo "Opening Backend API in new window..."
            osascript -e "tell app \"Terminal\" to do script \"cd '$(pwd)' && source venv/bin/activate && python3 start.py\""
            sleep 2
            echo "Opening Telegram Bot in new window..."
            osascript -e "tell app \"Terminal\" to do script \"cd '$(pwd)' && source venv/bin/activate && python3 start_telegram_bot.py\""
        else
            echo "Please open two terminals manually and run:"
            echo "Terminal 1: python3 start.py"
            echo "Terminal 2: python3 start_telegram_bot.py"
            exit 1
        fi
        
        echo ""
        echo "✅ Both services started!"
        echo ""
        echo "📊 Backend API: http://localhost:8000"
        echo "🤖 Telegram Bot: @borno_posting_bot"
        echo ""
        ;;
    4)
        echo ""
        echo "Goodbye! 👋"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Exiting..."
        exit 1
        ;;
esac
