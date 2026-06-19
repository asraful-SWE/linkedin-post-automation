# 🤖 Telegram Bot Module

এই ফোল্ডারে আপনার **LinkedIn Auto Posting System** এর Telegram Bot implementation আছে।

## 📁 Structure

```
telegram_bot/
├── bot.py                   # Main bot entry point
├── config.py               # Configuration management
│
├── handlers/               # Command & interaction handlers
│   ├── start.py           # /start, /help, /about
│   ├── dashboard.py       # Dashboard view
│   ├── generate.py        # Post generation
│   ├── stats.py           # Analytics
│   ├── schedule.py        # Scheduling
│   ├── settings.py        # Settings
│   └── callbacks.py       # Button callbacks
│
└── services/              # Utility services
    ├── formatter.py       # Message formatting
    └── telegram_publisher.py  # Telegram API
```

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_ADMIN_ID=your_user_id
```

### 2. Run
```bash
# Backend API
python start.py

# Telegram Bot (in another terminal)
python start_telegram_bot.py
```

### 3. Use
- Find bot on Telegram: `@your_bot_name`
- Send `/start`
- Done! ✅

## 📚 Documentation

- [Complete Setup Guide](../TELEGRAM_BOT_SETUP.md)
- [Quick Start (30 sec)](../TELEGRAM_BOT_QUICKSTART.md)
- [Implementation Details](../TELEGRAM_BOT_IMPLEMENTATION.md)

## 🤖 Main Features

| Feature | Command |
|---------|---------|
| Dashboard | `/dashboard` |
| Generate Posts | `/generate` |
| Statistics | `/stats` |
| Scheduling | `/schedule` |
| Settings | `/settings` |
| Help | `/help` |

## 📞 Support

See documentation files for detailed guides and troubleshooting.

---

**Made with ❤️ for LinkedIn Auto Posting**
