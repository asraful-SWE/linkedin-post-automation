# 🤖 Telegram Bot Setup Guide

## প্রিরিকুইজিট (প্রয়োজনীয় জিনিসপত্র)

### ১. Telegram Bot Token পান

1. Telegram এ **@BotFather** সার্চ করুন
2. `/start` দিয়ে start করুন
3. `/newbot` command দিন
4. Bot এর নাম দিন (e.g., "LinkedInAutoPostingBot")
5. Username দিন (e.g., "linkedin_auto_posting_bot")
6. BotFather আপনাকে একটি token দেবে: `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`

**এই token আপনার `.env` ফাইলে রাখবেন:**
```
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
```

### ২. আপনার Telegram User ID পান

1. Telegram এ **@userinfobot** সার্চ করুন
2. `/start` দিন
3. এটি আপনার User ID দেবে (e.g., `123456789`)

**এটিও `.env` ফাইলে রাখবেন:**
```
TELEGRAM_ADMIN_ID=123456789
```

---

## সেটআপ স্টেপ

### ১. Environment Variables সেট করুন

[`.env`](../.env) ফাইল এ এই ভেরিয়েবলগুলো যোগ করুন:

```bash
# === TELEGRAM BOT CONFIGURATION ===
TELEGRAM_BOT_TOKEN=আপনার_টোকেন_এখানে
TELEGRAM_ADMIN_ID=আপনার_ইউজার_আইডি_এখানে

# === OTHER CONFIGURATION ===
API_URL=http://localhost:8000
DATABASE_PATH=linkedin_ai_poster.db
OPENAI_API_KEY=আপনার_অপেনএআই_কী
TIMEZONE=Asia/Dhaka
```

### ২. Dependencies ইনস্টল করুন

```bash
# নতুন dependencies যোগ করা হয়েছে
pip install -r requirements.txt

# বা শুধু telegram libraries:
pip install python-telegram-bot>=20.0
pip install aiogram>=3.0.0
```

### ৩. Backend API চালু করুন (আলাদা Terminal এ)

```bash
cd linkedin_ai_poster
python start.py

# অথবা
python -m app.main
```

এটি `http://localhost:8000` এ চলবে

### ৪. Telegram Bot চালু করুন (নতুন Terminal এ)

```bash
cd linkedin_ai_poster
python start_telegram_bot.py
```

আপনি এই আউটপুট দেখবেন:
```
============================================================
🤖 LinkedIn Auto Posting Telegram Bot
============================================================

✅ Configuration loaded:
   Bot Token: 123456:ABC-DEF124...
   Admin ID: 123456789
   API URL: http://localhost:8000
   Database: linkedin_ai_poster.db

🚀 Starting bot...
```

---

## বট ব্যবহার করা

### Telegram এ শুরু করুন

1. আপনার bot খুঁজুন (e.g., `@linkedin_auto_posting_bot`)
2. `/start` দিন
3. আপনি এই মেনু পাবেন:

```
📊 Dashboard    🤖 Generate
📈 Stats        📅 Schedule
⚙️ Settings    ❓ Help
```

### কমান্ড

| কমান্ড | বর্ণনা |
|--------|--------|
| `/start` | মূল মেনু দেখান |
| `/dashboard` | ড্যাশবোর্ড স্ট্যাটিস্টিক্স |
| `/generate` | নতুন পোস্ট জেনারেট করুন |
| `/stats` | বিস্তারিত analytics |
| `/schedule` | পরবর্তী পোস্ট দেখুন |
| `/settings` | সেটিংস কনফিগার করুন |
| `/help` | সাহায্য পান |
| `/about` | বট সম্পর্কে |

---

## ফিচার বর্ণনা

### 📊 Dashboard
- Total posts এবং engagement দেখান
- System status (Database, Scheduler, LinkedIn, AI)
- পরবর্তী পোস্ট এর সময় দেখান

### 🤖 Generate
- Topic সিলেক্ট করুন
- AI দিয়ে পোস্ট generate করুন
- Preview দেখান
- সরাসরি Publish করুন

### 📈 Stats
- Total posts, engagement
- Topic-wise performance
- Best performing topics
- Weekly/Monthly reports

### 📅 Schedule
- পরবর্তী 5টি scheduled posts
- Scheduler status
- Pause/Resume scheduling

### ⚙️ Settings
- Max posts per day
- Min hours between posts
- Timezone
- API Keys
- Test mode

---

## ট্রাবলশুটিং

### ❌ "Bot Token not set" Error

**সমাধান:**
```bash
# .env ফাইল চেক করুন
# TELEGRAM_BOT_TOKEN সেট আছে কিনা দেখুন
cat .env | grep TELEGRAM_BOT_TOKEN
```

### ❌ "Failed to connect to API"

**সমাধান:**
```bash
# Backend API চলছে কিনা চেক করুন
curl http://localhost:8000/health

# যদি কাজ না করে তাহলে API চালু করুন:
python start.py
```

### ❌ "Connection timeout"

**সমাধান:**
1. Internet connection চেক করুন
2. Firewall check করুন
3. Bot token সঠিক কিনা দেখুন

### ❌ "No module named 'telegram'"

**সমাধান:**
```bash
pip install python-telegram-bot>=20.0
```

---

## Production Deployment

### Railway এ Deploy করুন

1. **start_telegram_bot.py এ procfile add করুন:**
```
worker: python start_telegram_bot.py
```

2. **Railway এ environment variables সেট করুন:**
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_ADMIN_ID`
   - `API_URL` (Railway API URL)

3. **Deploy করুন:**
```bash
railway up
```

---

## Architecture

```
telegram_bot/
├── bot.py                   # Main bot entry
├── config.py               # Configuration
├── handlers/
│   ├── start.py           # /start command
│   ├── dashboard.py       # Dashboard view
│   ├── generate.py        # Post generation
│   ├── stats.py           # Analytics
│   ├── schedule.py        # Scheduling
│   ├── settings.py        # Settings
│   └── callbacks.py       # Button callbacks
├── services/
│   ├── formatter.py       # Message formatting
│   └── telegram_publisher.py  # Telegram API
└── __init__.py

Backend API (FastAPI) ─── Telegram Bot
     ↓
   Database (SQLite)
     ↓
   Services (AI, Topic Engine, etc.)
```

---

## উন্নত ব্যবহার

### কাস্টম Command যোগ করুন

`handlers/` ফোল্ডারে নতুন ফাইল তৈরি করুন:

```python
# handlers/my_command.py
from telegram import Update
from telegram.ext import ContextTypes

async def my_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello!")
```

তারপর `bot.py` তে যোগ করুন:
```python
from handlers.my_command import my_command

self.application.add_handler(CommandHandler("mycommand", my_command))
```

### Multi-User Support

বর্তমান সেটআপ শুধুমাত্র Admin ব্যবহার করতে পারে। Multi-user যোগ করতে:

```python
# config.py তে
ALLOWED_USERS = [123456789, 987654321, ...]
```

```python
# bot.py এ
@wraps(handler_func)
async def check_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in Config.ALLOWED_USERS:
        await update.message.reply_text("❌ You are not authorized!")
        return
    return await handler_func(update, context)
```

---

## Support

কোন সমস্যা হলে:

1. Logs চেক করুন
2. Terminal output দেখুন
3. `.env` configuration verify করুন
4. API status চেক করুন

---

**Happy Posting! 🚀**
