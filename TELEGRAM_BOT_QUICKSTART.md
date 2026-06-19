# 🤖 Telegram Bot - Quick Start

## 30-সেকেন্ডের সেটআপ

### 1️⃣ Telegram Bot Token পান

- Telegram এ **@BotFather** সার্চ করুন
- `/newbot` দিন এবং নাম দিন
- Token পাবেন: `123:ABC...`

### 2️⃣ Your User ID পান

- Telegram এ **@userinfobot** সার্চ করুন
- `/start` দিন
- ID পাবেন: `123456789`

### 3️⃣ .env ফাইল আপডেট করুন

```bash
TELEGRAM_BOT_TOKEN=123:ABC...
TELEGRAM_ADMIN_ID=123456789
```

### 4️⃣ Bot চালান

```bash
# Terminal 1: Backend API
python start.py

# Terminal 2: Telegram Bot
python start_telegram_bot.py
```

### 5️⃣ Telegram এ Test করুন

- আপনার bot খুঁজুন: `@your_bot_name`
- `/start` দিন
- মেনু পাবেন! ✅

---

## 📱 Available Commands

```
/start       - মূল মেনু
/dashboard   - ড্যাশবোর্ড দেখান
/generate    - পোস্ট জেনারেট করুন
/stats       - Statistics
/schedule    - পরবর্তী পোস্ট
/settings    - সেটিংস
/help        - সাহায্য
/about       - বট সম্পর্কে
```

---

## 🔥 Main Features

| Feature | Command | Details |
|---------|---------|---------|
| **Dashboard** | `/dashboard` | Real-time stats & status |
| **Generate** | `/generate` | AI-powered post creation |
| **Analytics** | `/stats` | Topic performance analysis |
| **Scheduler** | `/schedule` | View & manage posting schedule |
| **Settings** | `/settings` | Configure bot settings |

---

## ❌ Common Issues

| Issue | Solution |
|-------|----------|
| Bot Token Error | চেক করুন `.env` এ সঠিক token আছে কিনা |
| API Connection Failed | Backend API চলছে কিনা দেখুন: `python start.py` |
| Module Not Found | Install করুন: `pip install -r requirements.txt` |

---

## 📚 Full Setup Guide

বিস্তারিত গাইড দেখুন: [TELEGRAM_BOT_SETUP.md](TELEGRAM_BOT_SETUP.md)

---

**Let's go! 🚀**
