#!/usr/bin/env python3
"""
🤖 Telegram Bot Health Check
সব কিছু ঠিক আছে কিনা চেক করুন
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

print("\n" + "=" * 70)
print("🔍 TELEGRAM BOT HEALTH CHECK")
print("=" * 70 + "\n")

# Load environment
load_dotenv()

checks = {
    "✅ Config Files": [],
    "✅ Environment Variables": [],
    "✅ Dependencies": [],
    "✅ Bot Setup": [],
}

# 1. Config Files Check
print("1️⃣  Checking Config Files...")
files_to_check = [
    ("telegram_bot/bot.py", "Main bot file"),
    ("telegram_bot/config.py", "Configuration"),
    ("telegram_bot/handlers/start.py", "Start handler"),
    ("telegram_bot/handlers/dashboard.py", "Dashboard handler"),
    ("telegram_bot/handlers/generate.py", "Generate handler"),
    ("telegram_bot/services/formatter.py", "Message formatter"),
    ("start_telegram_bot.py", "Bot launcher"),
    (".env", "Environment file"),
]

for file_path, desc in files_to_check:
    if Path(file_path).exists():
        checks["✅ Config Files"].append(f"   ✅ {file_path:<45} - {desc}")
    else:
        checks["✅ Config Files"].append(f"   ❌ {file_path:<45} - {desc} (MISSING)")

# 2. Environment Variables Check
print("\n2️⃣  Checking Environment Variables...")
env_vars = {
    "TELEGRAM_BOT_TOKEN": "Bot Token",
    "TELEGRAM_ADMIN_ID": "Admin ID",
    "OPENAI_API_KEY": "OpenAI Key",
    "LINKEDIN_ACCESS_TOKEN": "LinkedIn Token",
}

bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
admin_id = os.getenv("TELEGRAM_ADMIN_ID", "")

for var, desc in env_vars.items():
    value = os.getenv(var, "")
    if value:
        if "TOKEN" in var or "KEY" in var:
            masked = value[:10] + "..." + value[-5:] if len(value) > 15 else "***"
        else:
            masked = value
        
        status = "✅"
        checks["✅ Environment Variables"].append(f"   {status} {var:<30} - {masked}")
    else:
        checks["✅ Environment Variables"].append(f"   ⚠️  {var:<30} - NOT SET")

# 3. Dependencies Check
print("\n3️⃣  Checking Dependencies...")
dependencies = [
    ("telegram", "python-telegram-bot"),
    ("aiogram", "aiogram"),
    ("fastapi", "fastapi"),
    ("pydantic", "pydantic"),
    ("openai", "openai"),
    ("dotenv", "python-dotenv"),
]

for module, package in dependencies:
    try:
        __import__(module)
        checks["✅ Dependencies"].append(f"   ✅ {package:<30} - Installed")
    except ImportError:
        checks["✅ Dependencies"].append(f"   ❌ {package:<30} - NOT INSTALLED")

# 4. Bot Setup Check
print("\n4️⃣  Checking Bot Setup...")

if bot_token:
    checks["✅ Bot Setup"].append(f"   ✅ Bot Token is configured")
else:
    checks["✅ Bot Setup"].append(f"   ❌ Bot Token is NOT configured")

if admin_id:
    checks["✅ Bot Setup"].append(f"   ✅ Admin ID is configured: {admin_id}")
else:
    checks["✅ Bot Setup"].append(f"   ❌ Admin ID is NOT configured")

if bot_token and admin_id:
    checks["✅ Bot Setup"].append(f"   ✅ Bot is ready to start!")
else:
    checks["✅ Bot Setup"].append(f"   ⚠️  Bot needs configuration")

# Print Results
print("\n" + "=" * 70)
print("📊 HEALTH CHECK RESULTS")
print("=" * 70 + "\n")

for category, results in checks.items():
    print(f"\n{category}")
    print("-" * 70)
    for result in results:
        print(result)

# Summary
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

total_checks = sum(len(v) for v in checks.values())
passed_checks = sum(1 for v in checks.values() for item in v if "✅" in item)

print(f"\nTotal Checks: {total_checks}")
print(f"Passed: {passed_checks}")
print(f"Failed: {total_checks - passed_checks}")

if passed_checks == total_checks:
    print("\n✅ ALL CHECKS PASSED! Your bot is ready to start!")
    print("\nRun these commands:")
    print("  Terminal 1: python start.py")
    print("  Terminal 2: python start_telegram_bot.py")
elif passed_checks >= total_checks * 0.8:
    print("\n⚠️  SOME CHECKS FAILED!")
    print("Please check the errors above and fix them.")
else:
    print("\n❌ CRITICAL ERRORS FOUND!")
    print("Please check the errors above before starting the bot.")

print("\n" + "=" * 70)
print("🎊 Bot Information")
print("=" * 70)
print(f"\nBot Username:  @borno_posting_bot")
print(f"Bot URL:       t.me/borno_posting_bot")
print(f"Your ID:       {admin_id if admin_id else 'NOT SET'}")
print(f"Token Status:  {'✅ Configured' if bot_token else '❌ Not configured'}")

print("\n" + "=" * 70)
print("📚 Quick Links")
print("=" * 70)
print("\nDocumentation:")
print("  1. Quick Start:      TELEGRAM_BOT_QUICKSTART.md")
print("  2. Setup Guide:      TELEGRAM_BOT_SETUP.md")
print("  3. Implementation:   TELEGRAM_BOT_IMPLEMENTATION.md")
print("  4. Summary:          TELEGRAM_BOT_COMPLETE_SUMMARY.md")
print("  5. Bot Ready:        BOT_READY.txt")

print("\n" + "=" * 70 + "\n")

# Exit code
sys.exit(0 if passed_checks == total_checks else 1)
