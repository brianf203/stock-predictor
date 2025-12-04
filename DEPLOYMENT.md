# Deployment Guide - Keep Bot Running 24/7

## Option 1: Railway (Recommended - Free & Easy)

### Steps:
1. Go to [railway.app](https://railway.app) and sign up (free)
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your GitHub account and select this repository
4. Add environment variable:
   - Go to your project → Variables
   - Add: `DISCORD_BOT_TOKEN` = your bot token from `.env`
5. Railway will automatically deploy and keep it running!

### Railway automatically:
- Detects Python projects
- Installs dependencies from `requirements.txt`
- Runs `bot.py` on startup
- Keeps it running 24/7
- Restarts on crashes

---

## Option 2: Render (Free Tier Available)

### Steps:
1. Go to [render.com](https://render.com) and sign up
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python3 bot.py`
   - **Environment:** Python 3
5. Add environment variable:
   - `DISCORD_BOT_TOKEN` = your bot token
6. Click "Create Web Service"

**Note:** Free tier spins down after 15 minutes of inactivity. Upgrade to paid ($7/month) for always-on.

---

## Option 3: Replit (Free, Simple)

### Steps:
1. Go to [replit.com](https://replit.com) and sign up
2. Click "Create Repl" → "Import from GitHub"
3. Select this repository
4. Add environment variable (Secrets):
   - `DISCORD_BOT_TOKEN` = your bot token
5. Click "Run" button
6. For 24/7: Upgrade to "Hacker" plan ($7/month) or use UptimeRobot to ping it

---

## Option 4: Keep Running on Your Mac (Requires Computer On)

### Using launchd (macOS Service):

1. Create a plist file:

```bash
nano ~/Library/LaunchAgents/com.stockpredictor.bot.plist
```

2. Paste this (update paths):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.stockpredictor.bot</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/Users/brian/Stock Predictor/bot.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/brian/Stock Predictor</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/brian/Stock Predictor/bot.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/brian/Stock Predictor/bot_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
```

3. Load the service:

```bash
launchctl load ~/Library/LaunchAgents/com.stockpredictor.bot.plist
```

4. Start it:

```bash
launchctl start com.stockpredictor.bot
```

5. Check status:

```bash
launchctl list | grep stockpredictor
```

6. To stop:

```bash
launchctl unload ~/Library/LaunchAgents/com.stockpredictor.bot.plist
```

---

## Option 5: VPS (DigitalOcean, Linode, etc.)

### DigitalOcean Droplet ($6/month):

1. Create a Droplet (Ubuntu 22.04)
2. SSH into it
3. Install Python and dependencies
4. Clone your repo
5. Set up environment variables
6. Use `systemd` or `screen`/`tmux` to keep it running

---

## Quick Comparison

| Service | Cost | Always On | Difficulty |
|---------|------|-----------|------------|
| Railway | Free | ✅ Yes | ⭐ Easy |
| Render | Free/Paid | ⚠️ Free spins down | ⭐ Easy |
| Replit | Free/Paid | ⚠️ Free spins down | ⭐ Easy |
| VPS | $5-10/mo | ✅ Yes | ⭐⭐ Medium |
| Local Mac | Free | ⚠️ Needs computer on | ⭐⭐ Medium |

---

## Recommended: Railway

Railway is the easiest and truly free option:
- ✅ Free tier available
- ✅ Always on
- ✅ Auto-deploys from GitHub
- ✅ Easy environment variables
- ✅ Automatic restarts

Just push your code to GitHub and connect Railway!



