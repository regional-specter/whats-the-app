import os
import discord
from discord.ext import commands
from detection.ypt_detector import is_ypt_screenshot
from dotenv import load_dotenv
from rapidfuzz import fuzz
import re

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Intents for reading messages
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

def is_question_about_app_keywords(message_content):
    msg = message_content.lower()
    keywords_sets = [
        {"what", "app", "called"},
        {"whats", "name", "app"},
        {"which", "app"},
        {"app", "name"}
    ]
    for kw_set in keywords_sets:
        if all(kw in msg for kw in kw_set):
            return True
    return False

# Precompile regex patterns for detecting questions about the app
question_patterns = [
    re.compile(r"what['’]?s the app called", re.IGNORECASE),
    re.compile(r"whats the name of this app", re.IGNORECASE),
    re.compile(r"which app is this", re.IGNORECASE)
]

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} ✅")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check text for question patterns
    if is_question_about_app_keywords(message.content):
        await message.reply("The app is called YPT! ✅")
        return  # Optional: stop further processing

    # Check for image attachments
    for attachment in message.attachments:
        if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            file_path = f"temp_{attachment.filename}"
            await attachment.save(file_path)
            
            if is_ypt_screenshot(file_path):
                await message.reply("This looks like a YPT screenshot! ✅")
            
            os.remove(file_path)

# Run the bot
bot.run(TOKEN)
