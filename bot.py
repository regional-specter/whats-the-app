import os
import uuid
import discord
from discord.ext import commands
from detection.ypt_detector import is_ypt_screenshot
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

async def contains_ypt_image(message):
    """
    Returns True if the message OR its referenced message
    contains a YPT screenshot.
    """

    async def check_attachments(attachments):
        for attachment in attachments:
            if attachment.filename.lower().endswith(IMAGE_EXTS):
                temp_file = f"temp_{uuid.uuid4().hex}_{attachment.filename}"
                await attachment.save(temp_file)
                try:
                    if is_ypt_screenshot(temp_file):
                        return True
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        return False

    # 1. Check current message attachments
    if await check_attachments(message.attachments):
        return True

    # 2. Check referenced (replied-to) message attachments
    if message.reference:
        try:
            ref_msg = await message.channel.fetch_message(
                message.reference.message_id
            )
            if await check_attachments(ref_msg.attachments):
                return True
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            pass

    return False


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} ✅")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Step 1: Must contain a YPT image (current or referenced)
    has_ypt = await contains_ypt_image(message)
    if not has_ypt:
        return

    # Step 2: Check message context
    has_text = bool(message.content.strip())
    is_reply = message.reference is not None

    # ❌ Ignore standalone image-only messages
    if not has_text and not is_reply:
        return

    # ✅ Valid cases only
    await message.reply("The app is called YPT! ✅")


bot.run(TOKEN)
