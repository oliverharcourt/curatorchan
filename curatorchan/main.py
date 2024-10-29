from discord.ext import commands
import discord
import os
import sys

from anime_recommender.main import AnimeRecommender


DESCRIPTION = """
Curator-chan is a Discord bot that recommends anime to users.
"""

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    description=DESCRIPTION,
    intents=intents
)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")
    print(f"ID: {bot.user.id}")
