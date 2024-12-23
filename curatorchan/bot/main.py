import asyncio
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv


def load_secrets():
    env = os.getenv("ENV")
    if env == "dev":
        return
    if env == "production":
        path = "/run/secrets/keys"
        if not os.path.exists(path):
            raise FileNotFoundError("Secrets file not found.")
        load_dotenv(dotenv_path=path)
    else:
        raise ValueError("ENV must be set to 'dev' or 'production'.")


async def load_cogs():
    await bot.load_extension("recommend_cog")


async def main():
    await load_cogs()
    await bot.start(os.getenv("DISCORD_TOKEN"))


if __name__ == "__main__":
    load_secrets()
    DESCRIPTION = """
    Curator-chan is a Discord bot that recommends anime to users.
    """

    intents = discord.Intents.default()
    intents.message_content = True

    bot = commands.Bot(command_prefix="uwu", description=DESCRIPTION, intents=intents)

    asyncio.run(main())
