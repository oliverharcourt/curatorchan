# Curator-chan
# Copyright (C) 2024  Oliver Harcourt

# This file is part of Curator-chan.

# Curator-chan is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Curator-chan is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with Curator-chan.  If not, see <https://www.gnu.org/licenses/>.

import asyncio
import logging
import os

import discord
import settings
from discord.ext import commands
from dotenv import load_dotenv

from curatorchan.bot.recommend_cog import RecommendationCog


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
    await bot.add_cog(
        RecommendationCog(
            bot,
        )  # logger=logger.getChild("RecommendationCog"))
    )


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

    logging.basicConfig(level=logging.INFO)

    logger_name = "bot-dev" if os.getenv("ENV") == "dev" else "curatorchan"

    logger = settings.logging.getLogger(logger_name)

    logger.info("Starting Curator-chan...")

    asyncio.run(main())
