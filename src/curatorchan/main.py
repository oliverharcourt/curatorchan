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
from discord.ext import commands
from dotenv import load_dotenv

from curatorchan import (
    settings as _settings,  # noqa: F401 — imported for logging configuration side effect
)
from curatorchan.recommend_cog import RecommendationCog


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


async def load_cogs(bot: commands.Bot, logger: logging.Logger):
    await bot.add_cog(
        RecommendationCog(bot, logger=logger.getChild("RecommendationCog"))
    )


async def main(bot: commands.Bot, token: str, logger: logging.Logger):
    # Catch errors in non-command event handlers (on_message, on_ready, etc.)
    @bot.event
    async def on_error(event: str, *args, **kwargs):
        logger.exception(f"Unhandled error in event '{event}'")

    # Catch unhandled slash command errors and ensure the interaction gets a reply
    @bot.tree.error
    async def on_app_command_error(
        interaction: discord.Interaction, error: discord.app_commands.AppCommandError
    ):
        cmd = interaction.command.name if interaction.command else "unknown"
        logger.exception(
            f"Unhandled error in app command '{cmd}' "
            f"[user={interaction.user.id} guild={interaction.guild_id}]",
            exc_info=error,
        )
        msg = "An unexpected error occurred. Please try again later."
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)

    await load_cogs(bot, logger)
    try:
        await bot.start(token)
    finally:
        logger.info("Curator-chan has shut down.")


if __name__ == "__main__":
    load_secrets()

    DESCRIPTION = """
    Curator-chan is a Discord bot that recommends anime to users.
    """

    intents = discord.Intents.default()

    bot = commands.Bot(command_prefix="uwu", description=DESCRIPTION, intents=intents)

    logger = logging.getLogger(__file__)

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable is not set.")

    logger.info("Starting Curator-chan...")

    try:
        asyncio.run(main(bot, token, logger))
    except discord.LoginFailure:
        logger.critical("Invalid Discord token — check DISCORD_TOKEN and try again.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception:
        logger.exception("Fatal error during bot startup or runtime.")
