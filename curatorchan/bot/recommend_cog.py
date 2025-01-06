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

import logging
import time
from typing import Literal, Optional

import discord
import pandas as pd
import settings
from anime_recommender import exceptions
from anime_recommender.recommender import AnimeRecommender
from discord import app_commands
from discord.ext import commands


class RecommendationCog(commands.Cog):
    def __init__(self, bot, logger=None):
        self.bot = bot
        self.recommender = AnimeRecommender(config_path="config.json")
        self.logger = (
            settings.logging.getLogger("bot-dev") if logger is None else logger
        )

    @commands.Cog.listener()
    async def on_ready(self):
        self.logger.info("RecommendationCog is ready.")

    @commands.command(name="sync")
    @commands.is_owner()
    async def sync(self, ctx, spec: Optional[Literal["~"]] = None):
        try:
            await ctx.send("Syncing commands...")
            if spec == "~":
                synced = await ctx.bot.tree.sync(guild=ctx.guild)
                self.logger.info(
                    f"Guild sync - commands: {[cmd.name for cmd in synced]}"
                )

            else:
                synced = await ctx.bot.tree.sync()
                self.logger.info(
                    f"Global sync {ctx.guild} - commands: {[cmd.name for cmd in synced]}"
                )

            await ctx.send(
                f"Synced {len(synced)} command(s) {'to the current guild' if spec is not None else 'globally'}."
            )
        except Exception as e:
            self.logger.error(f"Sync error: {e}")
            await ctx.send(f"Failed to sync commands: {str(e)}")

    def generate_recommendations(
        self, recommender: AnimeRecommender, mode: str, search_string: str
    ) -> pd.DataFrame:
        result = None

        if mode != "user" and mode != "anime":
            raise ValueError("Invalid mode. Choose 'user' or 'anime'.")

        result = recommender.run(
            autoselect=True if mode == "anime" else False,
            search_str=search_string,
            anime_mode=True if mode == "anime" else False,
            limit=10,
        )

        if isinstance(result, pd.Series):
            self.logger.debug(f"Result is a series:\n {result}")
            result = result.to_frame()

        self.logger.debug(f"Result columns: {result.columns}")

        dataset = pd.read_json("data/raw/mal_anime_data.json", orient="records")

        merged_result = result.merge(dataset, on="id", how="left")

        cols = ["title", "distance", "link", "mean", "nsfw"]

        if set(cols).issubset(merged_result.columns):
            self.logger.debug(f"Valid result: {merged_result.columns}")
            merged_result = merged_result[cols]
        else:
            self.logger.debug(f"Invalid result: {merged_result.columns}")
            raise NotImplementedError(
                "Recommendation generation produced invalid result."
            )

        merged_result["distance"] = merged_result["distance"].apply(
            lambda x: int(round(x, 2) * 10)
        )
        return merged_result.to_dict(orient="records")

    def make_embed(self, recommendation: dict) -> discord.Embed:
        """Create an embed from a recommendation."""
        # thumbnail = get the thumbnail from mal and add it here
        # make this some color in the thumbnail?
        color = discord.Color.from_rgb(255, 215, 0)
        nsfw = "🔞 NSFW" if recommendation["nsfw"] == "black" else "✅ SFW"
        embed = discord.Embed(
            title=recommendation["title"],
            url=recommendation["link"],
            description="",
            color=color,
        )
        embed.add_field(
            name="Match", value=f"{recommendation['distance']}%", inline=False
        )
        embed.add_field(
            name="Mean Score", value=f"{round(recommendation['mean'], 2)}", inline=True
        )
        embed.add_field(name="NSFW", value=f"{nsfw}", inline=True)
        return embed

    @app_commands.command(
        name="recommend", description="Get personalized recommendations."
    )
    @discord.app_commands.describe(
        mode="Select recommendation mode (user or anime).",
        search_string="Enter the username or anime name.",
    )
    async def recommend(
        self,
        interaction: discord.Interaction,
        mode: Literal["user", "anime"],
        search_string: str,
    ):
        """Get recommendations based on mode and search string."""

        search_string = str(search_string).strip()
        mode = str(mode).strip()

        await interaction.response.defer(ephemeral=True)

        start_time = time.time()

        recommender = AnimeRecommender(config_path="config.json")

        try:
            recommendations = self.generate_recommendations(
                recommender, mode, search_string
            )
        except exceptions.InvalidTokenError:
            self.logger.error("InvalidTokenError")
            await interaction.followup.send(
                "There was an error when fetching your recommendations. Please contact the bot owner for assistance.",
                ephemeral=True,
            )
            return
        except exceptions.RateLimitExceededError:
            self.logger.warning("RateLimitExceededError")
            await interaction.followup.send(
                "The bot is currently rate limited. Please try again later.",
                ephemeral=True,
            )
            return
        except exceptions.UserNotFoundError:
            await interaction.followup.send(
                f"User '{search_string}' not found. Please check the username and try again.",
                ephemeral=True,
            )
            return
        except NotImplementedError as e:
            self.logger.error(f"NotImplementedError: {str(e)}")
            await interaction.followup.send(
                "There was an error when fetching your recommendations. If this issue persists, please contact the bot owner for assistance.",
                ephemeral=True,
            )

        recommendations_embeds = [
            self.make_embed(recommendation) for recommendation in recommendations
        ]

        end_time = time.time()

        self.logger.info(
            f"Responded to '{search_string}' in {mode} mode in {end_time - start_time:.2f} seconds."
        )

        await interaction.followup.send(embeds=recommendations_embeds, ephemeral=True)


async def setup(bot):
    await bot.add_cog(RecommendationCog(bot))
