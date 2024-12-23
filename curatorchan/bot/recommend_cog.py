import logging
import time
from typing import Literal, Optional

import discord
import pandas as pd
from anime_recommender import exceptions
from anime_recommender.recommender import AnimeRecommender
from discord import app_commands
from discord.ext import commands


class RecommendationCog(commands.Cog):
    def __init__(self, bot, logger: logging.Logger):
        self.bot = bot
        self.recommender = AnimeRecommender(config_path="config.json")
        self.logger = logger

    @commands.Cog.listener()
    async def on_ready(self):
        print("RecommendationCog is ready.")

    @commands.command(name="sync")
    @commands.is_owner()
    async def sync(self, ctx, spec: Optional[Literal["~"]] = None):
        try:
            await ctx.send("Syncing commands...")
            if spec == "~":
                # ctx.bot.tree.copy_global_to(guild=ctx.guild)
                synced = await ctx.bot.tree.sync(guild=ctx.guild)
                print(f"Guild sync - commands: {[cmd.name for cmd in synced]}")

            else:
                synced = await ctx.bot.tree.sync()
                print(f"Global sync - commands: {[cmd.name for cmd in synced]}")

            await ctx.send(
                f"Synced {len(synced)} command(s) {'to the current guild' if spec is not None else 'globally'}."
            )
        except Exception as e:
            print(f"Sync error: {e}")
            await ctx.send(f"Failed to sync commands: {str(e)}")

    @app_commands.command(name="test")
    async def test(self, interaction: discord.Interaction):
        await interaction.response.send_message("Test command works!")

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

        dataset = pd.read_json("data/raw/mal_anime_data.json", orient="records")

        merged_result = result.merge(dataset, on="id")

        cols = ["title", "distance", "link", "mean", "nsfw"]

        merged_result = merged_result[cols]

        merged_result["distance"] = merged_result["distance"].apply(
            lambda x: int(round(x, 2) * 10)
        )
        return merged_result.to_dict(orient="records")

    def make_embed(self, recommendation: dict) -> discord.Embed:
        """Create an embed from a recommendation."""
        # thumbnail = get the thumbnail from mal and add it here
        # make this some color in the thumbnail?
        color = discord.Color.from_rgb(255, 215, 0)
        nsfw = "🔞 YES" if recommendation["nsfw"] == "black" else "✅ NO"
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
        await interaction.response.defer(ephemeral=True)

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
            time.sleep(300)
            return
        except exceptions.UserNotFoundError:
            await interaction.followup.send(
                f"User {search_string} not found. Please check the username and try again.",
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

        await interaction.followup.send(embeds=recommendations_embeds, ephemeral=True)


async def setup(bot):
    logger = logging.getLogger("curatorchan")
    logger.setLevel(logging.DEBUG)
    logging.getLogger("discord.http").setLevel(logging.INFO)

    handler = logging.handlers.RotatingFileHandler(
        filename="discord.log",
        encoding="utf-8",
        maxBytes=32 * 1024 * 1024,  # 32 MiB
        backupCount=5,  # Rotate through 5 files
    )
    dt_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        "[{asctime}] [{levelname:<8}] {name}: {message}", dt_fmt, style="{"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    await bot.add_cog(RecommendationCog(bot, logger))
