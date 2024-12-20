from typing import Literal, Optional

import discord
import pandas as pd
from anime_recommender.recommender import AnimeRecommender
from discord import app_commands
from discord.ext import commands


class RecommendationCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.recommender = AnimeRecommender(config_path="config.json")

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

        if mode == "user":
            result = recommender.run(
                search_str=search_string,
                anime_mode=False,
                limit=10,
            )

        elif mode == "anime":
            result = recommender.run(
                autoselect=True,
                search_str=search_string,
                anime_mode=True,
                limit=10,
            )

        else:
            raise ValueError("Invalid mode. Choose 'user' or 'anime'.")

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

        recommendations = self.generate_recommendations(
            recommender, mode, search_string
        )
        recommendations_embeds = [
            self.make_embed(recommendation) for recommendation in recommendations
        ]

        await interaction.followup.send(embeds=recommendations_embeds, ephemeral=True)


async def setup(bot):
    await bot.add_cog(RecommendationCog(bot))
