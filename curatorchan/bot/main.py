import os
from typing import Literal

import discord
import pandas as pd
from anime_recommender.main import AnimeRecommender
from discord.ext import commands
from dotenv import load_dotenv

DESCRIPTION = """
Curator-chan is a Discord bot that recommends anime to users.
"""

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", description=DESCRIPTION, intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")
    print(f"ID: {bot.user.id}")
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")


def generate_recommendations(mode: str, search_string: str) -> pd.DataFrame:
    result = None

    if mode == "user":
        result = AnimeRecommender(
            config_path="config.json",
            search_str=search_string,
            anime_mode=False,
            limit=10,
        ).run()

    elif mode == "anime":
        result = AnimeRecommender(
            config_path="config.json",
            search_str=search_string,
            anime_mode=True,
            limit=10,
        ).run(autoselect=True)

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


def make_embed(recommendation: dict) -> discord.Embed:
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
    embed.add_field(name="Match", value=f"{recommendation['distance']}%", inline=False)
    embed.add_field(
        name="Mean Score", value=f"{round(recommendation['mean'], 2)}", inline=True
    )
    embed.add_field(name="NSFW", value=f"{nsfw}", inline=True)
    return embed


@bot.tree.command(name="recommend", description="Get personalized recommendations.")
@discord.app_commands.describe(
    mode="Select recommendation mode (user or anime).",
    search_string="Enter the username or anime name.",
)
async def recommend(
    interaction: discord.Interaction, mode: Literal["user", "anime"], search_string: str
):
    """Get recommendations based on mode and search string."""
    await interaction.response.defer(ephemeral=True)

    recommendations = generate_recommendations(mode, search_string)
    recommendations_embeds = [
        make_embed(recommendation) for recommendation in recommendations
    ]

    await interaction.followup.send(embeds=recommendations_embeds, ephemeral=True)


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


load_secrets()
print("Starting bot...")
bot.run(os.getenv("DISCORD_TOKEN"))
