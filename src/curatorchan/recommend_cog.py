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

# Assumed Recommendation fields:
#   title: str                  — anime title
#   url: str                    — link to the anime on AniList or MAL
#   description: str            — short synopsis
#   thumbnail_url: str | None   — cover image URL
#   community_score: float      — community rating (0–10)
#   match_score: float          — engine confidence (0–1)
#   genres: list[str]           — genre tags
#   nsfw: bool                  — whether the title is NSFW

import logging
import time
from typing import Callable, Coroutine, Literal, Optional

import discord
from discord import app_commands
from discord.ext import commands
from recommendation_engine import RecommendationEngine  # type: ignore[import-untyped]
from recommendation_engine.models import Recommendation  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Interactive view
# ---------------------------------------------------------------------------


class RecommendationView(discord.ui.View):
    """Paginated view for a list of Recommendation objects."""

    def __init__(
        self,
        recommendations: list[Recommendation],
        invoker_id: int,
        retry_callback: Callable[[], Coroutine],
        *,
        timeout: float = 180.0,
    ):
        super().__init__(timeout=timeout)
        self.recommendations = recommendations
        self.invoker_id = invoker_id
        self.retry_callback = retry_callback
        self.index = 0
        self._refresh_nav_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_nav_state(self) -> None:
        self.prev_button.disabled = self.index == 0
        self.next_button.disabled = self.index >= len(self.recommendations) - 1

    def build_embed(self) -> discord.Embed:
        rec: Recommendation = self.recommendations[self.index]
        total = len(self.recommendations)

        color = discord.Color.from_rgb(114, 137, 218)  # Discord blurple-ish
        nsfw_tag = "🔞 NSFW" if rec.nsfw else "✅ SFW"
        match_pct = f"{int(rec.match_score * 100)}%"
        genres = ", ".join(rec.genres[:4]) if rec.genres else "—"

        embed = discord.Embed(
            title=rec.title,
            url=rec.url,
            description=rec.description or "",
            color=color,
        )
        embed.set_footer(text=f"Recommendation {self.index + 1} of {total}")

        if rec.thumbnail_url:
            embed.set_thumbnail(url=rec.thumbnail_url)

        embed.add_field(name="Match", value=match_pct, inline=True)
        embed.add_field(
            name="Score", value=f"⭐ {rec.community_score:.1f}/10", inline=True
        )
        embed.add_field(name="Rating", value=nsfw_tag, inline=True)
        embed.add_field(name="Genres", value=genres, inline=False)

        return embed

    async def _check_invoker(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.invoker_id:
            await interaction.response.send_message(
                "These recommendations aren't yours to control!", ephemeral=True
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary, disabled=True)
    async def prev_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ):
        if not await self._check_invoker(interaction):
            return
        self.index -= 1
        self._refresh_nav_state()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary)
    async def next_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ):
        if not await self._check_invoker(interaction):
            return
        self.index += 1
        self._refresh_nav_state()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @discord.ui.button(label="🔄 Retry", style=discord.ButtonStyle.primary)
    async def retry_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ):
        if not await self._check_invoker(interaction):
            return
        await interaction.response.defer()
        new_recs = await self.retry_callback()
        if not new_recs:
            await interaction.followup.send(
                "Couldn't fetch new recommendations. Please try again later.",
                ephemeral=True,
            )
            return
        self.recommendations = new_recs
        self.index = 0
        self._refresh_nav_state()
        await interaction.edit_original_response(embed=self.build_embed(), view=self)

    @discord.ui.button(label="👍", style=discord.ButtonStyle.success)
    async def like_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ):
        if not await self._check_invoker(interaction):
            return
        rec = self.recommendations[self.index]
        # Placeholder: hook into feedback system here
        await interaction.response.send_message(
            f"Glad you liked **{rec.title}**! Feedback noted.", ephemeral=True
        )

    @discord.ui.button(label="👎", style=discord.ButtonStyle.danger)
    async def dislike_button(
        self, interaction: discord.Interaction, _button: discord.ui.Button
    ):
        if not await self._check_invoker(interaction):
            return
        rec = self.recommendations[self.index]
        # Placeholder: hook into feedback system here
        await interaction.response.send_message(
            f"Thanks for the feedback on **{rec.title}**.", ephemeral=True
        )

    async def on_timeout(self) -> None:
        for item in self.children:
            if isinstance(item, (discord.ui.Button, discord.ui.Select)):
                item.disabled = True


# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------


class RecommendationCog(commands.Cog):
    def __init__(self, bot, logger=None):
        self.bot = bot
        self.engine = RecommendationEngine()
        self.logger = logging.getLogger(__file__) if logger is None else logger

    @commands.Cog.listener()
    async def on_ready(self):
        self.logger.info("RecommendationCog is ready.")

    # ------------------------------------------------------------------
    # Owner utility
    # ------------------------------------------------------------------

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
                f"Synced {len(synced)} command(s) "
                f"{'to the current guild' if spec is not None else 'globally'}."
            )
        except Exception as e:
            self.logger.error(f"Sync error: {e}")
            await ctx.send(f"Failed to sync commands: {str(e)}")

    # ------------------------------------------------------------------
    # /recommend group
    # ------------------------------------------------------------------

    recommend_group = app_commands.Group(
        name="recommend",
        description="Get personalized anime recommendations.",
    )

    @recommend_group.command(
        name="username",
        description="Recommendations based on your AniList or MAL profile.",
    )
    @app_commands.describe(
        username="Your AniList or MyAnimeList username.",
        platform="The platform your profile is on (default: anilist).",
    )
    async def recommend_username(
        self,
        interaction: discord.Interaction,
        username: str,
        platform: Literal["anilist", "mal"] = "anilist",
    ):
        await interaction.response.defer(ephemeral=True)
        username = username.strip()

        async def _fetch() -> list[Recommendation]:
            return await self.engine.recommend_by_username(
                username=username, platform=platform, limit=10
            )

        await self._run_and_respond(interaction, _fetch, label=f"{platform}:{username}")

    @recommend_group.command(
        name="example",
        description="Recommendations based on anime titles you already enjoy.",
    )
    @app_commands.describe(
        titles="Comma-separated list of anime titles you like.",
    )
    async def recommend_example(
        self,
        interaction: discord.Interaction,
        titles: str,
    ):
        await interaction.response.defer(ephemeral=True)
        title_list = [t.strip() for t in titles.split(",") if t.strip()]

        if not title_list:
            await interaction.followup.send(
                "Please provide at least one anime title.", ephemeral=True
            )
            return

        async def _fetch() -> list[Recommendation]:
            return await self.engine.recommend_by_examples(titles=title_list, limit=10)

        await self._run_and_respond(
            interaction, _fetch, label=f"examples:{titles[:40]}"
        )

    @recommend_group.command(
        name="query",
        description="Recommendations based on a free-text description of what you want to watch.",
    )
    @app_commands.describe(
        query="Describe the kind of anime you're in the mood for.",
    )
    async def recommend_query(
        self,
        interaction: discord.Interaction,
        query: str,
    ):
        await interaction.response.defer(ephemeral=True)
        query = query.strip()

        async def _fetch() -> list[Recommendation]:
            return await self.engine.recommend_by_query(query=query, limit=10)

        await self._run_and_respond(interaction, _fetch, label=f"query:{query[:40]}")

    # ------------------------------------------------------------------
    # Shared response helper
    # ------------------------------------------------------------------

    async def _run_and_respond(
        self,
        interaction: discord.Interaction,
        fetch: Callable[[], Coroutine],
        *,
        label: str,
    ) -> None:
        start = time.time()
        try:
            recommendations: list[Recommendation] = await fetch()
        except Exception as e:
            self.logger.error(f"Recommendation error [{label}]: {e}")
            await interaction.followup.send(
                "Something went wrong while fetching your recommendations. "
                "Please try again later or contact the bot owner.",
                ephemeral=True,
            )
            return

        elapsed = time.time() - start
        self.logger.info(f"Recommendations for [{label}] in {elapsed:.2f}s")

        if not recommendations:
            await interaction.followup.send(
                "No recommendations found. Try a different query or username.",
                ephemeral=True,
            )
            return

        view = RecommendationView(
            recommendations=recommendations,
            invoker_id=interaction.user.id,
            retry_callback=fetch,
        )
        await interaction.followup.send(
            embed=view.build_embed(), view=view, ephemeral=True
        )


async def setup(bot):
    await bot.add_cog(RecommendationCog(bot))
