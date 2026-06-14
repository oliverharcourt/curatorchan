# Curator-chan
# Copyright (C) 2026  Oliver Harcourt

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

"""Deterministic anime feature preprocessing.

This module is the single source of truth for turning an anime's raw fields
into model-ready features. It is shared by two callers so that the offline
training build and online serving never drift:

  * scripts/clean_data.py fits the pipeline on the AniList catalogue and
    persists the dataset artifacts.
  * curatorchan.engine.ItemEncoder loads the fitted pipeline and encodes
    *unseen* anime at serving time via FeaturePipeline.encode_items.

Design boundary: every method here operates on **native Python structures**
(title is a dict, genres a list[str], tags a
list[dict], startDate a dict). The offline scraper dumped these as
Python reprs into media.csv, so clean_data.py parses those repr-strings
back into native objects before calling the pipeline; the live AniList GraphQL
API already returns native objects. Same code, both paths.
"""

import html
import re
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, StandardScaler

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_DEVICE = "mps"

SCORE_BUCKETS = np.arange(10, 101, 10)
SCORE_PRIOR_STRENGTH = 250.0


@dataclass
class FeatureBatch:
    """Everything derived from one pass over a batch of anime.

    Holds both human-readable serving metadata (for the sqlite catalogue) and
    the numeric model features (for the npz dataset / the learned encoders),
    so neither caller has to recompute anything. embeddings is filled in by
    FeaturePipeline.embed; it is None until then.
    """

    # Display / serving metadata
    title: list
    title_english: list
    description: list
    genres_str: list
    tags_str: list
    tag_desc_str: list
    format: list
    year_raw: list
    nsfw: np.ndarray
    is_released: np.ndarray
    adjusted_score_raw: np.ndarray
    polarization_raw: np.ndarray
    # Model features
    text_feat: list
    genres: np.ndarray
    formats: np.ndarray
    popularity: np.ndarray
    year: np.ndarray
    adjusted_score: np.ndarray
    polarization: np.ndarray
    embeddings: np.ndarray | None = None


class FeaturePipeline:
    """Fit-once / apply-anywhere feature transforms for anime items.

    The fitted state (sklearn binarizers + scalers, the imputation values and
    the embedding model name) is small and is the only thing persisted by
    save. The SentenceTransformer itself is referenced by name and
    lazy-loaded on the first embed call, so importing this module or
    running transform never pays the torch/model-loading cost.
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        prior_strength: float = SCORE_PRIOR_STRENGTH,
    ):
        self.model_name = model_name
        self.device = device
        self.prior_strength = prior_strength
        self.genre_binarizer: MultiLabelBinarizer | None = None
        self.format_binarizer: LabelBinarizer | None = None
        self.popularity_scaler: StandardScaler | None = None
        self.year_scaler: StandardScaler | None = None
        self.adjusted_score_scaler: StandardScaler | None = None
        self.polarization_scaler: StandardScaler | None = None
        self.pop_median: float | None = None
        self.year_median: float | None = None
        self.score_prior: np.ndarray | None = None
        self.score_concentration: float | None = None
        self._model = None

    @staticmethod
    def parse_title(title: dict) -> tuple:
        """Split an AniList title dict into romaji and English.

        Args:
            title: AniList title mapping; must contain romaji.

        Returns:
            A (romaji, english_or_None) tuple. Every anime has a romaji
            title; english is None when absent.
        """
        return title["romaji"], title.get("english")

    @staticmethod
    def clean_description(raw) -> str:
        """Normalize a raw synopsis into clean plain text.

        Args:
            raw: The raw description; non-string values are treated as empty.

        Returns:
            The text with newlines, HTML entities, HTML tags and the trailing
            (Source: ...) note removed.
        """
        text = raw.replace("\n", " ").strip() if isinstance(raw, str) else ""
        text = html.unescape(text)
        text = re.sub(r"<.*?>", "", text).strip()
        text = re.sub(r"\s*\(Source:\s*[^)]*\)", "", text).strip()
        return text

    @staticmethod
    def genres_to_str(genres: list) -> str:
        return ", ".join(genres) if genres else ""

    @staticmethod
    def tags_to_str(tags: list) -> str:
        return ", ".join(t["name"] for t in tags) if tags else ""

    @staticmethod
    def tag_descs_to_str(tags: list) -> str:
        return ", ".join(t["description"] for t in tags) if tags else ""

    @staticmethod
    def build_text_feature(
        title: str, description: str, genres_str: str, tags_str: str
    ) -> str:
        """Assemble the composite document that gets embedded and FTS-indexed.

        Args:
            title: The romaji title.
            description: The cleaned description.
            genres_str: Comma-separated genre names.
            tags_str: Comma-separated tag names.

        Returns:
            The single text document built from the four fields.
        """
        return (
            f"Title: {title}. Description: {description}. "
            f"Genres: {genres_str}. Tags: {tags_str}"
        )

    @staticmethod
    def bucket_format(fmt) -> str:
        """Collapse AniList formats into the coarse buckets the model uses.

        Args:
            fmt: The raw AniList format value (may be missing).

        Returns:
            One of TV, MOVIE, ONA, SPECIAL, MUSIC or
            UNKNOWN.
        """
        if fmt in ("TV", "MOVIE", "ONA"):
            return fmt
        elif fmt in ("OVA", "SPECIAL", "TV_SHORT"):
            return "SPECIAL"
        elif fmt == "MUSIC":
            return "MUSIC"
        else:
            return "UNKNOWN"

    @staticmethod
    def compute_nsfw(is_adult, genres: list, tags: list) -> bool:
        """Decide whether an item is NSFW.

        Args:
            is_adult: The AniList isAdult flag.
            genres: The item's genre names.
            tags: The item's tag dicts.

        Returns:
            True if the isAdult flag is set, the Hentai genre is
            present or any tag is flagged adult.
        """
        if any(t.get("isAdult") for t in (tags or [])):
            return True
        if "Hentai" in (genres or []):
            return True
        return bool(is_adult)

    @staticmethod
    def extract_year(start_date) -> int | None:
        """Pull the release year out of an AniList startDate dict.

        Args:
            start_date: The AniList startDate mapping.

        Returns:
            The release year, or None if absent or malformed.
        """
        if not isinstance(start_date, dict):
            return None
        return start_date.get("year")

    @staticmethod
    def parse_score_counts(stats) -> np.ndarray:
        """Read an AniList stats dict into a 10-vector of vote counts.

        Args:
            stats: The AniList stats mapping (with a scoreDistribution list of
                {score, amount} entries). Non-dict / empty yields all zeros.

        Returns:
            Vote counts aligned to SCORE_BUCKETS (10..100); missing buckets 0.
        """
        dist = stats.get("scoreDistribution") if isinstance(stats, dict) else None
        amount_by_score = {
            int(b["score"]): b["amount"] for b in dist or [] if b.get("amount")
        }
        return np.array(
            [amount_by_score.get(int(s), 0) for s in SCORE_BUCKETS], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Stateful fit / transform / embed
    # ------------------------------------------------------------------
    def _smoothed_scores(self, counts: np.ndarray) -> tuple:
        """Dirichlet-multinomial posterior mean and spread per anime.

        Args:
            counts: An (n, 10) array of per-anime vote counts over SCORE_BUCKETS.

        Returns:
            A (adjusted_score, polarization) pair of length-n float arrays: the
            posterior expected score (0-100) and its standard deviation (score
            points). Zero-vote rows resolve to the prior mean / prior std.
        """
        alpha = self.score_concentration * self.score_prior
        votes = counts.sum(axis=1, keepdims=True)
        posterior = (counts + alpha) / (votes + self.score_concentration)
        adjusted = posterior @ SCORE_BUCKETS
        variance = posterior @ (SCORE_BUCKETS**2) - adjusted**2
        polarization = np.sqrt(np.clip(variance, 0, None))
        return adjusted, polarization

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit the binarizers, scalers and imputation medians on a catalogue.

        Args:
            df: The catalogue, with native genres, format, popularity,
                startDate and stats columns.

        Returns:
            This pipeline, fitted (for chaining).
        """
        genres = df["genres"].apply(lambda g: g if len(g) > 0 else ["UNKNOWN"])
        self.genre_binarizer = MultiLabelBinarizer().fit([genres.explode().unique()])

        formats = df["format"].apply(self.bucket_format)
        self.format_binarizer = LabelBinarizer().fit(formats)

        pop = pd.to_numeric(df["popularity"], errors="coerce")
        self.pop_median = pop.median()
        pop = np.log1p(pop.fillna(self.pop_median).to_numpy(dtype=np.float64)).reshape(
            -1, 1
        )
        self.popularity_scaler = StandardScaler().fit(pop)

        years = pd.Series(
            [self.extract_year(sd) for sd in df["startDate"]], dtype="float64"
        )
        self.year_median = years.median()
        years = years.fillna(self.year_median).to_numpy(dtype=np.float64).reshape(-1, 1)
        self.year_scaler = StandardScaler().fit(years)

        # Equal-weighted prior
        counts = np.vstack([self.parse_score_counts(s) for s in df["stats"]])
        votes = counts.sum(axis=1)
        voted = votes > 0
        self.score_prior = (counts[voted] / votes[voted, None]).mean(axis=0)
        self.score_concentration = self.prior_strength
        adjusted, polarization = self._smoothed_scores(counts)
        self.adjusted_score_scaler = StandardScaler().fit(adjusted.reshape(-1, 1))
        self.polarization_scaler = StandardScaler().fit(polarization.reshape(-1, 1))
        return self

    def transform(self, df: pd.DataFrame) -> FeatureBatch:
        """Apply the fitted transforms, producing a fully-derived FeatureBatch.

        Args:
            df: A frame of native anime fields, the same shape fit expects.

        Returns:
            A FeatureBatch with display metadata and numeric features filled
            in; embeddings is left as None.

        Raises:
            RuntimeError: If the pipeline has not been fit or loaded.
        """
        if self.genre_binarizer is None:
            raise RuntimeError(
                "FeaturePipeline must be fit or loaded before transform()."
            )

        genres_list = [list(g) if g is not None else [] for g in df["genres"]]
        tags_list = [t if isinstance(t, list) else [] for t in df["tags"]]

        title, title_english = [], []
        for t in df["title"]:
            romaji, english = self.parse_title(t)
            title.append(romaji)
            title_english.append(english)

        description = [self.clean_description(d) for d in df["description"]]
        genres_str = [self.genres_to_str(g) for g in genres_list]
        tags_str = [self.tags_to_str(t) for t in tags_list]
        tag_desc_str = [self.tag_descs_to_str(t) for t in tags_list]
        text_feat = [
            self.build_text_feature(ti, de, ge, ta)
            for ti, de, ge, ta in zip(title, description, genres_str, tags_str)
        ]

        formats = [self.bucket_format(f) for f in df["format"]]
        nsfw = np.array(
            [
                self.compute_nsfw(a, g, t)
                for a, g, t in zip(df["isAdult"], genres_list, tags_list)
            ],
            dtype=bool,
        )

        # Genres: empty -> ["UNKNOWN"] before binarizing, matching fit.
        genres_for_mlb = [g if len(g) > 0 else ["UNKNOWN"] for g in genres_list]
        genres_feat = self.genre_binarizer.transform(genres_for_mlb).astype(np.uint8)
        format_feat = self.format_binarizer.transform(formats).astype(np.uint8)

        pop = pd.to_numeric(df["popularity"], errors="coerce").fillna(self.pop_median)
        pop = np.log1p(pop.to_numpy(dtype=np.float64)).reshape(-1, 1)
        popularity_feat = (
            self.popularity_scaler.transform(pop).astype(np.float32).ravel()
        )

        year_raw = [self.extract_year(sd) for sd in df["startDate"]]
        years = pd.Series(year_raw, dtype="float64").fillna(self.year_median)
        years = years.to_numpy(dtype=np.float64).reshape(-1, 1)
        current_year = pd.Timestamp.now().year
        is_released = (years <= current_year).ravel()
        year_feat = self.year_scaler.transform(years).astype(np.float32).ravel()

        counts = np.vstack([self.parse_score_counts(s) for s in df["stats"]])
        adjusted_raw, polarization_raw = self._smoothed_scores(counts)
        adjusted_feat = (
            self.adjusted_score_scaler.transform(adjusted_raw.reshape(-1, 1))
            .astype(np.float32)
            .ravel()
        )
        polarization_feat = (
            self.polarization_scaler.transform(polarization_raw.reshape(-1, 1))
            .astype(np.float32)
            .ravel()
        )

        return FeatureBatch(
            title=title,
            title_english=title_english,
            description=description,
            genres_str=genres_str,
            tags_str=tags_str,
            tag_desc_str=tag_desc_str,
            format=formats,
            year_raw=year_raw,
            nsfw=nsfw,
            is_released=is_released,
            adjusted_score_raw=adjusted_raw.astype(np.float32),
            polarization_raw=polarization_raw.astype(np.float32),
            text_feat=text_feat,
            genres=genres_feat,
            formats=format_feat,
            popularity=popularity_feat,
            year=year_feat,
            adjusted_score=adjusted_feat,
            polarization=polarization_feat,
        )

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(
        self, texts, batch_size: int = 32, show_progress_bar: bool = False
    ) -> np.ndarray:
        """Embed text features into vectors.

        Args:
            texts: The text documents to embed.
            batch_size: Encoder batch size.
            show_progress_bar: Whether to display a progress bar.

        Returns:
            A float array of embeddings, L2-normalized so a dot product equals
            cosine similarity.
        """
        model = self._get_model()
        return model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )

    def encode_items(self, raw_items, batch_size: int = 32) -> FeatureBatch:
        """Encode one or more unseen native AniList items (serving entry point).

        Args:
            raw_items: A single item dict or a list of them, each with native
                fields.
            batch_size: Encoder batch size for the embedding step.

        Returns:
            A FeatureBatch with embeddings populated.
        """
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        df = pd.DataFrame(raw_items)
        for col in ("description", "format", "popularity", "isAdult", "stats"):
            if col not in df.columns:
                df[col] = None
        batch = self.transform(df)
        batch.embeddings = self.embed(batch.text_feat, batch_size=batch_size)
        return batch

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "genre_binarizer": self.genre_binarizer,
                "format_binarizer": self.format_binarizer,
                "popularity_scaler": self.popularity_scaler,
                "year_scaler": self.year_scaler,
                "adjusted_score_scaler": self.adjusted_score_scaler,
                "polarization_scaler": self.polarization_scaler,
                "pop_median": self.pop_median,
                "year_median": self.year_median,
                "score_prior": self.score_prior,
                "score_concentration": self.score_concentration,
                "model_name": self.model_name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = DEFAULT_DEVICE) -> "FeaturePipeline":
        state = joblib.load(path)
        obj = cls(model_name=state["model_name"], device=device)
        obj.genre_binarizer = state["genre_binarizer"]
        obj.format_binarizer = state["format_binarizer"]
        obj.popularity_scaler = state["popularity_scaler"]
        obj.year_scaler = state["year_scaler"]
        obj.adjusted_score_scaler = state["adjusted_score_scaler"]
        obj.polarization_scaler = state["polarization_scaler"]
        obj.pop_median = state["pop_median"]
        obj.year_median = state["year_median"]
        obj.score_prior = state["score_prior"]
        obj.score_concentration = state["score_concentration"]
        return obj
