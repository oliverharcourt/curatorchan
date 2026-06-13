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

import ast
import html
import re
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer, StandardScaler

media_df = pd.read_csv("../data/raw/media.csv")

preprocessing_objects = {}

# - id --> used to identify anime
# - title --> add romaji title to start of desctiption string
# - description --> generate embeddings from description text
# - genres --> add to end of description string, and make multi-hot vector feature
# - tags --> add to end of description string
# - isadult --> binary, combine with isAdult tags and 'Hentai' genre to get nsfw feature
# - format --> one-hot encode (TV, MOVIE, SPECIAL, UNKNOWN)
# - meanScore --> might be needed for the Dirichlet-multinormal smoothing of the scores, else drop
# - popularity --> log-transform and StandardScaler, could be used as debiasing weights for recommendations
# - relations --> use scipy connected components to find groups of connected anime for dedup during reranking
# - startDate --> StandardScaler to create a feature of how recent the anime is
# - endDate --> drop
# - season --> drop
# - updatedAt --> drop
# - stats --> use Dirichlet-multinomial smoothing with the catalogue-wide score distribution as a prior to derive, per anime, a shrunk score distribution, which gives us (1) a shrunk mean score (mean of the shrunk distribution) and (2) a score to judge how polarizing an anime is (var of shrunk distribution (higher --> more polarizing))
# - type --> drop


# Remove non-anime entries
media_df = media_df[media_df["type"] == "ANIME"].reset_index(drop=True)


# Clean title and description and create a text feature that is then embedded
# Text feature structure: "Title: {title}. Description: {description}. Genres: {genres}. Tags: {tag_names}."

# All anime have a romaji title, but not all have an english or native title
titles = media_df["title"].apply(ast.literal_eval)
media_df["title"] = titles.apply(lambda t: t["romaji"])
media_df["title_english"] = titles.apply(lambda t: t.get("english"))

# Clean description text and extract genres and tags
media_df["description"] = media_df["description"].apply(
    lambda x: x.replace("\n", " ").strip() if isinstance(x, str) else ""
)
media_df["description"] = media_df["description"].apply(html.unescape)
media_df["description"] = media_df["description"].apply(
    lambda x: re.sub(r"<.*?>", "", x).strip() if isinstance(x, str) else ""
)
media_df["description"] = media_df["description"].apply(
    lambda x: (
        re.sub(r"\s*\(Source:\s*[^)]*\)", "", x).strip() if isinstance(x, str) else ""
    )
)
genres_list = media_df["genres"].apply(ast.literal_eval)
genres_join_str = genres_list.apply(", ".join)
tag_names_str = media_df["tags"].apply(
    lambda x: (
        ", ".join([d["name"] for d in ast.literal_eval(x)])
        if isinstance(x, str)
        else ""
    )
)

# Extract tag descriptions for FTS
media_df["tag_desc_feat"] = media_df["tags"].apply(
    lambda x: (
        ", ".join([d["description"] for d in ast.literal_eval(x)])
        if isinstance(x, str)
        else ""
    )
)

# Add combined text feature for embeddings and FTS
media_df["text_feat"] = (
    "Title: "
    + media_df["title"]
    + ". Description: "
    + media_df["description"]
    + ". Genres: "
    + genres_join_str
    + ". Tags: "
    + tag_names_str
)


# Create a binary feature to catch adult content based on tags, genres and isAdult column
def detect_adult_tags(tags_list):
    for tag in tags_list:
        if "isAdult" in tag and tag["isAdult"]:
            return True
    return False


adult_tags = (
    media_df["tags"]
    .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    .apply(detect_adult_tags)
)

is_hentai = genres_list.apply(lambda x: "Hentai" in x)

media_df["nsfw"] = media_df["isAdult"] | adult_tags | is_hentai


# Multi-hot encode genres
genres = genres_list.apply(lambda x: x if len(x) > 0 else ["UNKNOWN"])
genre_binarizer = MultiLabelBinarizer().fit([genres.explode().unique()])
preprocessing_objects["genre_binarizer"] = genre_binarizer
genres_feat = genre_binarizer.transform(genres).astype(np.uint8)

# Replace the raw list repr with a display string for the DB
media_df["genres"] = genres_join_str


# Scale popularity (raw count stays in the DB; the scaled value is a model feature)
pop = media_df["popularity"]
pop = pop.fillna(pop.median())
pop = pop.to_numpy(dtype=np.float64)
pop = np.log1p(pop).reshape(-1, 1)
popularity_scaler = StandardScaler().fit(pop)
preprocessing_objects["popularity_scaler"] = popularity_scaler
popularity_feat = popularity_scaler.transform(pop).astype(np.float32)


# Scale meanScore
# mean_score = media_df["meanScore"].to_numpy(dtype=np.float32)
# mean_score = (
#     StandardScaler().fit_transform(mean_score.reshape(-1, 1))
# )
# media_df["meanScore"] = mean_score
# media_df = media_df[media_df["meanScore"].notna()]


# Create binary features from format
def clean_format(x):
    if x in ["TV", "MOVIE", "ONA"]:
        return x
    elif x in ["OVA", "SPECIAL", "TV_SHORT"]:
        return "SPECIAL"
    elif x == "MUSIC":
        return "MUSIC"
    else:
        return "UNKNOWN"


media_df["format"] = media_df["format"].apply(clean_format)
format_binarizer = LabelBinarizer().fit(media_df["format"])
preprocessing_objects["format_binarizer"] = format_binarizer
format_feat = format_binarizer.transform(media_df["format"]).astype(np.uint8)


# Extract year from startDate (raw year stays in the DB, NULL when unknown)
start_year = media_df["startDate"].apply(lambda x: ast.literal_eval(x)["year"])
media_df["year"] = start_year.astype("Int64")
median_year = start_year.median()
start_year = start_year.fillna(median_year)
start_year = start_year.to_numpy(dtype=np.float64).reshape(-1, 1)
current_year = pd.Timestamp.now().year
media_df["is_released"] = (start_year <= current_year).ravel()
year_scaler = StandardScaler().fit(start_year)
preprocessing_objects["year_scaler"] = year_scaler
year_feat = year_scaler.transform(start_year).astype(np.float32)


# Drop unused columns
media_df = media_df.drop(
    columns=[
        "type",
        "tags",
        "startDate",
        "endDate",
        "updatedAt",
        "season",
        "relations",
        "stats",
    ]
)

# Tag names replace the raw tag dicts as a display/FTS string
media_df["tags"] = tag_names_str


# Persist the fitted transforms so serving code can apply/invert them
joblib.dump(preprocessing_objects, "../data/preprocessing_objects.joblib")


# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="mps")

# Generate embeddings (normalized, so dot product == cosine similarity)
embeddings = model.encode(
    media_df["text_feat"].tolist(),
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True,
)


# Serving metadata goes to sqlite; numeric feature arrays go to the npz below
serving_cols = [
    "id",
    "title",
    "title_english",
    "description",
    "genres",
    "tags",
    "format",
    "isAdult",
    "nsfw",
    "is_released",
    "year",
    "meanScore",
    "popularity",
]

conn = sqlite3.connect("../data/anime.db")
media_df[serving_cols].to_sql("media", conn, if_exists="replace", index=False)
conn.execute("CREATE UNIQUE INDEX idx_media_id ON media(id)")

# Full-text search table; media_id joins back to media.id
conn.execute("DROP TABLE IF EXISTS media_fts")
conn.execute(
    "CREATE VIRTUAL TABLE media_fts USING fts5("
    "media_id UNINDEXED, title, title_english, genres, tags, tag_descriptions, "
    "description)"
)
conn.executemany(
    "INSERT INTO media_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
    zip(
        media_df["id"].tolist(),
        media_df["title"].tolist(),
        media_df["title_english"].fillna("").tolist(),
        media_df["genres"].tolist(),
        media_df["tags"].tolist(),
        media_df["tag_desc_feat"].tolist(),
        media_df["description"].tolist(),
    ),
)
conn.commit()
conn.close()


# Model-ready arrays, row-aligned with ids; feature column names live in the
# binarizers/scalers persisted above
np.savez_compressed(
    "../data/media_dataset.npz",
    ids=media_df["id"].to_numpy(),
    embeddings=embeddings.astype(np.float32),
    genres=genres_feat,
    formats=format_feat,
    popularity=popularity_feat.ravel(),
    year=year_feat.ravel(),
    nsfw=media_df["nsfw"].to_numpy(),
    is_released=media_df["is_released"].to_numpy(),
)
