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

"""Offline dataset build for the recommender.

Reads the raw AniList catalogue (media.csv), keeps the anime, and produces
three artifacts under --data-dir:

  anime.db                       sqlite catalogue: a media table of
                                 human-readable metadata (including a
                                 franchise_id for reranking dedup) plus a
                                 media_fts FTS5 index for lookup.
  media_dataset.npz              row-aligned numeric model features + the
                                 text embeddings.
  preprocessing_objects.joblib   the fitted FeaturePipeline state, so serving
                                 can apply the identical transforms.

All feature logic lives in curatorchan.preprocessing.FeaturePipeline; this
script is just the offline driver. The only thing it owns is the adapter that
parses the scraper's Python-repr CSV columns back into native objects, plus the
artifact writers.

Run from inside scripts/:

    uv run python clean_data.py
    uv run python clean_data.py --data-dir ../data --device cpu --batch-size 32
"""

import argparse
import ast
import logging
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from curatorchan.preprocessing import EMBED_MODEL_NAME, FeaturePipeline

logger = logging.getLogger(__name__)

# Catalogue columns the scraper stored as Python reprs (lists / dicts).
REPR_COLUMNS = ("title", "genres", "tags", "startDate", "relations", "stats")

# MediaRelation values that keep two anime in the same franchise. The complete
# enum is ADAPTATION, PREQUEL, SEQUEL, PARENT, SIDE_STORY, CHARACTER, SUMMARY,
# ALTERNATIVE, SPIN_OFF, OTHER, SOURCE, COMPILATION, CONTAINS. Only the strong
# story links below are kept; weak ties (CHARACTER, OTHER, SPIN_OFF, the
# anime<->source ADAPTATION/SOURCE links) are dropped so a shared character or
# loose tie-in can't merge unrelated franchises. COMPILATION/CONTAINS are left
# out too — add them if recap compilations should fold into the franchise.
FRANCHISE_RELATION_TYPES = frozenset(
    {
        "PREQUEL",
        "SEQUEL",
        "SIDE_STORY",
        "PARENT",
        "ALTERNATIVE",
        "SUMMARY",
    }
)

# Order of the human-readable columns in the sqlite media table.
SERVING_COLUMNS = (
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
    "adjusted_score",
    "polarization",
    "popularity",
    "franchise_id",
)


def load_raw_media(path: str) -> pd.DataFrame:
    """
    Read the catalogue, keep anime, and parse repr columns into native objects.
    """
    df = pd.read_csv(path)
    df = df[df["type"] == "ANIME"].reset_index(drop=True)
    for col in REPR_COLUMNS:
        df[col] = df[col].apply(ast.literal_eval)
    logger.info(f"Loaded {len(df)} anime from {path}")
    return df


def compute_franchise_ids(
    df: pd.DataFrame, keep_relation_types: frozenset | None = None
) -> np.ndarray:
    """
    Label each anime with a franchise id via connected components of the
    relation graph.

    One node per catalogue anime (indexed by row position); an undirected edge
    joins two anime whenever one lists the other in its relations. Each
    connected component is a franchise, so an anime with no in-catalogue
    relation is its own singleton. The returned labels are row-aligned with df.

    The graph is built sparsely (an edge list, not a dense id-by-id matrix): the
    max anime id is ~195k, so a dense matrix would need hundreds of GB. Bridging
    through non-catalogue nodes (manga, unscraped anime) was measured to change
    almost nothing, so only catalogue-internal edges are added.

    relationType is now captured per relation edge, so passing
    keep_relation_types=FRANCHISE_RELATION_TYPES keeps only strong story links
    (sequels, side stories, ...) and drops weak ones (a shared CHARACTER, an
    OTHER relation) that would otherwise merge unrelated franchises — the kind of
    link that once collapsed Evangelion and Gundam into one 445-title component.

    Args:
        df: Catalogue with an `id` column and a parsed `relations` column in the
            scraper's shape:
            `{"edges": [{"relationType": ..., "node": {"id": ...}}, ...]}`.
        keep_relation_types: If given, only relations whose `relationType` is in
            this set become graph edges; the rest are skipped. If None, every
            relation becomes an edge.

    Returns:
        An int array of franchise labels, one per row of df.
    """
    ids = df["id"].to_numpy()
    n = len(ids)
    id_to_pos = {int(i): pos for pos, i in enumerate(ids)}

    rows: list[int] = []
    cols: list[int] = []
    for pos, rel in enumerate(df["relations"]):
        edges = rel.get("edges") if isinstance(rel, dict) else None
        for edge in edges or []:
            if (
                keep_relation_types is not None
                and edge.get("relationType") not in keep_relation_types
            ):
                continue
            node = edge.get("node") or {}
            j = id_to_pos.get(node.get("id"))
            if j is not None:
                rows.append(pos)
                cols.append(j)

    adj = coo_array((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


def build_serving_frame(
    df: pd.DataFrame, batch, franchise_ids: np.ndarray
) -> pd.DataFrame:
    """
    Assemble the human-readable media table.
    """
    return pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "title": batch.title,
            "title_english": batch.title_english,
            "description": batch.description,
            "genres": batch.genres_str,
            "tags": batch.tags_str,
            "format": batch.format,
            "isAdult": df["isAdult"].to_numpy(),
            "nsfw": batch.nsfw,
            "is_released": batch.is_released,
            "year": pd.array(batch.year_raw, dtype="Int64"),
            "meanScore": df["meanScore"].to_numpy(),
            "adjusted_score": batch.adjusted_score_raw,
            "polarization": batch.polarization_raw,
            "popularity": df["popularity"].to_numpy(),
            "franchise_id": franchise_ids,
        },
        columns=list(SERVING_COLUMNS),
    )


def write_sqlite(serving_df: pd.DataFrame, batch, db_path: str) -> None:
    """
    Write the media table, a unique id index and the media_fts index.
    """
    conn = sqlite3.connect(db_path)
    try:
        serving_df.to_sql("media", conn, if_exists="replace", index=False)
        conn.execute("DROP INDEX IF EXISTS idx_media_id")
        conn.execute("CREATE UNIQUE INDEX idx_media_id ON media(id)")

        conn.execute("DROP TABLE IF EXISTS media_fts")
        conn.execute(
            "CREATE VIRTUAL TABLE media_fts USING fts5("
            "media_id UNINDEXED, title, title_english, genres, tags, "
            "tag_descriptions, description)"
        )
        conn.executemany(
            "INSERT INTO media_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
            zip(
                serving_df["id"].tolist(),
                serving_df["title"].tolist(),
                serving_df["title_english"].fillna("").tolist(),
                serving_df["genres"].tolist(),
                serving_df["tags"].tolist(),
                batch.tag_desc_str,
                serving_df["description"].tolist(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info(f"Wrote sqlite catalogue -> {db_path}")


def write_npz(df: pd.DataFrame, batch, npz_path: str) -> None:
    """
    Write the row-aligned numeric features and text embeddings.
    """
    np.savez_compressed(
        npz_path,
        ids=df["id"].to_numpy(),
        embeddings=batch.embeddings.astype(np.float32),
        genres=batch.genres,
        formats=batch.formats,
        popularity=batch.popularity,
        year=batch.year,
        adjusted_score=batch.adjusted_score,
        polarization=batch.polarization,
        nsfw=batch.nsfw,
        is_released=batch.is_released,
    )
    logger.info(f"Wrote model dataset -> {npz_path}")


def main(args: argparse.Namespace) -> None:
    raw_path = args.raw or os.path.join(args.data_dir, "raw", "media.csv")
    db_path = os.path.join(args.data_dir, "anime.db")
    npz_path = os.path.join(args.data_dir, "media_dataset.npz")
    pipeline_path = os.path.join(args.data_dir, "preprocessing_objects.joblib")

    df = load_raw_media(raw_path)

    pipeline = FeaturePipeline(model_name=args.model, device=args.device)
    pipeline.fit(df)
    batch = pipeline.transform(df)

    logger.info(f"Embedding {len(batch.text_feat)} text features...")
    batch.embeddings = pipeline.embed(
        batch.text_feat, batch_size=args.batch_size, show_progress_bar=True
    )

    franchise_ids = compute_franchise_ids(
        df, keep_relation_types=FRANCHISE_RELATION_TYPES
    )
    counts = np.unique(franchise_ids, return_counts=True)[1]
    logger.info(
        f"Grouped {len(df)} anime into {len(counts)} franchises "
        f"(largest {counts.max()} titles)"
    )
    if counts.max() > 50:
        logger.warning(
            "Largest franchise still spans %d titles after relationType "
            "filtering; inspect it for a spurious strong link.",
            counts.max(),
        )

    serving_df = build_serving_frame(df, batch, franchise_ids)
    write_sqlite(serving_df, batch, db_path)
    write_npz(df, batch, npz_path)
    pipeline.save(pipeline_path)
    logger.info(f"Saved fitted pipeline -> {pipeline_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description="Build the recommender dataset.")
    parser.add_argument("--data-dir", default="../data", help="Artifact directory.")
    parser.add_argument(
        "--raw",
        default=None,
        help="Raw catalogue CSV (default: <data-dir>/raw/media.csv).",
    )
    parser.add_argument("--device", default="mps", help="Embedding device.")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Embedding batch size."
    )
    parser.add_argument(
        "--model", default=EMBED_MODEL_NAME, help="SentenceTransformer model."
    )
    main(parser.parse_args())
