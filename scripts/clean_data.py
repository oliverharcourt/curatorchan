import ast
import os
import re
import time

import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.preprocessing import StandardScaler


def clean_data():
    load_dotenv()

    media_df = pd.read_csv("../data/media.csv")
    # users_df = pd.read_csv("../data/users.csv")

    # - id --> dropped for training; used in vector db to identify anime
    # - title --> add romaji title to start of desctiption string
    # - description --> generate embeddings from description text
    # - genres --> add to end of description string
    # - tags --> add to end of description string
    # - isadult --> keep as binary feature
    # - format --> create is_tv, is_movie and is_other binary features
    # - meanScore --> StandardScaler
    # - popularity --> log-transform and StandardScaler
    # - relations --> drop (create is_parent and is_child binary features)
    # - startDate --> StandardScaler to create a feature of how recent the anime is
    # - endDate --> drop or (endDate - startDate) could be used as a feature of anime length
    # - season --> drop
    # - updatedAt --> drop
    # - stats --> drop or create a 'controversy score' feature based on std of score distribution
    # - type --> drop

    # Remove non-anime entries
    media_df = media_df[media_df["type"] == "ANIME"]

    # Clean title and description and create text feature
    # First: remove entries with missing description, since its critical for recommendations
    media_df = media_df[media_df["description"].notna()]

    # All anime have a romaji title, but not all have an english or native title
    media_df["title"] = media_df["title"].apply(lambda x: ast.literal_eval(x)["romaji"])

    # Clean description text and extract genres and tags
    media_df["description"] = media_df["description"].apply(
        lambda x: x.replace("\n", " ").strip() if isinstance(x, str) else x
    )
    media_df["description"] = media_df["description"].apply(
        lambda x: re.sub(r"<.*?>", "", x).strip() if isinstance(x, str) else x
    )
    media_df["genres"] = media_df["genres"].apply(
        lambda x: ", ".join(ast.literal_eval(x)) if isinstance(x, str) else ""
    )
    media_df["tag_names"] = media_df["tags"].apply(
        lambda x: (
            ", ".join([d["name"] for d in ast.literal_eval(x)])
            if isinstance(x, str)
            else ""
        )
    )
    media_df["tag_desc"] = media_df["tags"].apply(
        lambda x: (
            " ".join([d["description"] for d in ast.literal_eval(x)])
            if isinstance(x, str)
            else ""
        )
    )

    # Add combined text feature for embeddings and FTS
    media_df["text_feature"] = (
        media_df["title"]
        + ". "
        + media_df["description"]
        + ". Genres: "
        + media_df["genres"]
        + ". Tags: "
        + media_df["tag_names"]
        + ". Tag Descriptions: "
        + media_df["tag_desc"]
    )

    # Scale popularity
    pop = media_df["popularity"].to_numpy(dtype=np.float16)
    pop = np.log1p(pop)
    pop = StandardScaler().fit_transform(pop.reshape(-1, 1))
    media_df["popularity"] = pop
    media_df = media_df[media_df["popularity"].notna()]

    # Scale meanScore
    mean_score = media_df["meanScore"].to_numpy(dtype=np.float16)
    mean_score = StandardScaler().fit_transform(mean_score.reshape(-1, 1))
    media_df["meanScore"] = mean_score
    media_df = media_df[media_df["meanScore"].notna()]

    # Create binary features from format
    media_df["is_tv"] = media_df["format"].apply(lambda x: 1 if x == "TV" else 0)
    media_df["is_movie"] = media_df["format"].apply(lambda x: 1 if x == "MOVIE" else 0)
    media_df["is_special"] = media_df["format"].apply(
        lambda x: 1 if x in ["OVA", "ONA", "SPECIAL"] else 0
    )
    media_df["is_other"] = media_df["format"].apply(
        lambda x: 1 if x not in ["TV", "MOVIE", "OVA", "ONA", "SPECIAL"] else 0
    )

    # isAdult feature
    media_df["isAdult"] = media_df["isAdult"].apply(lambda x: 1 if x else 0)

    # Extract year from startDate
    start_date = (
        media_df["startDate"].apply(lambda x: ast.literal_eval(x)["year"]).to_numpy()
    )
    start_date = StandardScaler().fit_transform(start_date.reshape(-1, 1))
    media_df["start_year"] = start_date
    media_df = media_df[media_df["start_year"].notna()]

    # Drop unused columns
    media_df = media_df.drop(
        columns=[
            "type",
            "format",
            "title",
            "description",
            "genres",
            "tags",
            "tag_names",
            "tag_desc",
            "startDate",
            "endDate",
            "updatedAt",
            "season",
            "relations",
            "stats",
        ]
    )

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    text_feature = list(media_df["text_feature"].values)
    total_features = len(text_feature)
    responses = []

    current_index = 0
    batch_size = 20
    max_batch_size = 20
    consecutive_429s = 0
    max_429s = 10

    print(f"Starting embedding with {total_features} items...")

    while current_index < total_features:
        end_index = min(current_index + batch_size, total_features)
        batch = text_feature[current_index:end_index]

        print(
            f"Embedding batch: items {current_index} to {end_index} (size: {len(batch)})"
        )

        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            responses.append(response)

            # Success!
            current_index += len(batch)
            consecutive_429s = 0

            # Slowly increase batch size if we were throttled, up to max
            if batch_size < max_batch_size:
                batch_size += 1

            time.sleep(30)  # Polite delay

        except Exception as e:
            error_str = str(e)
            status_code = getattr(e, "code", None)

            # Client side errors - fail hard. Use status_code if available.
            # Note: 429 is handled below, so we exclude it here.
            is_client_error = False
            if status_code is not None:
                if status_code in [400, 401, 403]:
                    is_client_error = True
            else:
                # Fallback to string check if no code available (risky due to false positives like 403 in timestamps)
                if "400" in error_str or "401" in error_str or "403" in error_str:
                    is_client_error = True

            if is_client_error or "InvalidArgument" in error_str:
                print(f"Client error encountered: {e}")
                raise e

            # Throttling errors
            if (
                (status_code == 429)
                or (status_code is None and "429" in error_str)
                or "ResourceExhausted" in error_str
            ):
                consecutive_429s += 1
                print(f"Hit rate limit (429). consecutive_errors={consecutive_429s}")

                # If we hit too many 429s, reduce batch size
                if consecutive_429s >= max_429s:
                    old_size = batch_size
                    batch_size = max(1, int(batch_size * 0.8))  # Reduce by 20%, min 1
                    print(
                        f"Too many 429s. Reducing batch size from {old_size} to {batch_size}"
                    )
                    consecutive_429s = (
                        0  # Reset counter after adjustment to give it a chance
                    )

                # Backoff wait
                sleep_time = min(2 * (2**consecutive_429s), 60)
                print(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)

            else:
                # Other errors
                print(f"Unexpected error: {e}")
                time.sleep(5)

    embedding_arrays = []
    for batch in responses:
        for embedding in batch.embeddings:
            array = embedding.values
            embedding_arrays.append(array)

    embeddings = np.vstack(embedding_arrays, dtype=np.float16)

    ids = media_df["id"].to_numpy()
    metadata = media_df.drop(columns=["id", "text_feature"]).to_numpy()

    np.savez_compressed(
        "../data/media_dataset.npz",
        ids=ids,
        embeddings=embeddings,
        metadata=metadata,
    )


if __name__ == "__main__":
    clean_data()
