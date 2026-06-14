"""AniList dataset scraper.

Two-phase user scrape:
  1. Discover — enumerate accounts with Page.users (50 per request) and read
     cheap profile aggregates from User.statistics.
  2. Hydrate  — fetch each user's complete anime list with MediaListCollection
     (500 entries per chunk). The statistics mediaIds buckets are capped at
     25 ids per bucket server-side, so full lists must be fetched per user.

The phases are interleaved per discovery page so progress can be checkpointed:
after every fully-processed page the harvested rows are appended to the output
CSVs and the resume state is written. Re-running the script continues from the
last checkpoint. Users whose lists come back empty (private or actually empty)
are dropped.

Outputs (in --data-dir, default ../data):
  users.csv             one row per kept user (profile aggregates)
  user_anime_lists.csv  one row per (user, anime) list entry; score uses the
                        POINT_100 scale, 0 = unscored
  media.csv             anime/manga catalogue (target "media")
  fetch_state.json      resume state for the user scrape

Run from inside scripts/:
  uv run python fetch_data.py users               # resumes automatically
  uv run python fetch_data.py users --max-pages 100
  uv run python fetch_data.py media
"""

import argparse
import json
import logging
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

API_URL = "https://graphql.anilist.co"

PER_PAGE = 50  # server-side maximum for Page queries
PER_CHUNK = 500  # server-side maximum for MediaListCollection chunks
MAX_CHUNKS = 40  # safety bound per user (40 * 500 = 20k entries)
DEFAULT_RPM = 30  # the API is currently degraded to 30 req/min
MAX_RPM = 90  # normal limit; the advertised limit is picked up from headers
# Use only this fraction of the per-minute budget. Pacing right at the cap
# drifts against the server's fixed window and forces 60s recovery pauses,
# which costs more throughput than the headroom does. 0.85 is the highest
# setting whose worst-case window stays clear of the remaining<=3 soft brake;
# anything above trades steady throughput for stalls.
SAFETY_FACTOR = 0.85
MEDIA_CHECKPOINT_PAGES = 50

REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "curator-chan dataset builder",
}

USER_COLUMNS = [
    "user_id",
    "user_name",
    "anime_count",
    "mean_score",
    "standard_deviation",
    "minutes_watched",
    "episodes_watched",
    "list_entries",
]
ENTRY_COLUMNS = ["user_id", "media_id", "status", "score", "progress", "repeat"]

USER_PAGE_QUERY = """
query UserPage($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    users(sort: ID) {
      id
      name
      statistics {
        anime {
          count
          meanScore
          standardDeviation
          minutesWatched
          episodesWatched
        }
      }
    }
  }
}
"""

USER_LIST_QUERY = """
query UserList($userId: Int, $chunk: Int, $perChunk: Int) {
  MediaListCollection(userId: $userId, type: ANIME, chunk: $chunk, perChunk: $perChunk) {
    hasNextChunk
    lists {
      entries {
        mediaId
        status
        score(format: POINT_100)
        progress
        repeat
      }
    }
  }
}
"""

MEDIA_QUERY = """
query MediaPage($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    media(sort: ID) {
      id
      type
      title {
        native
        romaji
        english
      }
      description
      genres
      tags {
        id
        name
        category
        isAdult
        description
        rank
      }
      isAdult
      format
      meanScore
      popularity
      relations {
        nodes {
          id
          title {
            english
          }
        }
      }
      startDate {
        year
      }
      endDate {
        year
      }
      season
      updatedAt
      stats {
        scoreDistribution {
          amount
          score
        }
      }
    }
  }
}
"""


class RateLimiter:
    """Paces requests to stay under the API's requests-per-minute limit."""

    def __init__(self, rpm=DEFAULT_RPM):
        self._interval = 60.0 / (rpm * SAFETY_FACTOR)
        self._next_ok = 0.0

    def wait(self):
        delay = self._next_ok - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_ok = time.monotonic() + self._interval

    def update_from_headers(self, headers):
        limit = headers.get("X-RateLimit-Limit", "")
        if limit.isdigit():
            # Track the advertised limit (90 normally, 30 while the API is
            # degraded), spending only SAFETY_FACTOR of it.
            rpm = max(min(int(limit), MAX_RPM), 1)
            self._interval = 60.0 / (rpm * SAFETY_FACTOR)
        remaining = headers.get("X-RateLimit-Remaining", "")
        if not remaining.isdigit():
            return
        if int(remaining) == 0:
            reset = headers.get("X-RateLimit-Reset", "")
            wait_s = max(int(reset) - time.time(), 5.0) if reset.isdigit() else 60.0
            logger.warning(f"Rate budget exhausted; pausing {wait_s:.0f}s")
            self._next_ok = max(self._next_ok, time.monotonic() + wait_s)
        elif int(remaining) <= 3:
            # Nearly dry (window drift, or another process sharing the
            # budget) — ease off instead of hitting the 60s cliff.
            logger.info(f"Rate budget low ({remaining} left); easing off 5s")
            self._next_ok = max(self._next_ok, time.monotonic() + 5.0)


def make_request(query, variables, limiter, max_failures=5, max_backoff=60):
    """POST a GraphQL request with rate limiting and retries.

    Returns the parsed JSON on success, or None on permanent request-specific
    errors (e.g. 404 for a deleted account). Raises RuntimeError once
    transient failures exceed max_failures.
    """
    failures = 0
    backoff = 3
    while True:
        limiter.wait()
        try:
            response = requests.post(
                API_URL,
                json={"query": query, "variables": variables},
                headers=REQUEST_HEADERS,
                timeout=30,
            )
        except requests.RequestException as e:
            logger.error(f"Error during post request: {e}")
            failures += 1
            if failures >= max_failures:
                raise RuntimeError("Max failures reached while making request")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
            continue

        limiter.update_from_headers(response.headers)

        if response.status_code == 200:
            return response.json()

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "")
            wait_s = int(retry_after) + 1 if retry_after.isdigit() else 60
            logger.warning(f"Rate limited (429); sleeping {wait_s}s")
            time.sleep(wait_s)
            continue  # throttling is expected; not counted as a failure

        if response.status_code in (400, 404):
            logger.warning(
                f"Permanent error {response.status_code}: {response.text[:200]}"
            )
            return None

        logger.error(
            f"Request failed with status code {response.status_code}: "
            f"{response.text[:200]}"
        )
        failures += 1
        if failures >= max_failures:
            raise RuntimeError("Max failures reached while making request")
        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


def append_csv(rows, path, columns):
    if not rows:
        return
    pd.DataFrame(rows, columns=columns).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )


def load_state(state_path):
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"next_user_page": 1}


def save_state(state, state_path):
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, state_path)


def load_harvested_ids(users_csv):
    """User ids already in users.csv, used to skip replayed work on resume."""
    if not os.path.exists(users_csv):
        return set()
    try:
        return set(pd.read_csv(users_csv, usecols=["user_id"])["user_id"])
    except ValueError as e:
        raise SystemExit(
            f"{users_csv} is incompatible with this scraper (old format?): {e} "
            "Move the file away or rerun with --fresh."
        )


def fetch_user_list(user_id, limiter):
    """Fetch a user's complete anime list, deduplicated by media id.

    Returns [] for private, empty, or deleted accounts.
    """
    entries = {}
    for _chunk in range(1, MAX_CHUNKS + 1):
        response = make_request(
            USER_LIST_QUERY,
            {"userId": user_id, "chunk": _chunk, "perChunk": PER_CHUNK},
            limiter,
        )
        if response is None:
            return []
        collection = (response.get("data") or {}).get("MediaListCollection")
        if collection is None:
            return []
        # Custom lists repeat entries from the status lists; dedupe by media id.
        for group in collection.get("lists") or []:
            for entry in group.get("entries") or []:
                entries.setdefault(entry["mediaId"], entry)
        if not collection.get("hasNextChunk"):
            break
    else:
        logger.warning(f"User {user_id} exceeded {MAX_CHUNKS} chunks; list truncated")
    return [
        (user_id, e["mediaId"], e["status"], e["score"], e["progress"], e["repeat"])
        for e in entries.values()
    ]


def scrape_users(limiter, data_dir, max_pages=None, fresh=False):
    users_csv = os.path.join(data_dir, "users.csv")
    lists_csv = os.path.join(data_dir, "user_anime_lists.csv")
    state_path = os.path.join(data_dir, "fetch_state.json")
    os.makedirs(data_dir, exist_ok=True)

    if fresh:
        for path in (users_csv, lists_csv, state_path):
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed {path}")

    state = load_state(state_path)
    page = state.get("next_user_page", 1)
    harvested = load_harvested_ids(users_csv)
    if page > 1 or harvested:
        logger.info(
            f"Resuming at page {page} ({len(harvested)} users already harvested)"
        )

    user_buffer, entry_buffer = [], []
    pages_done = 0
    has_next_page = True
    pbar = tqdm(desc="Scraping users", unit="page")
    try:
        while has_next_page and (max_pages is None or pages_done < max_pages):
            response = make_request(
                USER_PAGE_QUERY, {"page": page, "perPage": PER_PAGE}, limiter
            )
            if response is None or not response.get("data"):
                raise RuntimeError(f"Discovery failed permanently on page {page}")
            page_data = response["data"]["Page"]
            has_next_page = page_data["pageInfo"]["hasNextPage"]

            for user in page_data["users"]:
                stats = (user.get("statistics") or {}).get("anime") or {}
                # Skip users already harvested (page replay after a crash)
                # and users with nothing on their list.
                if user["id"] in harvested or not stats.get("count"):
                    continue
                rows = fetch_user_list(user["id"], limiter)
                if not rows:
                    continue  # private or empty list — drop the user
                entry_buffer.extend(rows)
                user_buffer.append(
                    (
                        user["id"],
                        user["name"],
                        stats.get("count"),
                        stats.get("meanScore"),
                        stats.get("standardDeviation"),
                        stats.get("minutesWatched"),
                        stats.get("episodesWatched"),
                        len(rows),
                    )
                )
                harvested.add(user["id"])

            page += 1
            pages_done += 1

            # Checkpoint: flush data first, then advance the resume state.
            # If we crash in between, the replayed page is skipped via the
            # harvested-ids guard above.
            append_csv(user_buffer, users_csv, USER_COLUMNS)
            append_csv(entry_buffer, lists_csv, ENTRY_COLUMNS)
            user_buffer.clear()
            entry_buffer.clear()
            state["next_user_page"] = page
            save_state(state, state_path)

            pbar.update(1)
            pbar.set_postfix(users=len(harvested))
    except KeyboardInterrupt:
        # Flush the partial page; resume skips these users via users.csv.
        append_csv(user_buffer, users_csv, USER_COLUMNS)
        append_csv(entry_buffer, lists_csv, ENTRY_COLUMNS)
        logger.info("Interrupted — progress checkpointed; rerun to resume.")
    finally:
        pbar.close()

    if not has_next_page:
        logger.info("Reached the last user page; scrape complete.")


def scrape_media(limiter, data_dir, start_page=1, max_pages=None):
    media_csv = os.path.join(data_dir, "media.csv")
    os.makedirs(data_dir, exist_ok=True)

    data = []
    page = start_page
    pages_done = 0
    has_next_page = True
    pbar = tqdm(desc="Fetching media", unit="page")
    while has_next_page and (max_pages is None or pages_done < max_pages):
        response = make_request(
            MEDIA_QUERY, {"page": page, "perPage": PER_PAGE}, limiter
        )
        if response is None or not response.get("data"):
            raise RuntimeError(f"Media fetch failed permanently on page {page}")
        page_data = response["data"]["Page"]
        data.extend(page_data["media"])
        has_next_page = page_data["pageInfo"]["hasNextPage"]
        page += 1
        pages_done += 1
        pbar.update(1)
        if pages_done % MEDIA_CHECKPOINT_PAGES == 0:
            pd.DataFrame(data).to_csv(media_csv, index=False)
            logger.info(f"Checkpointed {len(data)} media records to {media_csv}")
    pbar.close()

    pd.DataFrame(data).to_csv(media_csv, index=False)
    logger.info(f"Saved {len(data)} media records to {media_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AniList dataset scraper")
    parser.add_argument("target", choices=["users", "media"], help="What to scrape.")
    parser.add_argument("--data-dir", default="../data", help="Output directory.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Stop after this many pages (default: run to the end).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="users only: discard previous state and outputs, start from page 1.",
    )
    parser.add_argument(
        "--start-page", type=int, default=1, help="media only: first page to fetch."
    )
    args = parser.parse_args()

    limiter = RateLimiter()
    if args.target == "users":
        scrape_users(limiter, args.data_dir, max_pages=args.max_pages, fresh=args.fresh)
    else:
        scrape_media(
            limiter, args.data_dir, start_page=args.start_page, max_pages=args.max_pages
        )
