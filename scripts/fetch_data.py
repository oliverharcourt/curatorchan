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
  media.csv             anime catalogue (target "media"), one row per anime
  fetch_state.json      resume state for the user scrape

The media scrape fetches the anime catalogue by id, in batches of 50 via the
`id_in` filter, reading the ids to fetch from a prior media.csv (--ids-csv).
AniList now caps offset pagination at 5000 entries ("Page depth exceeds maximum
allowed"), so the catalogue can no longer be walked with a plain `sort:ID` page
counter; id-batching sidesteps the cap. Anime added to AniList since the id
source was built are discovered first (ids above the highest known id, found by
paging ID_DESC) and folded into the work list; pass --no-discover to skip this.
The output resumes automatically (ids already present in media.csv are skipped)
and --fresh starts over.

Run from inside scripts/:
  uv run python fetch_data.py users               # resumes automatically
  uv run python fetch_data.py users --max-pages 100
  uv run python fetch_data.py media               # ids from ../data/raw/media.csv, + new
  uv run python fetch_data.py media --no-discover  # only ids already in --ids-csv
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
MAX_LIST_PAGES = 400  # safety bound for the paginated fallback (400 * 50 = 20k)
DEFAULT_RPM = 30  # the API is currently degraded to 30 req/min
MAX_RPM = 90  # normal limit; the advertised limit is picked up from headers
# Use only this fraction of the per-minute budget. Pacing right at the cap
# drifts against the server's fixed window and forces 60s recovery pauses,
# which costs more throughput than the headroom does. 0.85 is the highest
# setting whose worst-case window stays clear of the remaining<=3 soft brake;
# anything above trades steady throughput for stalls.
SAFETY_FACTOR = 0.85
MEDIA_BATCH_SIZE = PER_PAGE  # ids per id_in request (Page perPage caps at 50)
MEDIA_CHECKPOINT_BATCHES = 20  # flush every 20 batches (~1000 anime)
MAX_DISCOVERY_PAGES = 100  # 100 * 50 = 5000, the API's offset-pagination cap

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
# Column order for media.csv. Nested fields (title/tags/relations/studios/stats/
# dates) are written as their Python-repr, matching how the catalogue has always
# been stored; clean_data.py parses them back to native objects.
MEDIA_COLUMNS = [
    "id",
    "idMal",
    "type",
    "title",
    "description",
    "genres",
    "synonyms",
    "tags",
    "isAdult",
    "format",
    "source",
    "season",
    "seasonYear",
    "episodes",
    "duration",
    "countryOfOrigin",
    "averageScore",
    "meanScore",
    "popularity",
    "favourites",
    "relations",
    "studios",
    "startDate",
    "endDate",
    "updatedAt",
    "stats",
]

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

# Fallback for users whose list reproducibly 500s on MediaListCollection: the
# flat, paginated mediaList endpoint resolves the same entries one page at a
# time. Same fields, so it yields identical rows.
USER_LIST_PAGE_QUERY = """
query UserListPage($userId: Int, $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    mediaList(userId: $userId, type: ANIME) {
      mediaId
      status
      score(format: POINT_100)
      progress
      repeat
    }
  }
}
"""

# Fetched by id (id_in, batches of 50) rather than by paging sort:ID, because
# AniList caps offset pagination at 5000 entries. relations now carries the
# relationType on each edge (the previous nodes-only form lost it); studios,
# statusDistribution, and the extra metadata fields are new this scrape.
MEDIA_QUERY = """
query MediaByIds($ids: [Int], $perPage: Int) {
  Page(perPage: $perPage) {
    media(id_in: $ids, type: ANIME, sort: ID) {
      id
      idMal
      type
      title {
        native
        romaji
        english
      }
      description
      genres
      synonyms
      tags {
        id
        name
        category
        description
        rank
        isGeneralSpoiler
        isMediaSpoiler
        isAdult
        userId
      }
      isAdult
      format
      source
      season
      seasonYear
      episodes
      duration
      countryOfOrigin
      averageScore
      meanScore
      popularity
      favourites
      relations {
        edges {
          relationType
          node {
            id
            type
          }
        }
      }
      studios {
        edges {
          isMain
          node {
            id
            name
            isAnimationStudio
          }
        }
      }
      startDate {
        year
      }
      endDate {
        year
      }
      updatedAt
      stats {
        scoreDistribution {
          score
          amount
        }
        statusDistribution {
          status
          amount
        }
      }
    }
  }
}
"""

# Discovery query: just the ids of the newest anime. AniList assigns media ids
# in increasing order, so anime added since the last scrape have ids above the
# highest id we already know. Paging type:ANIME by ID_DESC walks the newest
# entries first, so the scan can stop the moment it reaches known ids.
MEDIA_DISCOVERY_QUERY = """
query NewAnime($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    media(type: ANIME, sort: ID_DESC) {
      id
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
            logger.error(f"Error during post request for {variables}: {e}")
            failures += 1
            if failures >= max_failures:
                raise RuntimeError(f"Max failures reached for request {variables}")
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
            f"Request failed with status code {response.status_code} for "
            f"{variables}: {response.text[:200]}"
        )
        failures += 1
        if failures >= max_failures:
            raise RuntimeError(f"Max failures reached for request {variables}")
        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


# Trivial query against a stable id — used as a liveness probe to tell a
# transient API outage apart from a cluster of genuinely-broken users.
HEALTH_QUERY = "query { Media(id: 1) { id } }"
HEALTH_WAIT_BUDGET = 1800  # give a sustained outage this long to recover, then stop


def api_is_healthy(limiter):
    """True if the API answers the trivial health query right now."""
    try:
        response = make_request(HEALTH_QUERY, {}, limiter, max_failures=1)
    except RuntimeError:
        return False
    return bool(response and (response.get("data") or {}).get("Media"))


def wait_until_healthy(limiter, max_wait=HEALTH_WAIT_BUDGET):
    """Block until a health probe succeeds, or give up after max_wait seconds.

    Returns True the moment the API answers (immediately if it never went down),
    or False once a sustained outage outlasts the budget. Sleeps with capped
    exponential backoff between probes so we neither hammer nor drift.
    """
    waited = 0
    delay = 30
    while True:
        if api_is_healthy(limiter):
            return True
        if waited >= max_wait:
            return False
        logger.warning(
            f"API health probe failed; outage suspected, waiting {delay}s "
            f"(waited {waited}s of {max_wait}s)"
        )
        time.sleep(delay)
        waited += delay
        delay = min(delay * 2, 300)


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


def load_anime_ids(ids_csv):
    """Anime media ids to (re)scrape, taken from a prior media.csv.

    The `type` column selects ANIME rows. This is the work list for the media
    scrape: AniList caps offset pagination at 5000 entries, so the catalogue is
    fetched by `id_in` over ids we already know rather than by paging `sort:ID`.
    Returns a deduplicated, sorted list so resume batches line up run to run.
    """
    if not os.path.exists(ids_csv):
        raise SystemExit(
            f"Id source {ids_csv} not found. Point --ids-csv at a prior media.csv "
            "(the completed catalogue scrape) to supply the anime ids to fetch."
        )
    try:
        df = pd.read_csv(ids_csv, usecols=["id", "type"])
    except ValueError as e:
        raise SystemExit(f"{ids_csv} lacks the id/type columns this scraper needs: {e}")
    ids = df.loc[df["type"] == "ANIME", "id"].dropna().astype(int)
    return sorted(set(ids))


def discover_new_media_ids(limiter, watermark):
    """Anime ids newer than `watermark` (the highest id we already know).

    Pages type:ANIME by ID_DESC (newest first) and stops as soon as a page
    reaches ids at or below the watermark, since everything beyond is already
    known. Stays under the 5000-entry pagination cap as long as fewer than that
    many anime were added since the watermark. Returns the new ids (descending).
    """
    new_ids = []
    for page in range(1, MAX_DISCOVERY_PAGES + 1):
        response = make_request(
            MEDIA_DISCOVERY_QUERY, {"page": page, "perPage": PER_PAGE}, limiter
        )
        if response is None or not (response.get("data") or {}).get("Page"):
            if not wait_until_healthy(limiter):
                raise RuntimeError(f"Anime discovery failed; API down on page {page}")
            continue  # transient blip — retry the same page
        page_data = response["data"]["Page"]
        ids = [m["id"] for m in page_data["media"]]
        fresh = [i for i in ids if i > watermark]
        new_ids.extend(fresh)
        # A page that includes ids at/below the watermark means we've crossed
        # into already-known territory; nothing newer remains.
        if len(fresh) < len(ids) or not page_data["pageInfo"]["hasNextPage"]:
            break
    else:
        logger.warning(
            f"Discovery hit the {MAX_DISCOVERY_PAGES}-page cap before reaching "
            f"the watermark (id {watermark}); more than "
            f"~{MAX_DISCOVERY_PAGES * PER_PAGE} new anime — some ids may be missed."
        )
    return new_ids


def load_fetched_media_ids(media_csv):
    """Anime ids already written to media.csv, used to resume mid-scrape."""
    if not os.path.exists(media_csv):
        return set()
    try:
        return set(pd.read_csv(media_csv, usecols=["id"])["id"].astype(int))
    except ValueError:
        return set()


def append_media(rows, media_csv):
    if not rows:
        return
    pd.DataFrame(rows, columns=MEDIA_COLUMNS).to_csv(
        media_csv, mode="a", header=not os.path.exists(media_csv), index=False
    )


def _entries_via_collection(user_id, limiter):
    """Chunked MediaListCollection fetch. Returns a {mediaId: entry} dict, or
    None for a private/deleted account. Raises RuntimeError if the endpoint
    keeps erroring (some lists reproducibly 500 here). Uses a small retry
    budget so a reproducible failure falls back quickly instead of stalling.
    """
    entries = {}
    for chunk in range(1, MAX_CHUNKS + 1):
        response = make_request(
            USER_LIST_QUERY,
            {"userId": user_id, "chunk": chunk, "perChunk": PER_CHUNK},
            limiter,
            max_failures=2,
        )
        if response is None:
            return None
        collection = (response.get("data") or {}).get("MediaListCollection")
        if collection is None:
            return None
        # Custom lists repeat entries from the status lists; dedupe by media id.
        for group in collection.get("lists") or []:
            for entry in group.get("entries") or []:
                entries.setdefault(entry["mediaId"], entry)
        if not collection.get("hasNextChunk"):
            break
    else:
        logger.warning(f"User {user_id} exceeded {MAX_CHUNKS} chunks; list truncated")
    return entries


def _entries_via_pages(user_id, limiter):
    """Paginated mediaList fetch — the fallback endpoint. Same return contract
    as _entries_via_collection."""
    entries = {}
    for page in range(1, MAX_LIST_PAGES + 1):
        response = make_request(
            USER_LIST_PAGE_QUERY,
            {"userId": user_id, "page": page, "perPage": PER_PAGE},
            limiter,
        )
        if response is None:
            return None
        page_data = (response.get("data") or {}).get("Page")
        if page_data is None:
            return None
        for entry in page_data.get("mediaList") or []:
            entries.setdefault(entry["mediaId"], entry)
        if not page_data["pageInfo"]["hasNextPage"]:
            break
    else:
        logger.warning(
            f"User {user_id} exceeded {MAX_LIST_PAGES} pages; list truncated"
        )
    return entries


def fetch_user_list(user_id, limiter):
    """Fetch a user's complete anime list, deduplicated by media id.

    Tries the chunked MediaListCollection endpoint first, then falls back to the
    paginated mediaList endpoint for users whose list reproducibly errors there
    (some lists 500 on MediaListCollection but resolve fine page by page).
    Returns [] for private, empty, or deleted accounts. Raises RuntimeError only
    if both endpoints fail, so the caller can skip the user and retry later.
    """
    try:
        entries = _entries_via_collection(user_id, limiter)
    except RuntimeError:
        logger.warning(
            f"User {user_id}: MediaListCollection failed; trying paginated mediaList"
        )
        entries = _entries_via_pages(user_id, limiter)
    if not entries:
        return []
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


def scrape_media(
    limiter, data_dir, ids_csv, max_batches=None, fresh=False, discover=True
):
    """Fetch the anime catalogue by id, in batches of 50 via id_in.

    Reads the anime ids to fetch from a prior media.csv (ids_csv) and, unless
    discover is False, also pulls in anime added to AniList since that scrape
    (ids above the highest known id). Skips ids already present in the output and
    appends new rows in checkpoints so a crashed run resumes by rerunning.
    --fresh discards the existing output.
    """
    media_csv = os.path.join(data_dir, "media.csv")
    os.makedirs(data_dir, exist_ok=True)

    if os.path.abspath(ids_csv) == os.path.abspath(media_csv) and not fresh:
        raise SystemExit(
            f"--ids-csv and the output {media_csv} are the same file, so resume "
            "would treat every id as already fetched. Use a different --data-dir, "
            "or --fresh to rescrape from scratch."
        )

    # Load the work list before any --fresh deletion (the id source may itself
    # be the output of a previous run).
    all_ids = load_anime_ids(ids_csv)

    if discover:
        watermark = max(all_ids) if all_ids else 0
        new_ids = discover_new_media_ids(limiter, watermark)
        logger.info(f"Discovered {len(new_ids)} anime newer than id {watermark}")
        all_ids = sorted(set(all_ids) | set(new_ids))

    if fresh and os.path.exists(media_csv):
        os.remove(media_csv)
        logger.info(f"Removed {media_csv}")

    fetched = load_fetched_media_ids(media_csv)
    todo = [i for i in all_ids if i not in fetched]
    batches = [
        todo[i : i + MEDIA_BATCH_SIZE] for i in range(0, len(todo), MEDIA_BATCH_SIZE)
    ]
    if max_batches is not None:
        batches = batches[:max_batches]
    logger.info(
        f"{len(all_ids)} anime ids; {len(fetched)} already fetched; "
        f"{len(todo)} to fetch in {len(batches)} batches of {MEDIA_BATCH_SIZE}"
    )

    buffer = []
    batches_done = 0
    pbar = tqdm(total=len(batches), desc="Fetching media", unit="batch")
    try:
        for batch in batches:
            response = make_request(
                MEDIA_QUERY, {"ids": batch, "perPage": MEDIA_BATCH_SIZE}, limiter
            )
            # A null Page can be a site-wide outage or a transient blip; tell
            # them apart with a liveness probe before failing the whole run.
            if response is None or not (response.get("data") or {}).get("Page"):
                if not wait_until_healthy(limiter):
                    raise RuntimeError(
                        f"Media fetch failed and the API stayed down "
                        f"(batch starting at id {batch[0]})"
                    )
                response = make_request(
                    MEDIA_QUERY, {"ids": batch, "perPage": MEDIA_BATCH_SIZE}, limiter
                )
                if response is None or not (response.get("data") or {}).get("Page"):
                    raise RuntimeError(
                        f"Media fetch failed permanently for batch starting at "
                        f"id {batch[0]}"
                    )
            buffer.extend(response["data"]["Page"]["media"])
            batches_done += 1
            pbar.update(1)

            # Checkpoint: append harvested rows; resume skips them via the
            # ids already in media.csv.
            if batches_done % MEDIA_CHECKPOINT_BATCHES == 0:
                append_media(buffer, media_csv)
                logger.info(f"Checkpointed {len(buffer)} media records to {media_csv}")
                buffer.clear()
    except KeyboardInterrupt:
        logger.info("Interrupted — checkpointing; rerun to resume.")
    finally:
        append_media(buffer, media_csv)
        buffer.clear()
        pbar.close()

    logger.info(f"Media scrape finished; output at {media_csv}")


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
        help="discard previous outputs and start over (users: also resets the "
        "discovery state; media: deletes media.csv before refetching).",
    )
    parser.add_argument(
        "--ids-csv",
        default="../data/raw/media.csv",
        help="media only: prior media.csv supplying the anime ids to (re)fetch.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="media only: stop after this many 50-id batches (e.g. for testing).",
    )
    parser.add_argument(
        "--no-discover",
        action="store_true",
        help="media only: skip discovering anime added since the id source was "
        "built; only (re)fetch ids already in --ids-csv.",
    )
    args = parser.parse_args()

    limiter = RateLimiter()
    if args.target == "users":
        scrape_users(limiter, args.data_dir, max_pages=args.max_pages, fresh=args.fresh)
    else:
        scrape_media(
            limiter,
            args.data_dir,
            args.ids_csv,
            max_batches=args.max_batches,
            fresh=args.fresh,
            discover=not args.no_discover,
        )
