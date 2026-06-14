"""AniList user scrape.

Reads user ids from an existing Page.users scrape and pulls each user's
complete anime list via the MediaListCollection endpoint. Users whose stats
show no activity (zero minutes watched, empty scores and statuses buckets)
are skipped.

Users whose hydration comes back empty (private, deleted or empty lists) are
dropped and remembered in the state file so they are never polled again.
Harvested rows are stored as a Parquet dataset: every checkpoint writes
one immutable part file, so re-running the script resumes from where it left
off; once --compact-every part files pile up they are merged into one larger
compacted file to keep the file count bounded. A user_anime_lists.csv left
over from a pre-Parquet run is converted into part files on startup and
renamed to user_anime_lists.csv.migrated.

Outputs (in --out-dir, default: the directory of --users-csv):
  user_anime_lists/     Parquet dataset, one part file per checkpoint; one row
                        per (user, anime) list entry; score uses the POINT_100
                        scale, 0 = unscored.
  hydrate_state.json    ids of users polled but found empty/private
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys

import pandas as pd
from fetch_data import (
    ENTRY_COLUMNS,
    RateLimiter,
    fetch_user_list,
    save_state,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Stats prefix of an account with no activity whatsoever. The old crawl wrote
# statistics dicts in query-field order, so a fully-inactive user's string
# always starts exactly like this. False negatives only cost one API call.
INACTIVE_PREFIX = (
    "{'anime': {'meanScore': 0, 'minutesWatched': 0, 'scores': [], 'statuses': []"
)

CHECKPOINT_USERS = 25  # flush data + state every this many polled users
COMPACT_EVERY = 40  # merge per-checkpoint part files once this many accumulate
MIGRATE_CHUNK_ROWS = 1_000_000  # rows per part file when converting a legacy CSV

PART_NAME = re.compile(r"part-(\d+)\.parquet")
COMPACT_NAME = re.compile(r"compact-(\d+)-(\d+)\.parquet")


def load_empty_ids(state_path):
    if os.path.exists(state_path):
        with open(state_path) as f:
            return set(json.load(f).get("empty_ids", []))
    return set()


def normalize_entries(df):
    """Coerce dtypes so every part file carries an identical schema."""
    df = df.loc[:, ENTRY_COLUMNS].copy()
    for col in ENTRY_COLUMNS:
        if col == "status":
            df[col] = df[col].astype("string")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
    return df


def write_part(df, path):
    """Write one part file atomically — a crash never leaves a partial file."""
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def part_files(parts_dir):
    return sorted(
        os.path.join(parts_dir, name)
        for name in os.listdir(parts_dir)
        if name.endswith(".parquet")
    )


def live_parts(parts_dir):
    """Per-checkpoint part files not yet compacted, keyed by part index."""
    return {
        int(m.group(1)): os.path.join(parts_dir, name)
        for name in os.listdir(parts_dir)
        if (m := PART_NAME.fullmatch(name))
    }


def next_part_index(parts_dir):
    indices = [-1]
    for name in os.listdir(parts_dir):
        if m := PART_NAME.fullmatch(name):
            indices.append(int(m.group(1)))
        elif m := COMPACT_NAME.fullmatch(name):
            indices.append(int(m.group(2)))
    return max(indices) + 1


def compact_parts(parts_dir):
    """Merge the accumulated per-checkpoint part files into one file.

    The compacted file's name records the part-index range it absorbed, so a
    crash between writing it and deleting its sources is healed by
    clean_parts_dir() on the next startup instead of duplicating rows.
    """
    parts = live_parts(parts_dir)
    if len(parts) < 2:
        return
    lo, hi = min(parts), max(parts)
    df = pd.concat(
        [pd.read_parquet(parts[i]) for i in sorted(parts)], ignore_index=True
    )
    write_part(df, os.path.join(parts_dir, f"compact-{lo:05d}-{hi:05d}.parquet"))
    for path in parts.values():
        os.remove(path)
    logger.info(
        f"Compacted {len(parts)} part files into "
        f"compact-{lo:05d}-{hi:05d}.parquet ({len(df)} rows)"
    )


def clean_parts_dir(parts_dir):
    """Heal crash leftovers: stray tmp files and parts already compacted."""
    covered = [
        (int(m.group(1)), int(m.group(2)))
        for name in os.listdir(parts_dir)
        if (m := COMPACT_NAME.fullmatch(name))
    ]
    for name in os.listdir(parts_dir):
        path = os.path.join(parts_dir, name)
        if name.endswith(".tmp"):
            os.remove(path)
        elif (m := PART_NAME.fullmatch(name)) and any(
            lo <= int(m.group(1)) <= hi for lo, hi in covered
        ):
            os.remove(path)
            logger.info(f"Dropped {name} (covered by a compacted file)")


def load_harvested_ids(parts_dir):
    """User ids already in the dataset, used to skip replayed work on resume."""
    ids = set()
    for path in part_files(parts_dir):
        ids.update(pd.read_parquet(path, columns=["user_id"])["user_id"])
    return ids


def migrate_legacy_csv(legacy_csv, parts_dir):
    """Convert user_anime_lists.csv from a pre-Parquet run into part files.

    Restart-safe: the CSV is renamed away only after the last part is written,
    so an interrupted conversion is discarded and redone on the next run.
    """
    header = pd.read_csv(legacy_csv, nrows=0).columns
    if not set(ENTRY_COLUMNS) <= set(header):
        raise SystemExit(
            f"{legacy_csv} is incompatible with this scraper (old format?). "
            "Move the file away or rerun with --fresh."
        )
    for name in os.listdir(parts_dir):
        if name.startswith("legacy-"):
            os.remove(os.path.join(parts_dir, name))
    rows = parts = 0
    for chunk in pd.read_csv(legacy_csv, chunksize=MIGRATE_CHUNK_ROWS):
        if chunk.empty:
            continue
        write_part(
            normalize_entries(chunk),
            os.path.join(parts_dir, f"legacy-{parts:05d}.parquet"),
        )
        rows += len(chunk)
        parts += 1
    os.replace(legacy_csv, legacy_csv + ".migrated")
    logger.info(
        f"Migrated {rows} rows from {legacy_csv} into {parts} Parquet part(s); "
        f"original kept as {legacy_csv}.migrated"
    )


def iter_user_ids(users_csv, skip_inactive=True):
    """Stream (user_id, is_inactive) from the discovery crawl."""
    csv.field_size_limit(sys.maxsize)
    with open(users_csv, newline="") as f:
        for row in csv.DictReader(f):
            stats = row.get("statistics") or ""
            inactive = skip_inactive and stats.startswith(INACTIVE_PREFIX)
            yield int(row["id"]), inactive


def hydrate(
    users_csv,
    out_dir,
    max_users=None,
    fresh=False,
    skip_inactive=True,
    compact_every=COMPACT_EVERY,
):
    parts_dir = os.path.join(out_dir, "user_anime_lists")
    legacy_csv = os.path.join(out_dir, "user_anime_lists.csv")
    state_path = os.path.join(out_dir, "hydrate_state.json")

    if fresh:
        for path in (legacy_csv, state_path):
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed {path}")
        if os.path.isdir(parts_dir):
            shutil.rmtree(parts_dir)
            logger.info(f"Removed {parts_dir}")

    os.makedirs(parts_dir, exist_ok=True)
    clean_parts_dir(parts_dir)
    if os.path.exists(legacy_csv):
        migrate_legacy_csv(legacy_csv, parts_dir)

    harvested = load_harvested_ids(parts_dir)
    empty_ids = load_empty_ids(state_path)
    if harvested or empty_ids:
        logger.info(
            f"Resuming: {len(harvested)} users harvested, {len(empty_ids)} known empty"
        )

    limiter = RateLimiter()
    entry_buffer = []
    next_part = next_part_index(parts_dir)
    polled = kept = inactive_count = 0
    polled_since_flush = 0

    def checkpoint():
        # Flush data first, then the state; on a crash in between, the
        # harvested-ids guard skips the re-polled users on resume.
        nonlocal next_part, polled_since_flush
        if entry_buffer:
            df = normalize_entries(pd.DataFrame(entry_buffer, columns=ENTRY_COLUMNS))
            write_part(df, os.path.join(parts_dir, f"part-{next_part:05d}.parquet"))
            next_part += 1
            entry_buffer.clear()
            if compact_every and len(live_parts(parts_dir)) >= compact_every:
                compact_parts(parts_dir)
        save_state({"empty_ids": sorted(empty_ids)}, state_path)
        polled_since_flush = 0

    pbar = tqdm(iter_user_ids(users_csv, skip_inactive), desc="Hydrating", unit="user")
    try:
        for user_id, is_inactive in pbar:
            if user_id in harvested or user_id in empty_ids:
                continue
            if is_inactive:
                inactive_count += 1
                continue
            if max_users is not None and polled >= max_users:
                break

            rows = fetch_user_list(user_id, limiter)
            polled += 1
            polled_since_flush += 1
            if rows:
                entry_buffer.extend(rows)
                harvested.add(user_id)
                kept += 1
            else:
                empty_ids.add(user_id)  # private or empty list — drop the user

            if polled_since_flush >= CHECKPOINT_USERS:
                checkpoint()
                pbar.set_postfix(polled=polled, kept=kept, inactive=inactive_count)
    except KeyboardInterrupt:
        logger.info("Interrupted — flushing progress; rerun to resume.")
    finally:
        checkpoint()
        pbar.close()

    logger.info(
        f"Done: polled {polled} users, kept {kept}, "
        f"{polled - kept} empty/private, {inactive_count} skipped as inactive"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hydrate user anime lists from an existing Page.users crawl"
    )
    parser.add_argument(
        "--users-csv",
        default="../data/users.csv",
        help="Discovery crawl to read user ids from.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: directory of --users-csv).",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Stop after polling this many users (default: run to the end).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard previous output and state, start from the top.",
    )
    parser.add_argument(
        "--no-skip-inactive",
        action="store_true",
        help="Also poll users whose crawled statistics show no activity.",
    )
    parser.add_argument(
        "--compact-every",
        type=int,
        default=COMPACT_EVERY,
        help="Merge part files once this many accumulate (0 disables).",
    )
    args = parser.parse_args()

    hydrate(
        args.users_csv,
        args.out_dir or os.path.dirname(os.path.abspath(args.users_csv)),
        max_users=args.max_users,
        fresh=args.fresh,
        skip_inactive=not args.no_skip_inactive,
        compact_every=args.compact_every,
    )
