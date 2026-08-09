"""Merge a hydrated user_anime_lists/ Parquet dataset into one Parquet file.

The hydrator writes the dataset as many files — part-* (one per checkpoint),
compact-* (merged batches of those), and legacy-* (migrated from an old CSV).
This consolidates them into a single Parquet file you can load directly, e.g.
    pd.read_parquet("user_anime_lists.parquet")

Streams one source file at a time through a single writer, so it runs in
bounded memory regardless of how big the dataset is. Two correctness guards:
  - part-* files already absorbed into a compact-LO-HI range are skipped, so a
    directory copied mid-compaction (compact written, sources not yet deleted)
    does not double its rows;
  - half-written *.tmp files are ignored.

Every kept (user, anime) row therefore appears exactly once.

Usage (from inside scripts/):
  uv run python merge_lists.py path/to/user_anime_lists
  uv run python merge_lists.py path/to/user_anime_lists --output lists.parquet
"""

import argparse
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq

PART_NAME = re.compile(r"part-(\d+)\.parquet")
COMPACT_NAME = re.compile(r"compact-(\d+)-(\d+)\.parquet")

# The hydrated list schema (POINT_100 score, 0 = unscored). Every source file is
# read into exactly this — column order and dtypes — so the output is uniform
# even if files drift in arrow string encoding across crawl sessions.
COLUMNS = ["user_id", "media_id", "status", "score", "progress", "repeat"]
SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("media_id", pa.int64()),
        ("status", pa.string()),
        ("score", pa.int64()),
        ("progress", pa.int64()),
        ("repeat", pa.int64()),
    ]
)


def source_files(input_dir):
    """Parquet files to merge: every compact-* and legacy-*, plus part-* files
    not already covered by a compacted range. .tmp and unrelated files skipped.
    """
    names = os.listdir(input_dir)
    covered = [
        (int(m.group(1)), int(m.group(2)))
        for n in names
        if (m := COMPACT_NAME.fullmatch(n))
    ]
    keep = []
    for n in sorted(names):
        if not n.endswith(".parquet"):
            continue  # ignore .tmp leftovers and anything non-Parquet
        m = PART_NAME.fullmatch(n)
        if m and any(lo <= int(m.group(1)) <= hi for lo, hi in covered):
            continue  # already inside a compacted file — would duplicate rows
        keep.append(os.path.join(input_dir, n))
    return keep


def merge(input_dir, output_path):
    if not os.path.isdir(input_dir):
        raise SystemExit(f"{input_dir} is not a directory")
    files = source_files(input_dir)
    if not files:
        raise SystemExit(f"No Parquet files to merge in {input_dir}")

    tmp = output_path + ".tmp"
    rows = 0
    writer = pq.ParquetWriter(tmp, SCHEMA)
    try:
        for path in files:
            # select+cast normalises column order and arrow types to SCHEMA.
            table = pq.read_table(path).select(COLUMNS).cast(SCHEMA)
            writer.write_table(table)
            rows += table.num_rows
            print(f"  + {os.path.basename(path):<28} {table.num_rows:>10,} rows")
    finally:
        writer.close()
    os.replace(tmp, output_path)  # publish atomically
    print(f"\nMerged {len(files)} files into {output_path} ({rows:,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge a user_anime_lists/ Parquet dataset into one file"
    )
    parser.add_argument("input_dir", help="The user_anime_lists/ dataset directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output Parquet file (default: <input_dir>.parquet alongside it).",
    )
    args = parser.parse_args()
    output = args.output or os.path.normpath(args.input_dir) + ".parquet"
    merge(args.input_dir, output)
