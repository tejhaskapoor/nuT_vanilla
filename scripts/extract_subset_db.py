"""
Extract a balanced 200k-event subset from a large Prometheus SQLite database.

Creates:
  - merged_100k.db            : SQLite DB with 100k track + 100k cascade events
  - merged_100k_selection.parquet : parquet with all event_nos in the new DB
                                    (drop-in replacement for the original selection parquet)

Usage:
    python extract_subset_db.py \
        --src  /path/to/merged.db \
        --dst  /path/to/merged_100k.db \
        --n    100000 \
        --pulse-table merged_photons \
        --truth-table mc_truth \
        --seed 42
"""

import argparse
import logging
import os
import random
import sqlite3

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def fetch_balanced_event_nos(
    src_conn: sqlite3.Connection,
    truth_table: str,
    n_each: int,
    seed: int,
) -> list[int]:
    """Return n_each track + n_each cascade event_nos from the truth table."""
    df = pd.read_sql_query(
        f"SELECT event_no, initial_state_type, interaction FROM {truth_table}", src_conn
    )
    log.info(f"Total events in source DB: {len(df)}")

    is_track = (df["initial_state_type"].abs() == 14) & (df["interaction"] == 1)
    tracks   = df[is_track]["event_no"].astype(int).tolist()
    cascades = df[~is_track]["event_no"].astype(int).tolist()
    log.info(f"  Tracks available  : {len(tracks)}")
    log.info(f"  Cascades available: {len(cascades)}")

    if len(tracks) < n_each:
        raise ValueError(f"Requested {n_each} tracks but only {len(tracks)} available.")
    if len(cascades) < n_each:
        raise ValueError(f"Requested {n_each} cascades but only {len(cascades)} available.")

    rng = random.Random(seed)
    selected_tracks   = rng.sample(tracks,   n_each)
    selected_cascades = rng.sample(cascades, n_each)

    log.info(f"Sampled {n_each} tracks + {n_each} cascades = {2 * n_each} events total")
    return selected_tracks + selected_cascades


def copy_table_schema(src_conn: sqlite3.Connection, dst_conn: sqlite3.Connection, table: str) -> None:
    """Recreate the CREATE TABLE statement from source in destination."""
    schema_row = src_conn.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if schema_row is None:
        raise RuntimeError(f"Table '{table}' not found in source database.")
    dst_conn.execute(schema_row[0])


def copy_rows_chunked(
    src_conn: sqlite3.Connection,
    dst_conn: sqlite3.Connection,
    table: str,
    event_nos: list[int],
    chunk_size: int = 5_000,
) -> None:
    """Copy rows for the given event_nos from src to dst in chunks."""
    total_rows = 0
    for i in range(0, len(event_nos), chunk_size):
        chunk = event_nos[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        df = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE event_no IN ({placeholders})",
            src_conn,
            params=chunk,
        )
        df.to_sql(table, dst_conn, if_exists="append", index=False)
        total_rows += len(df)
        log.info(f"  [{table}] copied events {i}–{i + len(chunk) - 1}  ({total_rows} rows so far)")
    log.info(f"  [{table}] done — {total_rows} rows total")


def main(args: argparse.Namespace) -> None:
    if os.path.exists(args.dst):
        raise FileExistsError(
            f"Destination database already exists: {args.dst}\n"
            "Delete it first to avoid accidentally overwriting data."
        )

    log.info(f"Source DB : {args.src}")
    log.info(f"Target DB : {args.dst}")

    src_conn = sqlite3.connect(args.src)
    dst_conn = sqlite3.connect(args.dst)

    try:
        # ── 1. Select balanced event_nos ─────────────────────────────────────
        event_nos = fetch_balanced_event_nos(
            src_conn, args.truth_table, args.n, args.seed
        )

        # ── 2. Copy mc_truth rows ────────────────────────────────────────────
        log.info("Copying truth table …")
        copy_table_schema(src_conn, dst_conn, args.truth_table)
        copy_rows_chunked(src_conn, dst_conn, args.truth_table, event_nos)

        # ── 3. Copy pulse rows ───────────────────────────────────────────────
        log.info("Copying pulse table (this may take a while) …")
        copy_table_schema(src_conn, dst_conn, args.pulse_table)
        copy_rows_chunked(src_conn, dst_conn, args.pulse_table, event_nos, chunk_size=2_000)

        dst_conn.commit()
        log.info("Database written successfully.")

        # ── 4. Write matching selection parquet ──────────────────────────────
        parquet_path = args.dst.replace(".db", "_selection.parquet")
        pd.DataFrame({"event_no": event_nos}).to_parquet(parquet_path, index=False)
        log.info(f"Selection parquet written: {parquet_path}")

        # ── 5. Quick sanity check ────────────────────────────────────────────
        n_truth = dst_conn.execute(f"SELECT COUNT(*) FROM {args.truth_table}").fetchone()[0]
        n_pulse = dst_conn.execute(f"SELECT COUNT(*) FROM {args.pulse_table}").fetchone()[0]
        log.info(f"Sanity check — truth rows: {n_truth}, pulse rows: {n_pulse}")

    finally:
        src_conn.close()
        dst_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src",         required=True,  help="Path to source .db file")
    parser.add_argument("--dst",         required=True,  help="Path for output merged_100k.db")
    parser.add_argument("--n",           type=int, default=100_000, help="Events per class (default: 100000)")
    parser.add_argument("--pulse-table", default="merged_photons",  help="Pulse/hit table name")
    parser.add_argument("--truth-table", default="mc_truth",        help="Truth table name")
    parser.add_argument("--seed",        type=int, default=42,       help="Random seed")
    main(parser.parse_args())
