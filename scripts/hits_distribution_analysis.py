"""Analyse the hits-per-event distribution in a Prometheus SQLite database.

For each sampled event this script records:
  - event_no          : event identifier
  - total_hits        : total number of photon hits in the raw pulsemap
  - signal_hits       : hits where is_signal == 1
  - noise_hits        : hits where is_signal == 0

Results are appended to a CSV file so repeated runs never re-process the
same events.  A summary (mean values) is printed at the end.

Usage
-----
python -m nuT_vanilla.scripts.hits_distribution_analysis \\
    --db       /path/to/merged.db \\
    --n-events 5000 \\
    --output   hits_distribution.csv \\
    [--pulsemap merged_photons] \\
    [--truth-table mc_truth]
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHUNK = 999  # SQLite SQLITE_MAX_VARIABLE_NUMBER limit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db",          required=True,  help="Path to the SQLite database file")
    p.add_argument("--n-events",    required=True,  type=int, help="Number of new events to sample and analyse")
    p.add_argument("--output",      required=True,  help="Path to the output CSV file (appended if it exists)")
    p.add_argument("--pulsemap",    default="merged_photons", help="Pulsemap table name (default: merged_photons)")
    p.add_argument("--truth-table", default="mc_truth",       help="Truth table name (default: mc_truth)")
    return p.parse_args()


def fetch_all_event_nos(conn: sqlite3.Connection, truth_table: str) -> np.ndarray:
    """Return every event_no present in the truth table."""
    cur = conn.cursor()
    cur.execute(f"SELECT event_no FROM {truth_table}")
    return np.array([r[0] for r in cur.fetchall()], dtype=np.int64)


def count_hits_for_events(
    conn: sqlite3.Connection,
    pulsemap: str,
    event_nos: list,
) -> dict:
    """Return {event_no: (total, signal, noise)} for all requested events."""
    results = {}
    cur = conn.cursor()
    for i in range(0, len(event_nos), CHUNK):
        chunk = event_nos[i : i + CHUNK]
        placeholders = ",".join("?" * len(chunk))
        cur.execute(
            f"SELECT event_no, is_signal FROM {pulsemap} WHERE event_no IN ({placeholders})",
            chunk,
        )
        for event_no, is_signal in cur.fetchall():
            if event_no not in results:
                results[event_no] = [0, 0, 0]   # [total, signal, noise]
            results[event_no][0] += 1
            if is_signal == 1:
                results[event_no][1] += 1
            else:
                results[event_no][2] += 1
    return results


def main():
    args = parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    output_path = Path(args.output)

    # Load already-processed event numbers so we never repeat them
    already_seen: set = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        already_seen = set(existing["event_no"].astype(np.int64).tolist())
        logger.info(f"CSV exists — {len(already_seen)} events already recorded, will skip them")
    else:
        existing = pd.DataFrame()

    conn = sqlite3.connect(str(db_path))

    all_event_nos = fetch_all_event_nos(conn, args.truth_table)
    logger.info(f"Total events in DB: {len(all_event_nos)}")

    # Filter out already-seen events and sample n_events from the remainder
    available = np.setdiff1d(all_event_nos, np.array(list(already_seen), dtype=np.int64))
    if len(available) == 0:
        logger.warning("All events in the DB have already been processed. Nothing to do.")
        conn.close()
        sys.exit(0)

    if args.n_events > len(available):
        logger.warning(
            f"Requested {args.n_events} events but only {len(available)} unseen events available. "
            f"Using all {len(available)}."
        )
        n = len(available)
    else:
        n = args.n_events

    rng = np.random.default_rng()
    sampled = rng.choice(available, size=n, replace=False).tolist()
    logger.info(f"Sampling {n} new events …")

    hit_counts = count_hits_for_events(conn, args.pulsemap, sampled)
    conn.close()

    # Build dataframe for this batch
    rows = []
    for event_no in sampled:
        if event_no in hit_counts:
            total, signal, noise = hit_counts[event_no]
        else:
            # Event exists in truth table but has no hits in pulsemap
            total, signal, noise = 0, 0, 0
        rows.append({"event_no": event_no, "total_hits": total, "signal_hits": signal, "noise_hits": noise})

    new_df = pd.DataFrame(rows, dtype=np.int64)

    # Append to (or create) the CSV
    if output_path.exists():
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(output_path, index=False)
    logger.info(f"Saved {len(combined)} total rows to {output_path}")

    # Summary over the newly processed batch
    print("\n--- Summary for this batch ---")
    print(f"  Events analysed : {len(new_df)}")
    print(f"  Avg total hits  : {new_df['total_hits'].mean():.1f}  (std {new_df['total_hits'].std():.1f})")
    print(f"  Avg signal hits : {new_df['signal_hits'].mean():.1f}  (std {new_df['signal_hits'].std():.1f})")
    print(f"  Avg noise hits  : {new_df['noise_hits'].mean():.1f}  (std {new_df['noise_hits'].std():.1f})")
    print(f"  Signal fraction : {new_df['signal_hits'].sum() / max(new_df['total_hits'].sum(), 1):.3f}")

    if len(already_seen) > 0:
        print("\n--- Cumulative summary (all runs) ---")
        print(f"  Events total    : {len(combined)}")
        print(f"  Avg total hits  : {combined['total_hits'].mean():.1f}  (std {combined['total_hits'].std():.1f})")
        print(f"  Avg signal hits : {combined['signal_hits'].mean():.1f}  (std {combined['signal_hits'].std():.1f})")
        print(f"  Avg noise hits  : {combined['noise_hits'].mean():.1f}  (std {combined['noise_hits'].std():.1f})")
        print(f"  Signal fraction : {combined['signal_hits'].sum() / max(combined['total_hits'].sum(), 1):.3f}")

    # Percentile table — useful for choosing max_hits
    print("\n--- Percentile breakdown of total_hits (new batch) ---")
    for pct in [50, 75, 90, 95, 99, 100]:
        print(f"  {pct:>3}th percentile : {np.percentile(new_df['total_hits'], pct):.0f} hits")


if __name__ == "__main__":
    main()
