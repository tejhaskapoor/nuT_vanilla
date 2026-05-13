"""
Copy the first N events from a Prometheus SQLite database.

Usage:
    python extract_first_n.py \
        --src  /path/to/merged.db \
        --dst  /path/to/merged_2k.db \
        --n    2000 \
        --pulse-table merged_photons \
        --truth-table mc_truth
"""

import argparse
import logging
import os
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def copy_table_schema(src_conn, dst_conn, table):
    row = src_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Table '{table}' not found in source database.")
    dst_conn.execute(row[0])


def main(args):
    if os.path.exists(args.dst):
        raise FileExistsError(f"Destination already exists: {args.dst}\nDelete it first.")

    log.info(f"Source : {args.src}")
    log.info(f"Target : {args.dst}")

    src_conn = sqlite3.connect(args.src)
    dst_conn = sqlite3.connect(args.dst)

    try:
        dst_conn.execute(f"ATTACH DATABASE ? AS src", (args.src,))

        # pick the first N event_nos from the truth table
        event_nos = [
            row[0] for row in src_conn.execute(
                f"SELECT event_no FROM {args.truth_table} LIMIT ?", (args.n,)
            )
        ]
        log.info(f"Selected {len(event_nos)} events")

        placeholders = ",".join("?" * len(event_nos))

        for table in (args.truth_table, args.pulse_table):
            copy_table_schema(src_conn, dst_conn, table)
            dst_conn.execute(
                f"INSERT INTO {table} SELECT * FROM src.{table} WHERE event_no IN ({placeholders})",
                event_nos,
            )
            n = dst_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            log.info(f"  {table}: {n} rows copied")

        dst_conn.execute("DETACH DATABASE src")
        dst_conn.commit()
        log.info("Done.")

    finally:
        src_conn.close()
        dst_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src",          required=True,          help="Source .db file")
    parser.add_argument("--dst",          required=True,          help="Output .db file")
    parser.add_argument("--n",            type=int, default=2000, help="Number of events (default: 2000)")
    parser.add_argument("--pulse-table",  default="merged_photons")
    parser.add_argument("--truth-table",  default="mc_truth")
    main(parser.parse_args())
