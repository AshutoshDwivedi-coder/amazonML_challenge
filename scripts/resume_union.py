"""Resume union on D: drive (C: is near full). Distinct-only; optional rules later."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("resume_union")

PAIRS_DIR = Path(r"c:\Users\Lenovo\OneDrive\Desktop\amazon ML\outputs\work\block_pairs")
S1_PQ = Path(r"c:\Users\Lenovo\OneDrive\Desktop\amazon ML\outputs\work\s1_normalized.parquet")
OUT_DIR = Path(r"D:\amazon_ml_blocking\outputs")
WORK = Path(r"D:\amazon_ml_blocking\work")


def esc(p: Path) -> str:
    return p.resolve().as_posix().replace("'", "''")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in PAIRS_DIR.glob("*.parquet") if p.stat().st_size > 0)
    log.info("Spilled files: %s", len(files))

    db = WORK / "union.duckdb"
    for p in (db, Path(str(db) + ".wal")):
        p.unlink(missing_ok=True)
    tmp = WORK / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)

    con = duckdb.connect(str(db))
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=2")
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET temp_directory='{esc(tmp)}'")
    con.execute("PRAGMA max_temp_directory_size='100GiB'")

    # Distinct pairs only (no string_agg) — far cheaper / smaller
    con.execute(
        """
        CREATE TABLE pairs (
          source1_entity_id VARCHAR,
          candidate_entity_id VARCHAR,
          source VARCHAR
        )
        """
    )
    for f in files:
        log.info("Ingest %s (%.1f MB)", f.name, f.stat().st_size / 1e6)
        con.execute(
            f"""
            INSERT INTO pairs
            SELECT source1_entity_id, candidate_entity_id, source
            FROM read_parquet('{esc(f)}')
            """
        )

    union_pq = WORK / "candidate_pairs_union.parquet"
    log.info("Writing DISTINCT union parquet…")
    con.execute(
        f"""
        COPY (
          SELECT DISTINCT source1_entity_id, candidate_entity_id, source
          FROM pairs
        ) TO '{esc(union_pq)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    n_pairs, n_s1 = con.execute(
        f"""
        SELECT COUNT(*), COUNT(DISTINCT source1_entity_id)
        FROM read_parquet('{esc(union_pq)}')
        """
    ).fetchone()
    log.info("Union: pairs=%s unique_s1=%s avg=%.2f", f"{n_pairs:,}", f"{n_s1:,}", n_pairs / max(n_s1, 1))

    # Long TSV for Sudhanshu (on D:, then copy lightweight pointer note)
    long_tsv = OUT_DIR / "candidate_pairs.tsv"
    log.info("Writing long TSV → %s", long_tsv)
    con.execute(
        f"""
        COPY (
          SELECT source1_entity_id, candidate_entity_id, source,
                 '' AS blocking_rules,
                 ROW_NUMBER() OVER (
                   PARTITION BY source1_entity_id
                   ORDER BY source, candidate_entity_id
                 ) AS candidate_rank
          FROM read_parquet('{esc(union_pq)}')
        ) TO '{esc(long_tsv)}' (HEADER, DELIMITER '\t')
        """
    )

    # Official submission format
    sub_tsv = OUT_DIR / "candidate_pairs_submission.tsv"
    log.info("Writing submission TSV → %s", sub_tsv)
    con.execute(
        f"""
        COPY (
          SELECT s.entity_id AS source1_entity_id,
                 COALESCE(
                   string_agg(c.candidate_entity_id, ',' ORDER BY c.candidate_entity_id),
                   ''
                 ) AS candidate_entity_ids
          FROM (SELECT entity_id FROM read_parquet('{esc(S1_PQ)}')) s
          LEFT JOIN read_parquet('{esc(union_pq)}') c
            ON s.entity_id = c.source1_entity_id
          GROUP BY s.entity_id
          ORDER BY s.entity_id
        ) TO '{esc(sub_tsv)}' (HEADER, DELIMITER '\t')
        """
    )

    con.close()
    # free D: db if desired later; keep union parquet
    db.unlink(missing_ok=True)
    Path(str(db) + ".wal").unlink(missing_ok=True)

    # Symlink/copy small pointer files into repo outputs/
    repo_out = Path(r"c:\Users\Lenovo\OneDrive\Desktop\amazon ML\outputs")
    # Create junction/symlink for the large TSV if possible, else write a README pointer
    pointer = repo_out / "CANDIDATE_PAIRS_LOCATION.txt"
    pointer.write_text(
        f"Large artefacts written to D: (C: disk full).\n\n"
        f"Long pairs: {long_tsv}\n"
        f"Submission: {sub_tsv}\n"
        f"Union parquet: {union_pq}\n",
        encoding="utf-8",
    )
    # Try to hardlink/copy submission (smaller-ish) — may still be large
    log.info("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
