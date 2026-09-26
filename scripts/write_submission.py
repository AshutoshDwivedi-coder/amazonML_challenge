"""Build official submission candidate_pairs from long TSV without huge string_agg."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("submission")

UNION = Path(r"D:\amazon_ml_blocking\work\candidate_pairs_union.parquet")
S1 = Path(r"c:\Users\Lenovo\OneDrive\Desktop\amazon ML\outputs\work\s1_normalized.parquet")
OUT = Path(r"D:\amazon_ml_blocking\outputs\candidate_pairs_submission.tsv")
WORK = Path(r"D:\amazon_ml_blocking\work")


def esc(p: Path) -> str:
    return p.resolve().as_posix().replace("'", "''")


def main() -> None:
    # Free leftover DB if present
    for p in WORK.glob("union.duckdb*"):
        try:
            p.unlink()
        except OSError:
            pass

    db = WORK / "sub.duckdb"
    db.unlink(missing_ok=True)
    con = duckdb.connect(str(db))
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=1")
    con.execute("SET memory_limit='4GB'")
    tmp = WORK / "duckdb_tmp"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{esc(tmp)}'")
    con.execute("PRAGMA max_temp_directory_size='100GiB'")

    # Pre-aggregate candidates per S1 into a compact parquet (list_agg may still be heavy);
    # instead write sorted (s1, cand) then stream in Python.
    sorted_pq = WORK / "pairs_sorted.parquet"
    log.info("Sorting pairs by S1…")
    con.execute(
        f"""
        COPY (
          SELECT source1_entity_id, candidate_entity_id
          FROM read_parquet('{esc(UNION)}')
          ORDER BY source1_entity_id, candidate_entity_id
        ) TO '{esc(sorted_pq)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.close()

    log.info("Streaming submission TSV…")
    con2 = duckdb.connect(":memory:")
    con2.execute("SET threads=1")
    # Stream in batches via fetchmany
    cur = con2.execute(
        f"SELECT source1_entity_id, candidate_entity_id FROM read_parquet('{esc(sorted_pq)}')"
    )

    s1_ids = [
        r[0]
        for r in duckdb.connect(":memory:")
        .execute(f"SELECT entity_id FROM read_parquet('{esc(S1)}') ORDER BY 1")
        .fetchall()
    ]
    s1_set_order = s1_ids
    # Build map by streaming
    current = None
    bucket: list[str] = []
    mapping: dict[str, str] = {}

    def flush(s1: str, cands: list[str]) -> None:
        mapping[s1] = ",".join(cands)

    while True:
        rows = cur.fetchmany(500_000)
        if not rows:
            break
        for s1, cand in rows:
            if current is None:
                current = s1
            if s1 != current:
                flush(current, bucket)
                current = s1
                bucket = [cand]
            else:
                bucket.append(cand)
        log.info("streamed… map_size=%s", f"{len(mapping):,}")
    if current is not None:
        flush(current, bucket)

    log.info("Writing %s rows…", f"{len(s1_set_order):,}")
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_set_order:
            f.write(f"{s1}\t{mapping.get(s1, '')}\n")
    log.info("Wrote %s (%.2f GB)", OUT, OUT.stat().st_size / 1e9)

    # cleanup
    sorted_pq.unlink(missing_ok=True)
    db.unlink(missing_ok=True)
    Path(str(db) + ".wal").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
