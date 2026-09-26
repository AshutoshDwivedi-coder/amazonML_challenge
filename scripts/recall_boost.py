"""Re-run high-impact blocks with looser caps; merge into existing union; re-eval hybrid."""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import duckdb

from src.blocking import blocks as B
from src.blocking.candidate_generator import _spill_pairs
from src.blocking.config import load_config
from src.blocking.evaluate_full import evaluate_union_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("recall_boost")

ROOT = Path(r"c:\Users\Lenovo\OneDrive\Desktop\amazon ML")
# original normalized data still on C:
S1 = ROOT / "outputs/work/s1_normalized.parquet"
S2 = ROOT / "outputs/work/s2_normalized.parquet"
S3 = ROOT / "outputs/work/s3_normalized.parquet"
OLD_PAIRS = ROOT / "outputs/work/block_pairs"
NEW_PAIRS = Path(r"D:\amazon_ml_blocking\work\block_pairs_v2")
OUT = Path(r"D:\amazon_ml_blocking\outputs")
WORK = Path(r"D:\amazon_ml_blocking\work")
GT = ROOT / "dataset/train/train_ground_truth.tsv"


def esc(p: Path) -> str:
    return p.resolve().as_posix().replace("'", "''")


def main() -> int:
    cfg = load_config(ROOT / "configs/blocking.yaml")
    NEW_PAIRS.mkdir(parents=True, exist_ok=True)

    # Copy unchanged lower-volume blocks from v1
    keep = [
        "exact_name_address",
        "postal",
        "rare_name_token",
        "rare_address_token",
        "address_token",
        "char_ngram",
        "rapidfuzz",
    ]
    for name in keep:
        for src in ("S2", "S3"):
            src_p = OLD_PAIRS / f"{name}__{src}.parquet"
            if src_p.exists() and src_p.stat().st_size > 0:
                dst = NEW_PAIRS / src_p.name
                if not dst.exists():
                    log.info("copy %s", src_p.name)
                    shutil.copy2(src_p, dst)

    # Re-run boosted blocks
    boost = [
        ("exact_name", B.block_exact_name),
        ("exact_address", B.block_exact_address),
        ("house_number_name", B.block_house_number_name),
        ("name_prefix", B.block_name_prefix),
    ]
    for block_name, fn in boost:
        for source, cand in (("S2", S2), ("S3", S3)):
            log.info("RUN %s %s", block_name, source)
            result = fn(str(S1), str(cand), source, cfg)
            log.info(
                "  -> %s pairs, %.1fs",
                f"{len(result.pairs):,}",
                result.runtime_seconds,
            )
            out_p = NEW_PAIRS / f"{block_name}__{source}.parquet"
            if result.pairs:
                _spill_pairs(result, out_p)
            else:
                # empty marker
                import pyarrow as pa
                import pyarrow.parquet as pq
                from src.blocking.candidate_generator import PAIRS_SCHEMA

                pq.write_table(pa.Table.from_pylist([], schema=PAIRS_SCHEMA), out_p)

    # Union DISTINCT on D:
    files = sorted(p for p in NEW_PAIRS.glob("*.parquet") if p.stat().st_size > 0)
    log.info("Union %s files…", len(files))
    db = WORK / "union_v2.duckdb"
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
    con.execute(
        "CREATE TABLE pairs (source1_entity_id VARCHAR, candidate_entity_id VARCHAR, source VARCHAR)"
    )
    for f in files:
        log.info("ingest %s", f.name)
        con.execute(
            f"""
            INSERT INTO pairs
            SELECT source1_entity_id, candidate_entity_id, source
            FROM read_parquet('{esc(f)}')
            """
        )
    union_pq = WORK / "candidate_pairs_union_v2.parquet"
    con.execute(
        f"""
        COPY (
          SELECT DISTINCT source1_entity_id, candidate_entity_id, source FROM pairs
        ) TO '{esc(union_pq)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    n_pairs, n_s1 = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT source1_entity_id) FROM read_parquet('{esc(union_pq)}')"
    ).fetchone()
    log.info("v2 union pairs=%s s1=%s avg=%.2f", f"{n_pairs:,}", f"{n_s1:,}", n_pairs / max(n_s1, 1))

    long_tsv = OUT / "candidate_pairs.tsv"
    log.info("Writing long TSV…")
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
    con.close()
    db.unlink(missing_ok=True)

    # Replace primary union pointer
    primary = WORK / "candidate_pairs_union.parquet"
    if primary.exists():
        primary.unlink()
    shutil.copy2(union_pq, primary)

    hybrid = evaluate_union_parquet(union_pq, GT, block_name="hybrid_v2")
    log.info(
        "HYBRID_V2 recall=%.4f s1_full=%.4f avg=%.2f p95=%.1f max=%s pairs=%s",
        hybrid.pair_candidate_recall,
        hybrid.s1_full_recall,
        hybrid.avg_candidates_per_s1,
        hybrid.p95_candidates_per_s1,
        hybrid.max_candidates_per_s1,
        f"{hybrid.total_candidate_pairs:,}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
