"""Candidate generation: union of blocking rules → deduped pairs for matching."""
from __future__ import annotations

import csv
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

from . import blocks as B
from .config import BlockingConfig, load_config
from .io_utils import ensure_normalized

logger = logging.getLogger(__name__)

PAIRS_SCHEMA = pa.schema(
    [
        ("source1_entity_id", pa.string()),
        ("candidate_entity_id", pa.string()),
        ("source", pa.string()),
        ("blocking_rule", pa.string()),
    ]
)


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _log_block(result: B.BlockResult) -> None:
    n = len(result.pairs)
    s1s = {p[0] for p in result.pairs}
    avg = (n / len(s1s)) if s1s else 0.0
    logger.info(
        "[BLOCK] %s source=%s candidates=%s unique_s1=%s avg=%.2f runtime=%.1fs mem=%.0fMB",
        result.block_name,
        result.source,
        f"{n:,}",
        f"{len(s1s):,}",
        avg,
        result.runtime_seconds,
        _rss_mb(),
    )


def _spill_pairs(result: B.BlockResult, path: Path) -> Path:
    """Write block pairs to parquet and free the in-memory list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source1_entity_id": s1,
            "candidate_entity_id": cand,
            "source": source,
            "blocking_rule": result.block_name,
        }
        for s1, cand, source in result.pairs
    ]
    table = pa.Table.from_pylist(rows, schema=PAIRS_SCHEMA)
    pq.write_table(table, path, compression="zstd")
    result.pairs_path = str(path)
    n = len(result.pairs)
    result.pairs = []  # free RAM
    logger.debug("Spilled %s pairs → %s", f"{n:,}", path)
    return path


class CandidateGenerator:
    """Production blocking / candidate-generation engine.

    Boundary: produces candidate pairs only. Does NOT decide true matches.
    """

    def __init__(self, config: BlockingConfig | None = None):
        self.config = config or BlockingConfig()
        self.block_results: list[B.BlockResult] = []
        self.covered_s1: set[str] = set()
        self._pair_files: list[str] = []

    def generate(
        self,
        s1_path: str | Path,
        s2_path: str | Path,
        s3_path: str | Path,
        output_path: str | Path,
        *,
        force_normalize: bool = False,
        write_dev_columns: bool = True,
        write_submission_format: bool = True,
        keep_pairs_in_memory: bool = False,
    ) -> Path:
        cfg = self.config
        work = Path(cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        pairs_dir = work / "block_pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t_all = time.perf_counter()
        logger.info("Normalizing sources (chunk_size=%s)…", cfg.chunk_size)
        s1_pq = ensure_normalized(
            s1_path, work, "s1", chunk_size=cfg.chunk_size, force=force_normalize
        )
        s2_pq = ensure_normalized(
            s2_path, work, "s2", chunk_size=cfg.chunk_size, force=force_normalize
        )
        s3_pq = ensure_normalized(
            s3_path, work, "s3", chunk_size=cfg.chunk_size, force=force_normalize
        )
        logger.info("Normalized parquet ready. peak_rss=%.0fMB", _rss_mb())

        sources = [("S2", str(s2_pq)), ("S3", str(s3_pq))]
        enabled = list(cfg.enabled_blocks)
        ordered = [
            "exact_name",
            "exact_address",
            "exact_name_address",
            "postal",
            "house_number_name",
            "rare_name_token",
            "rare_address_token",
            "name_prefix",
            "address_token",
            "char_ngram",
            "rapidfuzz",
        ]
        ordered = [b for b in ordered if b in enabled]

        self.block_results = []
        self.covered_s1 = set()
        self._pair_files = []

        for block_name in ordered:
            if block_name in ("char_ngram", "rapidfuzz"):
                residual = self._residual_s1_ids(str(s1_pq), self.covered_s1)
                logger.info(
                    "Approximate block %s on %s residual S1s",
                    block_name,
                    f"{len(residual):,}",
                )
                for source, cand_pq in sources:
                    if block_name == "char_ngram":
                        result = B.block_char_ngram(
                            str(s1_pq),
                            cand_pq,
                            source,
                            cfg,
                            only_s1_ids=residual if residual else None,
                        )
                    else:
                        if not cfg.rapidfuzz_enabled:
                            continue
                        result = B.block_rapidfuzz(
                            str(s1_pq),
                            cand_pq,
                            source,
                            cfg,
                            only_s1_ids=residual if cfg.rapidfuzz_only_unresolved else None,
                            candidate_pools=None,
                        )
                    _log_block(result)
                    self._ingest(result, pairs_dir, keep_pairs_in_memory)
                continue

            for source, cand_pq in sources:
                result = self._run_block(block_name, str(s1_pq), cand_pq, source)
                _log_block(result)
                self._ingest(result, pairs_dir, keep_pairs_in_memory)

        union_pq = work / "candidate_pairs_union.parquet"
        self._union_to_parquet(union_pq)
        self._write_long_from_parquet(
            union_pq, output_path, write_dev_columns=write_dev_columns
        )
        if write_submission_format:
            sub_path = output_path.with_name(
                output_path.stem + "_submission" + output_path.suffix
            )
            self._write_submission(sub_path, s1_parquet=str(s1_pq), union_pq=union_pq)

        elapsed = time.perf_counter() - t_all
        stats = self._union_stats(union_pq)
        logger.info(
            "DONE hybrid candidates=%s unique_s1=%s avg=%.2f runtime=%.1fs peak_rss=%.0fMB",
            f"{stats['n_pairs']:,}",
            f"{stats['n_s1']:,}",
            stats["avg"],
            elapsed,
            _rss_mb(),
        )
        return output_path

    def _ingest(
        self,
        result: B.BlockResult,
        pairs_dir: Path,
        keep_pairs_in_memory: bool,
    ) -> None:
        for s1, _c, _s in result.pairs:
            self.covered_s1.add(s1)
        # keep a copy of pairs for small evals if requested
        if keep_pairs_in_memory:
            self.block_results.append(result)
            return
        path = pairs_dir / f"{result.block_name}__{result.source}.parquet"
        n = len(result.pairs)
        if n:
            _spill_pairs(result, path)
            self._pair_files.append(str(path))
        # store metadata-only result (pairs already cleared by spill, or empty)
        meta = B.BlockResult(
            block_name=result.block_name,
            source=result.source,
            pairs=[],
            runtime_seconds=result.runtime_seconds,
            index_build_seconds=result.index_build_seconds,
            query_seconds=result.query_seconds,
            pairs_path=result.pairs_path,
        )
        # stash count on object for logging/eval
        meta._pair_count = n  # type: ignore[attr-defined]
        self.block_results.append(meta)

    def _run_block(
        self, name: str, s1_pq: str, cand_pq: str, source: str
    ) -> B.BlockResult:
        cfg = self.config
        dispatch = {
            "exact_name": B.block_exact_name,
            "exact_address": B.block_exact_address,
            "exact_name_address": B.block_exact_name_address,
            "postal": B.block_postal,
            "house_number_name": B.block_house_number_name,
            "rare_name_token": B.block_rare_name_token,
            "rare_address_token": B.block_rare_address_token,
            "name_prefix": B.block_name_prefix,
            "address_token": B.block_address_token,
        }
        if name not in dispatch:
            raise ValueError(f"Unknown block: {name}")
        return dispatch[name](s1_pq, cand_pq, source, cfg)

    def _residual_s1_ids(self, s1_pq: str, covered: set[str]) -> set[str]:
        con = duckdb.connect(":memory:")
        escaped = s1_pq.replace("'", "''")
        con.execute(
            f"CREATE VIEW s1 AS SELECT entity_id FROM read_parquet('{escaped}')"
        )
        all_ids = {r[0] for r in con.execute("SELECT entity_id FROM s1").fetchall()}
        return all_ids - covered

    def _union_to_parquet(self, out_path: Path) -> None:
        """Dedup spilled block pairs via a disk-backed DuckDB database.

        Avoids in-memory UNION of ~100M+ rows (which OOMs on temp spill limits).
        """
        if not self._pair_files:
            pq.write_table(pa.Table.from_pylist([], schema=PAIRS_SCHEMA), out_path)
            return

        work = Path(self.config.work_dir)
        db_path = work / "union.duckdb"
        if db_path.exists():
            db_path.unlink()
        # also clear WAL leftovers
        for extra in (str(db_path) + ".wal",):
            p = Path(extra)
            if p.exists():
                p.unlink()

        con = duckdb.connect(str(db_path))
        con.execute("SET preserve_insertion_order=false")
        con.execute("SET threads=2")
        con.execute("SET memory_limit='6GB'")
        temp_dir = (work / "duckdb_tmp").as_posix()
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir}'")
        con.execute("PRAGMA max_temp_directory_size='50GiB'")

        con.execute(
            """
            CREATE TABLE pairs (
              source1_entity_id VARCHAR,
              candidate_entity_id VARCHAR,
              source VARCHAR,
              blocking_rule VARCHAR
            )
            """
        )
        for f in self._pair_files:
            esc = Path(f).resolve().as_posix().replace("'", "''")
            logger.info("Union ingest %s", Path(f).name)
            con.execute(
                f"""
                INSERT INTO pairs
                SELECT source1_entity_id, candidate_entity_id, source, blocking_rule
                FROM read_parquet('{esc}')
                """
            )

        out_esc = out_path.resolve().as_posix().replace("'", "''")
        logger.info("Aggregating distinct candidate pairs…")
        con.execute(
            f"""
            COPY (
              SELECT source1_entity_id,
                     candidate_entity_id,
                     source,
                     string_agg(DISTINCT blocking_rule, '|') AS blocking_rules,
                     COUNT(DISTINCT blocking_rule)::INTEGER AS n_rules
              FROM pairs
              GROUP BY 1, 2, 3
            ) TO '{out_esc}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        con.close()
        # free disk
        try:
            db_path.unlink(missing_ok=True)
            Path(str(db_path) + ".wal").unlink(missing_ok=True)
        except OSError:
            pass
        logger.info("Union parquet → %s", out_path)

    def _union_stats(self, union_pq: Path) -> dict:
        con = duckdb.connect(":memory:")
        esc = union_pq.resolve().as_posix().replace("'", "''")
        n_pairs, n_s1 = con.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT source1_entity_id)
            FROM read_parquet('{esc}')
            """
        ).fetchone()
        avg = (n_pairs / n_s1) if n_s1 else 0.0
        return {"n_pairs": int(n_pairs), "n_s1": int(n_s1), "avg": avg}

    def _write_long_from_parquet(
        self, union_pq: Path, path: Path, *, write_dev_columns: bool
    ) -> None:
        con = duckdb.connect(":memory:")
        esc = union_pq.resolve().as_posix().replace("'", "''")
        out_esc = path.resolve().as_posix().replace("'", "''")
        if write_dev_columns:
            con.execute(
                f"""
                COPY (
                  SELECT source1_entity_id,
                         candidate_entity_id,
                         source,
                         blocking_rules,
                         ROW_NUMBER() OVER (
                           PARTITION BY source1_entity_id
                           ORDER BY n_rules DESC, source, candidate_entity_id
                         ) AS candidate_rank
                  FROM read_parquet('{esc}')
                ) TO '{out_esc}' (HEADER, DELIMITER '\t')
                """
            )
        else:
            con.execute(
                f"""
                COPY (
                  SELECT source1_entity_id, candidate_entity_id, source
                  FROM read_parquet('{esc}')
                ) TO '{out_esc}' (HEADER, DELIMITER '\t')
                """
            )
        logger.info("Wrote %s", path)

    def _write_submission(
        self, path: Path, *, s1_parquet: str, union_pq: Path
    ) -> None:
        con = duckdb.connect(":memory:")
        s1_esc = Path(s1_parquet).resolve().as_posix().replace("'", "''")
        u_esc = union_pq.resolve().as_posix().replace("'", "''")
        out_esc = path.resolve().as_posix().replace("'", "''")
        con.execute(
            f"""
            COPY (
              SELECT s.entity_id AS source1_entity_id,
                     COALESCE(
                       string_agg(c.candidate_entity_id, ','
                                  ORDER BY c.candidate_entity_id),
                       ''
                     ) AS candidate_entity_ids
              FROM (SELECT entity_id FROM read_parquet('{s1_esc}')) s
              LEFT JOIN read_parquet('{u_esc}') c
                ON s.entity_id = c.source1_entity_id
              GROUP BY s.entity_id
              ORDER BY s.entity_id
            ) TO '{out_esc}' (HEADER, DELIMITER '\t')
            """
        )
        logger.info("Wrote submission-format %s", path)

    def iter_block_pairs(self) -> Iterable[tuple[str, list[tuple[str, str, str]], float]]:
        """Yield (block_name, pairs, runtime) reloading from spilled parquet for eval."""
        by_block: dict[str, list[B.BlockResult]] = defaultdict(list)
        for br in self.block_results:
            by_block[br.block_name].append(br)
        for name, results in by_block.items():
            pairs: list[tuple[str, str, str]] = []
            runtime = 0.0
            for br in results:
                runtime += br.runtime_seconds
                if br.pairs:
                    pairs.extend(br.pairs)
                elif br.pairs_path:
                    con = duckdb.connect(":memory:")
                    esc = br.pairs_path.replace("'", "''")
                    rows = con.execute(
                        f"SELECT source1_entity_id, candidate_entity_id, source "
                        f"FROM read_parquet('{esc}')"
                    ).fetchall()
                    pairs.extend((a, b, c) for a, b, c in rows)
            yield name, pairs, runtime


def generate_candidates(
    s1_path: str | Path,
    s2_path: str | Path,
    s3_path: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """Clean API for Sudhanshu / pipeline integration."""
    cfg = load_config(config_path) if config_path else BlockingConfig()
    gen = CandidateGenerator(cfg)
    return gen.generate(s1_path, s2_path, s3_path, output_path, **kwargs)


def union_pairs(
    results: Iterable[B.BlockResult],
) -> dict[tuple[str, str, str], set[str]]:
    out: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for res in results:
        for s1, cand, source in res.pairs:
            out[(s1, cand, source)].add(res.block_name)
    return out
