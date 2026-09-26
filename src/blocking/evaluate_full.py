"""DuckDB-backed evaluation for full-scale candidate sets."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import duckdb
import psutil

from .evaluate_blocking import EvalMetrics, write_evaluation_csv

logger = logging.getLogger(__name__)


def _esc(path: Path | str) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")


def evaluate_union_parquet(
    union_pq: Path | str,
    ground_truth_tsv: Path | str,
    *,
    block_name: str = "hybrid",
    runtime_seconds: float = 0.0,
) -> EvalMetrics:
    """Compute recall metrics by joining candidates to exploded ground truth in DuckDB."""
    con = duckdb.connect(":memory:")
    gt = _esc(ground_truth_tsv)
    u = _esc(union_pq)

    con.execute(
        f"""
        CREATE TABLE gt AS
        SELECT source1_entity_id AS s1,
               NULLIF(trim(mid), '') AS mid
        FROM read_csv('{gt}', delim='\t', header=true, quote='',
                      columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
        LEFT JOIN UNNEST(string_split(COALESCE(matched_entity_ids, ''), ',')) AS u(mid) ON TRUE
        """
    )
    con.execute(
        f"""
        CREATE TABLE cands AS
        SELECT source1_entity_id AS s1, candidate_entity_id AS mid
        FROM read_parquet('{u}')
        """
    )

    true_pairs = con.execute(
        "SELECT COUNT(*) FROM gt WHERE mid IS NOT NULL"
    ).fetchone()[0]
    recovered = con.execute(
        """
        SELECT COUNT(*) FROM gt g
        JOIN cands c ON g.s1 = c.s1 AND g.mid = c.mid
        WHERE g.mid IS NOT NULL
        """
    ).fetchone()[0]

    s1_full = con.execute(
        """
        WITH need AS (
          SELECT s1, COUNT(*) AS n_true
          FROM gt WHERE mid IS NOT NULL GROUP BY 1
        ),
        hit AS (
          SELECT g.s1, COUNT(*) AS n_hit
          FROM gt g JOIN cands c ON g.s1=c.s1 AND g.mid=c.mid
          WHERE g.mid IS NOT NULL
          GROUP BY 1
        )
        SELECT
          SUM(CASE WHEN h.n_hit = n.n_true THEN 1 ELSE 0 END)::DOUBLE / COUNT(*),
          COUNT(*)
        FROM need n LEFT JOIN hit h ON n.s1 = h.s1
        """
    ).fetchone()

    stats = con.execute(
        """
        WITH per AS (
          SELECT s1, COUNT(*) AS n FROM cands GROUP BY 1
        ),
        all_s1 AS (
          SELECT DISTINCT s1 FROM gt
        ),
        joined AS (
          SELECT a.s1, COALESCE(p.n, 0) AS n
          FROM all_s1 a LEFT JOIN per p ON a.s1 = p.s1
        )
        SELECT
          SUM(n),
          COUNT(*),
          AVG(n),
          approx_quantile(n, 0.5),
          approx_quantile(n, 0.95),
          MAX(n),
          SUM(CASE WHEN n > 0 THEN 1 ELSE 0 END)
        FROM joined
        """
    ).fetchone()

    zero = con.execute(
        """
        WITH zero_s1 AS (
          SELECT s1 FROM gt GROUP BY 1 HAVING COUNT(mid) FILTER (WHERE mid IS NOT NULL) = 0
        ),
        with_c AS (
          SELECT z.s1 FROM zero_s1 z
          JOIN cands c ON z.s1 = c.s1
          GROUP BY 1
        )
        SELECT
          (SELECT COUNT(*) FROM with_c),
          (SELECT COUNT(*) FROM zero_s1)
        """
    ).fetchone()

    total_pairs, n_s1, avg, med, p95, mx, unique_with = stats
    zero_with, zero_total = zero
    pair_recall = (recovered / true_pairs) if true_pairs else 1.0
    s1_full_recall = float(s1_full[0] or 0.0)
    zero_rate = (zero_with / zero_total) if zero_total else 0.0

    m = EvalMetrics(
        block_name=block_name,
        source="ALL",
        total_candidate_pairs=int(total_pairs or 0),
        unique_s1=int(unique_with or 0),
        avg_candidates_per_s1=float(avg or 0),
        median_candidates_per_s1=float(med or 0),
        p95_candidates_per_s1=float(p95 or 0),
        max_candidates_per_s1=int(mx or 0),
        true_pairs=int(true_pairs),
        recovered_true_pairs=int(recovered),
        pair_candidate_recall=float(pair_recall),
        s1_full_recall=s1_full_recall,
        zero_match_s1_with_candidates=int(zero_with or 0),
        zero_match_s1_total=int(zero_total or 0),
        zero_match_candidate_rate=float(zero_rate),
        runtime_seconds=runtime_seconds,
        memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024),
    )
    logger.info(
        "[%s] pairs=%s recall=%.4f s1_full=%.4f avg=%.2f p95=%.1f max=%s zero_rate=%.4f",
        block_name,
        f"{m.total_candidate_pairs:,}",
        m.pair_candidate_recall,
        m.s1_full_recall,
        m.avg_candidates_per_s1,
        m.p95_candidates_per_s1,
        m.max_candidates_per_s1,
        m.zero_match_candidate_rate,
    )
    return m


def evaluate_block_dir(
    block_pairs_dir: Path | str,
    ground_truth_tsv: Path | str,
    stage_order: list[str],
    out_csv: Path | str,
) -> list[EvalMetrics]:
    """Evaluate each spilled block file and cumulative stages via temp unions."""
    block_pairs_dir = Path(block_pairs_dir)
    metrics: list[EvalMetrics] = []
    # independent per file
    for path in sorted(block_pairs_dir.glob("*.parquet")):
        name_source = path.stem  # block__S2
        if "__" not in name_source:
            continue
        block, source = name_source.rsplit("__", 1)
        # write a one-column-compatible view as union schema
        con = duckdb.connect(":memory:")
        esc = _esc(path)
        tmp = path.with_suffix(".eval.tmp.parquet")
        tmp_esc = _esc(tmp)
        con.execute(
            f"""
            COPY (
              SELECT source1_entity_id, candidate_entity_id, source,
                     blocking_rule AS blocking_rules, 1 AS n_rules
              FROM read_parquet('{esc}')
            ) TO '{tmp_esc}' (FORMAT PARQUET)
            """
        )
        m = evaluate_union_parquet(tmp, ground_truth_tsv, block_name=block)
        m.source = source
        metrics.append(m)
        tmp.unlink(missing_ok=True)

    # cumulative
    accumulated: list[Path] = []
    for i, stage in enumerate(stage_order, start=1):
        stage_files = list(block_pairs_dir.glob(f"{stage}__*.parquet"))
        accumulated.extend(stage_files)
        if not accumulated:
            continue
        con = duckdb.connect(":memory:")
        selects = [
            f"SELECT source1_entity_id, candidate_entity_id, source FROM read_parquet('{_esc(p)}')"
            for p in accumulated
        ]
        tmp = block_pairs_dir / f"_stage_{i:02d}.parquet"
        con.execute(
            f"""
            COPY (
              SELECT source1_entity_id, candidate_entity_id, ANY_VALUE(source) AS source,
                     '{stage}' AS blocking_rules, 1 AS n_rules
              FROM ({' UNION ALL '.join(selects)})
              GROUP BY 1, 2
            ) TO '{_esc(tmp)}' (FORMAT PARQUET)
            """
        )
        m = evaluate_union_parquet(
            tmp, ground_truth_tsv, block_name=f"stage_{i:02d}_{stage}"
        )
        metrics.append(m)
        tmp.unlink(missing_ok=True)

    write_evaluation_csv(metrics, out_csv)
    return metrics
