"""Blocking evaluation against training ground truth."""
from __future__ import annotations

import csv
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import psutil

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    block_name: str
    source: str
    total_candidate_pairs: int
    unique_s1: int
    avg_candidates_per_s1: float
    median_candidates_per_s1: float
    p95_candidates_per_s1: float
    max_candidates_per_s1: int
    true_pairs: int
    recovered_true_pairs: int
    pair_candidate_recall: float
    s1_full_recall: float
    zero_match_s1_with_candidates: int
    zero_match_s1_total: int
    zero_match_candidate_rate: float
    runtime_seconds: float
    memory_usage_mb: float


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    gt: dict[str, set[str]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            s1, tab, rest = line.partition("\t")
            ids = set()
            rest = rest.rstrip("\n")
            if rest.strip():
                ids = {x for x in rest.split(",") if x}
            gt[s1] = ids
    return gt


def load_long_candidates(
    path: str | Path,
) -> dict[str, set[str]]:
    """Load long-format candidate_pairs.tsv → {s1: set(cand_ids)}."""
    out: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_entity_id"]
            cand = row["candidate_entity_id"]
            out[s1].add(cand)
    return out


def pairs_to_map(
    pairs: Iterable[tuple[str, str, str]],
    *,
    source_filter: str | None = None,
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for s1, cand, source in pairs:
        if source_filter and source != source_filter:
            continue
        out[s1].add(cand)
    return out


def _percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def evaluate_candidates(
    candidates: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    *,
    block_name: str,
    source: str = "ALL",
    runtime_seconds: float = 0.0,
    all_s1_ids: set[str] | None = None,
) -> EvalMetrics:
    if all_s1_ids is None:
        all_s1_ids = set(ground_truth.keys())

    true_pairs = 0
    recovered = 0
    full_recall_num = 0
    full_recall_den = 0
    zero_with_cands = 0
    zero_total = 0

    per_s1_counts: list[int] = []
    total_pairs = 0
    unique_s1_with = 0

    for s1 in all_s1_ids:
        cands = candidates.get(s1, set())
        n = len(cands)
        total_pairs += n
        if n:
            unique_s1_with += 1
            per_s1_counts.append(n)
        else:
            per_s1_counts.append(0)

        truth = ground_truth.get(s1, set())
        # optional source filter on truth
        if source in ("S2", "S3"):
            truth = {t for t in truth if t.startswith(source + "-")}

        true_pairs += len(truth)
        hit = truth & cands
        recovered += len(hit)
        if truth:
            full_recall_den += 1
            if truth <= cands:
                full_recall_num += 1
        else:
            zero_total += 1
            if cands:
                zero_with_cands += 1

    per_s1_counts.sort()
    avg = (total_pairs / len(all_s1_ids)) if all_s1_ids else 0.0
    med = float(statistics.median(per_s1_counts)) if per_s1_counts else 0.0
    p95 = _percentile(per_s1_counts, 0.95)
    mx = max(per_s1_counts) if per_s1_counts else 0
    pair_recall = (recovered / true_pairs) if true_pairs else 1.0
    s1_full = (full_recall_num / full_recall_den) if full_recall_den else 1.0
    zero_rate = (zero_with_cands / zero_total) if zero_total else 0.0

    return EvalMetrics(
        block_name=block_name,
        source=source,
        total_candidate_pairs=total_pairs,
        unique_s1=unique_s1_with,
        avg_candidates_per_s1=avg,
        median_candidates_per_s1=med,
        p95_candidates_per_s1=p95,
        max_candidates_per_s1=mx,
        true_pairs=true_pairs,
        recovered_true_pairs=recovered,
        pair_candidate_recall=pair_recall,
        s1_full_recall=s1_full,
        zero_match_s1_with_candidates=zero_with_cands,
        zero_match_s1_total=zero_total,
        zero_match_candidate_rate=zero_rate,
        runtime_seconds=runtime_seconds,
        memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024),
    )


def write_evaluation_csv(metrics: list[EvalMetrics], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return
    fieldnames = list(asdict(metrics[0]).keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in metrics:
            w.writerow(asdict(m))
    logger.info("Wrote evaluation → %s (%s rows)", path, len(metrics))


def evaluate_block_results(
    block_results: list,
    ground_truth: dict[str, set[str]],
    *,
    all_s1_ids: set[str] | None = None,
    pair_loader=None,
) -> list[EvalMetrics]:
    """Evaluate each block independently (per source) and hybrid union.

    If ``pair_loader`` is provided (callable yielding block_name, pairs, runtime),
    use it instead of in-memory ``BlockResult.pairs`` (disk-spilled mode).
    """
    metrics: list[EvalMetrics] = []
    by_key: dict[tuple[str, str], list] = defaultdict(list)
    runtimes: dict[tuple[str, str], float] = defaultdict(float)

    if pair_loader is not None:
        all_pairs = []
        total_runtime = 0.0
        for name, pairs, runtime in pair_loader():
            # pairs may mix sources
            for s1, cand, source in pairs:
                by_key[(name, source)].append((s1, cand, source))
            # split runtime roughly by source presence
            sources_present = {p[2] for p in pairs} or {"ALL"}
            share = runtime / max(len(sources_present), 1)
            for src in sources_present:
                runtimes[(name, src)] += share
            all_pairs.extend(pairs)
            total_runtime += runtime
        for (name, source), pairs in sorted(by_key.items()):
            cmap = pairs_to_map(pairs)
            metrics.append(
                evaluate_candidates(
                    cmap,
                    ground_truth,
                    block_name=name,
                    source=source,
                    runtime_seconds=runtimes[(name, source)],
                    all_s1_ids=all_s1_ids,
                )
            )
        cmap = pairs_to_map(all_pairs)
        metrics.append(
            evaluate_candidates(
                cmap,
                ground_truth,
                block_name="hybrid",
                source="ALL",
                runtime_seconds=total_runtime,
                all_s1_ids=all_s1_ids,
            )
        )
        return metrics

    for br in block_results:
        by_key[(br.block_name, br.source)].extend(br.pairs)

    for (name, source), pairs in sorted(by_key.items()):
        cmap = pairs_to_map(pairs)
        runtime = sum(
            br.runtime_seconds
            for br in block_results
            if br.block_name == name and br.source == source
        )
        metrics.append(
            evaluate_candidates(
                cmap,
                ground_truth,
                block_name=name,
                source=source,
                runtime_seconds=runtime,
                all_s1_ids=all_s1_ids,
            )
        )

    all_pairs = []
    runtime = 0.0
    for br in block_results:
        all_pairs.extend(br.pairs)
        runtime += br.runtime_seconds
    cmap = pairs_to_map(all_pairs)
    metrics.append(
        evaluate_candidates(
            cmap,
            ground_truth,
            block_name="hybrid",
            source="ALL",
            runtime_seconds=runtime,
            all_s1_ids=all_s1_ids,
        )
    )
    return metrics


def evaluate_cumulative_stages(
    block_results: list,
    ground_truth: dict[str, set[str]],
    stage_order: list[str],
    *,
    all_s1_ids: set[str] | None = None,
    pair_loader=None,
) -> list[EvalMetrics]:
    # Materialize pairs per block name
    by_block: dict[str, list] = defaultdict(list)
    runtime_by_block: dict[str, float] = defaultdict(float)
    if pair_loader is not None:
        for name, pairs, runtime in pair_loader():
            by_block[name].extend(pairs)
            runtime_by_block[name] += runtime
    else:
        for br in block_results:
            by_block[br.block_name].extend(br.pairs)
            runtime_by_block[br.block_name] += br.runtime_seconds

    accumulated: list = []
    metrics: list[EvalMetrics] = []
    runtime = 0.0
    for i, stage in enumerate(stage_order, start=1):
        accumulated.extend(by_block.get(stage, []))
        runtime += runtime_by_block.get(stage, 0.0)
        cmap = pairs_to_map(accumulated)
        m = evaluate_candidates(
            cmap,
            ground_truth,
            block_name=f"stage_{i:02d}_{stage}",
            source="ALL",
            runtime_seconds=runtime,
            all_s1_ids=all_s1_ids,
        )
        metrics.append(m)
        logger.info(
            "[STAGE %02d +%s] pairs=%s recall=%.4f s1_full=%.4f avg=%.2f p95=%.1f",
            i,
            stage,
            f"{m.total_candidate_pairs:,}",
            m.pair_candidate_recall,
            m.s1_full_recall,
            m.avg_candidates_per_s1,
            m.p95_candidates_per_s1,
        )
    return metrics
