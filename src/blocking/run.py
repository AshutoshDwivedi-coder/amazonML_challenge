"""CLI entrypoint for the blocking / candidate-generation subsystem.

Examples
--------
Train candidates + evaluation::

    python -m src.blocking.run \\
        --config configs/blocking.yaml \\
        --s1 dataset/train/train_source1.tsv \\
        --s2 dataset/train/train_source2.tsv \\
        --s3 dataset/train/train_source3.tsv \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --output outputs/candidate_pairs.tsv \\
        --evaluate

Test candidates (no GT)::

    python -m src.blocking.run \\
        --config configs/blocking.yaml \\
        --s1 dataset/test/test_source1.tsv \\
        --s2 dataset/test/test_source2.tsv \\
        --s3 dataset/test/test_source3.tsv \\
        --output outputs/candidate_pairs_test.tsv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .candidate_generator import CandidateGenerator
from .config import load_config
from .error_analysis import write_blocking_misses
from .evaluate_blocking import (
    evaluate_block_results,
    evaluate_cumulative_stages,
    evaluate_candidates,
    load_ground_truth,
    load_long_candidates,
    pairs_to_map,
    write_evaluation_csv,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ashutosh blocking / candidate generation")
    p.add_argument("--config", default="configs/blocking.yaml")
    p.add_argument("--s1", required=True)
    p.add_argument("--s2", required=True)
    p.add_argument("--s3", required=True)
    p.add_argument("--output", default="outputs/candidate_pairs.tsv")
    p.add_argument("--ground-truth", default=None)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--force-normalize", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--eval-out",
        default="outputs/blocking_evaluation.csv",
        help="Path for per-block + cumulative evaluation CSV",
    )
    p.add_argument(
        "--misses-out",
        default="outputs/blocking_misses.tsv",
        help="Path for missed true-match analysis",
    )
    p.add_argument(
        "--eval-full-scale",
        action="store_true",
        help="Use DuckDB-backed evaluation (required for full train scale)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    log = logging.getLogger("blocking.run")

    cfg = load_config(args.config)
    log.info("Config loaded from %s", args.config)
    log.info("Enabled blocks: %s", cfg.enabled_blocks)

    gen = CandidateGenerator(cfg)
    out = gen.generate(
        args.s1,
        args.s2,
        args.s3,
        args.output,
        force_normalize=args.force_normalize,
    )
    log.info("Candidate pairs → %s", out)

    if args.evaluate or args.ground_truth:
        if not args.ground_truth:
            log.error("--evaluate requires --ground-truth")
            return 2

        stage_order = [
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
        ran = {br.block_name for br in gen.block_results}
        stage_order = [s for s in stage_order if s in ran]

        work = Path(cfg.work_dir)
        union_pq = work / "candidate_pairs_union.parquet"

        if args.eval_full_scale:
            from .evaluate_full import evaluate_block_dir, evaluate_union_parquet

            hybrid = evaluate_union_parquet(
                union_pq, args.ground_truth, block_name="hybrid"
            )
            evaluate_block_dir(
                work / "block_pairs",
                args.ground_truth,
                stage_order,
                args.eval_out,
            )
            log.info(
                "HYBRID FULL-SCALE recall=%.4f s1_full=%.4f avg=%.2f p95=%.1f max=%s zero_rate=%.4f",
                hybrid.pair_candidate_recall,
                hybrid.s1_full_recall,
                hybrid.avg_candidates_per_s1,
                hybrid.p95_candidates_per_s1,
                hybrid.max_candidates_per_s1,
                hybrid.zero_match_candidate_rate,
            )
        else:
            gt = load_ground_truth(args.ground_truth)
            all_s1 = set(gt.keys())
            metrics = evaluate_block_results(
                gen.block_results,
                gt,
                all_s1_ids=all_s1,
                pair_loader=gen.iter_block_pairs,
            )
            cum = evaluate_cumulative_stages(
                gen.block_results,
                gt,
                stage_order,
                all_s1_ids=all_s1,
                pair_loader=gen.iter_block_pairs,
            )
            write_evaluation_csv(metrics + cum, args.eval_out)

            cands = load_long_candidates(out)
            hybrid = evaluate_candidates(
                cands, gt, block_name="hybrid_file", source="ALL", all_s1_ids=all_s1
            )
            log.info(
                "HYBRID FILE recall=%.4f s1_full=%.4f avg=%.2f p95=%.1f max=%s zero_rate=%.4f",
                hybrid.pair_candidate_recall,
                hybrid.s1_full_recall,
                hybrid.avg_candidates_per_s1,
                hybrid.p95_candidates_per_s1,
                hybrid.max_candidates_per_s1,
                hybrid.zero_match_candidate_rate,
            )

            write_blocking_misses(
                gt,
                cands,
                work / "s1_normalized.parquet",
                work / "s2_normalized.parquet",
                work / "s3_normalized.parquet",
                args.misses_out,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
