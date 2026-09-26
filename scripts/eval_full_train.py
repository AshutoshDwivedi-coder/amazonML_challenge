"""Full-train DuckDB evaluation against ground truth."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.blocking.evaluate_full import evaluate_block_dir, evaluate_union_parquet, write_evaluation_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eval_full")

GT = Path(r"c:\Users\Lenovo\OneDrive\Desktop\amazon ML\dataset\train\train_ground_truth.tsv")
UNION = Path(r"D:\amazon_ml_blocking\work\candidate_pairs_union.parquet")
BLOCK_DIR = Path(r"c:\Users\Lenovo\OneDrive\Desktop\amazon ML\outputs\work\block_pairs")
OUT_CSV = Path(r"c:\Users\Lenovo\OneDrive\Desktop\amazon ML\outputs\blocking_evaluation.csv")
OUT_D = Path(r"D:\amazon_ml_blocking\outputs\blocking_evaluation.csv")


def main() -> int:
    stages = [
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
    log.info("Evaluating hybrid union…")
    hybrid = evaluate_union_parquet(UNION, GT, block_name="hybrid")
    log.info("Evaluating per-block + cumulative (may take a while)…")
    metrics = evaluate_block_dir(BLOCK_DIR, GT, stages, OUT_D)
    # append hybrid
    from src.blocking.evaluate_blocking import write_evaluation_csv as wcsv

    wcsv(metrics + [hybrid], OUT_D)
    # also copy small CSV to repo outputs
    OUT_CSV.write_text(OUT_D.read_text(encoding="utf-8"), encoding="utf-8")
    log.info(
        "HYBRID recall=%.4f s1_full=%.4f avg=%.2f p95=%.1f max=%s zero_rate=%.4f pairs=%s",
        hybrid.pair_candidate_recall,
        hybrid.s1_full_recall,
        hybrid.avg_candidates_per_s1,
        hybrid.p95_candidates_per_s1,
        hybrid.max_candidates_per_s1,
        hybrid.zero_match_candidate_rate,
        f"{hybrid.total_candidate_pairs:,}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
