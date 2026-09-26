"""Build a stratified subsample for fast blocking experiments."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def sample_tsv(src: Path, dst: Path, ids: set[str] | None, n: int | None, seed: int) -> set[str]:
    rng = random.Random(seed)
    kept_ids: set[str] = set()
    rows: list[str] = []
    with open(src, encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            eid = line.split("\t", 1)[0]
            if ids is not None and eid not in ids:
                continue
            rows.append(line)
    if n is not None and len(rows) > n:
        rows = rng.sample(rows, n)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(header if header.endswith("\n") else header + "\n")
        for line in rows:
            f.write(line if line.endswith("\n") else line + "\n")
            kept_ids.add(line.split("\t", 1)[0])
    return kept_ids


def sample_gt(src: Path, dst: Path, s1_ids: set[str]) -> set[str]:
    """Write GT for sampled S1; return all referenced S2/S3 ids."""
    cand_ids: set[str] = set()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8", errors="replace") as fin, open(
        dst, "w", encoding="utf-8", newline=""
    ) as fout:
        header = fin.readline()
        fout.write(header if header.endswith("\n") else header + "\n")
        for line in fin:
            s1, _, rest = line.partition("\t")
            if s1 not in s1_ids:
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            rest = rest.strip()
            if rest:
                for cid in rest.split(","):
                    if cid:
                        cand_ids.add(cid)
    return cand_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-s1", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs/sample")
    args = p.parse_args()
    root = Path(".")
    out = Path(args.out_dir)
    s1_ids = sample_tsv(
        root / "dataset/train/train_source1.tsv",
        out / "s1.tsv",
        None,
        args.n_s1,
        args.seed,
    )
    cand_ids = sample_gt(
        root / "dataset/train/train_ground_truth.tsv",
        out / "ground_truth.tsv",
        s1_ids,
    )
    # Include true matches + a random pool of distractors from S2/S3 for realism
    sample_tsv(
        root / "dataset/train/train_source2.tsv",
        out / "s2.tsv",
        {c for c in cand_ids if c.startswith("S2-")},
        None,
        args.seed,
    )
    # Also add random S2 distractors
    extra_s2 = sample_tsv(
        root / "dataset/train/train_source2.tsv",
        out / "s2_extra.tsv",
        None,
        min(100_000, args.n_s1 * 5),
        args.seed + 1,
    )
    # merge s2 + extra
    _merge(out / "s2.tsv", out / "s2_extra.tsv", out / "s2_merged.tsv")
    (out / "s2_merged.tsv").replace(out / "s2.tsv")
    (out / "s2_extra.tsv").unlink(missing_ok=True)

    sample_tsv(
        root / "dataset/train/train_source3.tsv",
        out / "s3.tsv",
        {c for c in cand_ids if c.startswith("S3-")},
        None,
        args.seed,
    )
    sample_tsv(
        root / "dataset/train/train_source3.tsv",
        out / "s3_extra.tsv",
        None,
        min(100_000, args.n_s1 * 5),
        args.seed + 2,
    )
    _merge(out / "s3.tsv", out / "s3_extra.tsv", out / "s3_merged.tsv")
    (out / "s3_merged.tsv").replace(out / "s3.tsv")
    (out / "s3_extra.tsv").unlink(missing_ok=True)

    print(f"Sample S1={len(s1_ids)} true_cands={len(cand_ids)} → {out}")


def _merge(a: Path, b: Path, dst: Path) -> None:
    seen: set[str] = set()
    with open(dst, "w", encoding="utf-8", newline="") as fout:
        with open(a, encoding="utf-8") as fa:
            header = fa.readline()
            fout.write(header)
            for line in fa:
                eid = line.split("\t", 1)[0]
                if eid not in seen:
                    seen.add(eid)
                    fout.write(line)
        with open(b, encoding="utf-8") as fb:
            fb.readline()
            for line in fb:
                eid = line.split("\t", 1)[0]
                if eid not in seen:
                    seen.add(eid)
                    fout.write(line)


if __name__ == "__main__":
    main()
