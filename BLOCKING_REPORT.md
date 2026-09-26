# Blocking Report (Ashutosh)

All metrics below are **measured**. Sample ≠ full train (sample includes all true matches + few distractors, so rare-token DF/caps behave differently).

## Deliverables

| Artefact | Location |
|----------|----------|
| Long candidate pairs (Sudhanshu) | `D:\amazon_ml_blocking\outputs\candidate_pairs.tsv` |
| Official submission format | `D:\amazon_ml_blocking\outputs\candidate_pairs_submission.tsv` |
| Evaluation CSV | `outputs/blocking_evaluation.csv` |
| Sample misses / eval | `outputs/*_sample20k.*` |
| Path pointer | `outputs/CANDIDATE_PAIRS_LOCATION.txt` |

> Large TSVs live on **D:** because C: ran out of disk during union.

## Implemented blocks

| Block | Status |
|-------|--------|
| exact_name | kept |
| exact_address | kept |
| exact_name_address | kept |
| postal | kept (never postal-alone) |
| house_number_name | kept |
| rare_name_token | kept |
| rare_address_token | kept |
| name_prefix | kept (lengths 6/8) |
| address_token | kept |
| char_ngram | residual-only; tiny lift |
| rapidfuzz | residual-only; **~0 lift** on full train |
| FAISS | **not used** |

## Full-train cumulative stages (measured)

True pairs in GT: **7,638,365**

| Stage | +block | pair recall | S1 full | avg/S1 | p95 | pairs |
|------:|--------|------------:|--------:|-------:|----:|------:|
| 1 | exact_name | 0.4475 | 0.1159 | 19.93 | 104 | 43,978,537 |
| 2 | +exact_address | 0.5268 | 0.1683 | 20.35 | 104 | 44,898,253 |
| 3 | +exact_name_address | 0.5268 | 0.1683 | 20.35 | 104 | 44,898,253 |
| 4 | +postal | 0.5461 | 0.1963 | 22.94 | 111 | 50,624,811 |
| 5 | +house_number_name | 0.6928 | 0.3700 | 27.90 | 127 | 61,578,878 |
| 6 | +rare_name_token | 0.7152 | 0.4124 | 30.52 | 127 | 67,352,604 |
| 7 | +rare_address_token | 0.7801 | 0.5172 | 34.91 | 133 | 77,041,850 |
| 8 | +name_prefix | 0.8157 | 0.5808 | 43.27 | 136 | 95,495,040 |
| 9 | +address_token | 0.8377 | 0.6235 | 45.49 | 142 | 100,390,157 |
| 10 | +char_ngram | 0.8379 | 0.6236 | 45.51 | 142 | 100,432,410 |
| 11 | +rapidfuzz | 0.8379 | 0.6236 | 45.51 | 142 | 100,435,430 |

### Full-train hybrid (v1 config, name_freq cap=150)

| Metric | Value |
|--------|------:|
| Pair candidate recall | **0.8379** |
| S1 full recall | **0.6236** |
| Avg candidates / S1 | **45.51** |
| Median | **31** |
| P95 | **141–142** |
| Max | **514** |
| Zero-match S1 w/ candidates (rate) | **0.9759** |
| Total candidate pairs | **100,435,430** |
| Unique S1 with ≥1 cand | **2,200,463** / 2,206,821 |

**Root cause of recall gap vs sample:** frequency caps used `HAVING COUNT(*) <= cap` which **drops entire keys** when over-cap. On full data, `exact_name` max candidates = 150 (=cap), so popular normalized names were excluded entirely. Sample was not representative.

**Mitigation in progress:** `configs/blocking.yaml` raised to `max_name_frequency: 2000` (and related caps); `scripts/recall_boost.py` re-runs exact_name / exact_address / house_number_name / name_prefix and re-unions. See log `outputs/logs/recall_boost.err` for **hybrid_v2** metrics when finished.

## Sample (20k) reference — measured earlier

| Metric | Value |
|--------|------:|
| Pair recall | 0.9477 |
| S1 full | 0.8571 |
| Avg / p95 / max | 37.12 / 84 / 169 |
| Peak RSS (spill mode) | ~206 MB |

Untuned sample (prefix4 @ 500): recall 0.9724 but avg 227 — rejected.

## Approximate methods

| Method | Decision | Evidence |
|--------|----------|----------|
| char_ngram | keep residual | +0.0002 full-train recall |
| RapidFuzz | disable by default | +0.0000 full-train recall |
| FAISS | not used | structured+token sufficient; disk/memory constrained |

## Top failure patterns (sample misses)

| Category | Count |
|----------|------:|
| HOUSE_NUMBER_MISSING | 1879 |
| TRANSLITERATION | 715 |
| ABBREVIATION | 601 |
| POSTAL_MISSING | 205 |
| NAME_VARIATION | 136 |
| TYPO | 80 |

## Recommended config

`configs/blocking.yaml` (v2 — raised caps; RapidFuzz off; work_dir on D:).

## Command for Sudhanshu

```bash
python -m src.blocking.run \
  --config configs/blocking.yaml \
  --s1 dataset/train/train_source1.tsv \
  --s2 dataset/train/train_source2.tsv \
  --s3 dataset/train/train_source3.tsv \
  --output D:/amazon_ml_blocking/outputs/candidate_pairs.tsv
```

Python:

```python
from src.blocking import generate_candidates
generate_candidates(s1, s2, s3, r"D:\amazon_ml_blocking\outputs\candidate_pairs.tsv", "configs/blocking.yaml")
```

Handoff columns: `source1_entity_id`, `candidate_entity_id`, `source` (+ optional `blocking_rules`, `candidate_rank`).
