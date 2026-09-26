# Blocking / Candidate Generation (Ashutosh)

**Owner:** Ashutosh — retrieval, blocking, candidate generation only.  
**Consumer:** Sudhanshu — downstream matching model.

This module maximizes **candidate recall** while keeping candidate volume, runtime, and memory manageable. It does **not** decide final matches.

---

## HOW SUDHANSHU USES THIS MODULE

### 1. Input format

Tab-separated source files (same as the challenge):

| column | meaning |
|--------|---------|
| `entity_id` | `S1-…` / `S2-…` / `S3-…` |
| `business_name` | raw name |
| `business_address` | raw address |
| `country` | open-set string (do not hard-code) |

### 2. Command

From the repo root (`amazon ML/`):

```bash
python -m src.blocking.run \
  --config configs/blocking.yaml \
  --s1 dataset/train/train_source1.tsv \
  --s2 dataset/train/train_source2.tsv \
  --s3 dataset/train/train_source3.tsv \
  --ground-truth dataset/train/train_ground_truth.tsv \
  --output outputs/candidate_pairs.tsv \
  --evaluate
```

Test (no GT):

```bash
python -m src.blocking.run \
  --config configs/blocking.yaml \
  --s1 dataset/test/test_source1.tsv \
  --s2 dataset/test/test_source2.tsv \
  --s3 dataset/test/test_source3.tsv \
  --output outputs/candidate_pairs_test.tsv
```

### 3. Python API

```python
from src.blocking import generate_candidates, CandidateGenerator, load_config

generate_candidates(
    s1_path="dataset/train/train_source1.tsv",
    s2_path="dataset/train/train_source2.tsv",
    s3_path="dataset/train/train_source3.tsv",
    output_path="outputs/candidate_pairs.tsv",
    config_path="configs/blocking.yaml",
)

# or
cfg = load_config("configs/blocking.yaml")
CandidateGenerator(cfg).generate(s1, s2, s3, "outputs/candidate_pairs.tsv")
```

### 4. Output format (handoff to matching)

`outputs/candidate_pairs.tsv` (long / pair format):

| column | description |
|--------|-------------|
| `source1_entity_id` | S1 id |
| `candidate_entity_id` | S2 or S3 id |
| `source` | `S2` or `S3` |
| `blocking_rules` | pipe-joined rules that retrieved the pair |
| `candidate_rank` | rank by # agreeing rules (desc) |

Also written: `outputs/candidate_pairs_submission.tsv` — official challenge format  
(`source1_entity_id`, `candidate_entity_ids` comma list, one row per S1).

### 5–8. Metrics / runtime / memory

See **`BLOCKING_REPORT.md`** (filled from actual training runs). Do not invent numbers.

### 9. Configuration

All caps/thresholds live in `configs/blocking.yaml` (`max_name_frequency`, rare-token bounds, n-gram top-k, RapidFuzz cutoff, enabled block list, …).

### 10. Regenerate candidates

Re-run the command above. Cached normalized Parquet under `outputs/work/` is reused if newer than the TSV; pass `--force-normalize` to rebuild.

---

## Blocks implemented

1. `exact_name` — country + name_norm (freq cap)  
2. `exact_address` — country + address_norm (freq cap)  
3. `exact_name_address` — country + name + address  
4. `postal` — country+postal+name_prefix / +house (never postal alone)  
5. `house_number_name` — country + house + name_prefix_{4,6}  
6. `rare_name_token` — rare informative name tokens  
7. `rare_address_token` — rare address tokens (+ house combo)  
8. `name_prefix` — prefix 4/6/8 with caps  
9. `address_token` — house + address signature  
10. `char_ngram` — residual-only approximate n-gram retrieval  
11. `rapidfuzz` — residual/pool WRatio (optional; keep only if recall improves)

Union → dedupe. Country is an open-set partition key (works for France in test).

---

## Evaluation artefacts

| file | purpose |
|------|---------|
| `outputs/blocking_evaluation.csv` | per-block + cumulative stage metrics |
| `outputs/blocking_misses.tsv` | true matches not retrieved + failure category |

## Tests

```bash
pytest tests/ -q
```

## Constraints

No geocoding, Maps, registries, or external entity APIs. Only provided datasets.
