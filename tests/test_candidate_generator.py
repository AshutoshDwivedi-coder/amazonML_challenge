"""Candidate generator focused tests."""
from pathlib import Path

from src.blocking.candidate_generator import CandidateGenerator
from src.blocking.config import BlockingConfig


def _tsv(path: Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for r in rows:
            f.write(
                f"{r['entity_id']}\t{r['business_name']}\t{r['business_address']}\t{r['country']}\n"
            )


def test_dedup_and_rules_metadata(tmp_path):
    s1 = tmp_path / "s1.tsv"
    s2 = tmp_path / "s2.tsv"
    s3 = tmp_path / "s3.tsv"
    _tsv(
        s1,
        [
            {
                "entity_id": "S1-1",
                "business_name": "Zenith Labs",
                "business_address": "5 River Rd 75001",
                "country": "US",
            }
        ],
    )
    _tsv(
        s2,
        [
            {
                "entity_id": "S2-1",
                "business_name": "Zenith Labs Inc",
                "business_address": "5 River Road, 75001",
                "country": "US",
            }
        ],
    )
    _tsv(s3, [])
    # empty S3 still needs header
    _tsv(s3, [])

    out = tmp_path / "out.tsv"
    cfg = BlockingConfig(
        work_dir=str(tmp_path / "work"),
        enabled_blocks=["exact_name", "postal", "house_number_name", "name_prefix"],
        rare_token_min_frequency=1,
        rapidfuzz_enabled=False,
    )
    gen = CandidateGenerator(cfg)
    gen.generate(s1, s2, s3, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    # header + unique pairs only
    body = lines[1:]
    keys = {tuple(l.split("\t")[:3]) for l in body}
    assert len(keys) == len(body)
    if body:
        assert "blocking_rules" in lines[0]
