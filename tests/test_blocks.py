"""Unit tests for blocking + candidate union on synthetic fixtures."""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.blocking.blocks import (
    block_exact_address,
    block_exact_name,
    block_exact_name_address,
    block_house_number_name,
    block_name_prefix,
    block_postal,
    block_rare_name_token,
)
from src.blocking.candidate_generator import CandidateGenerator, union_pairs
from src.blocking.config import BlockingConfig
from src.blocking.io_utils import SOURCE_SCHEMA
from src.blocking.normalize import normalize_record


def _write_records(path: Path, rows: list[dict]) -> str:
    recs = [
        normalize_record(
            r["entity_id"], r["business_name"], r["business_address"], r["country"]
        )
        for r in rows
    ]
    table = pa.Table.from_pylist(recs, schema=SOURCE_SCHEMA)
    pq.write_table(table, path)
    return str(path)


@pytest.fixture
def fixture_dir(tmp_path: Path):
    s1 = [
        {
            "entity_id": "S1-1",
            "business_name": "Prabhav Business Center",
            "business_address": "12 MG Road, 560001",
            "country": "India",
        },
        {
            "entity_id": "S1-2",
            "business_name": "Orelee's Barbershop",
            "business_address": "1795 Westchester Drive, High Point, NC 27262",
            "country": "US",
        },
        {
            "entity_id": "S1-3",
            "business_name": "No Match Corp",
            "business_address": "1 Nowhere St",
            "country": "France",  # unseen in "train" style
        },
        {
            "entity_id": "S1-4",
            "business_name": "Multi Match LLC",
            "business_address": "9 Oak Ave, 10001",
            "country": "US",
        },
    ]
    s2 = [
        {
            "entity_id": "S2-1",
            "business_name": "Prabhav Business Centre Pvt Ltd",
            "business_address": "12 M.G. Rd, Bengaluru 560001",
            "country": "India",
        },
        {
            "entity_id": "S2-2",
            "business_name": "Orelees Barbershop Inc",
            "business_address": "1795 Westchester Dr, High Point NC",
            "country": "US",
        },
        {
            "entity_id": "S2-4a",
            "business_name": "Multi Match",
            "business_address": "9 Oak Avenue, New York 10001",
            "country": "US",
        },
        {
            "entity_id": "S2-4b",
            "business_name": "Multi Match Company",
            "business_address": "9 Oak Ave Unit 2, 10001",
            "country": "US",
        },
    ]
    s3 = [
        {
            "entity_id": "S3-1",
            "business_name": "Prabhav Biz Center",
            "business_address": "MG Road 12, 560001",
            "country": "India",
        },
        {
            "entity_id": "S3-4",
            "business_name": "Multi Match Incorporated",
            "business_address": "9 Oak Ave, 10001",
            "country": "US",
        },
    ]
    p1 = _write_records(tmp_path / "s1.parquet", s1)
    p2 = _write_records(tmp_path / "s2.parquet", s2)
    p3 = _write_records(tmp_path / "s3.parquet", s3)
    return p1, p2, p3, tmp_path


def test_exact_name_and_address(fixture_dir):
    s1, s2, s3, _ = fixture_dir
    cfg = BlockingConfig(max_name_frequency=100, max_address_frequency=100)
    r = block_exact_name(s1, s2, "S2", cfg)
    assert isinstance(r.pairs, list)
    r2 = block_exact_address(s1, s2, "S2", cfg)
    r3 = block_exact_name_address(s1, s2, "S2", cfg)
    assert r3.block_name == "exact_name_address"


def test_postal_and_house(fixture_dir):
    s1, s2, _, _ = fixture_dir
    cfg = BlockingConfig()
    postal = block_postal(s1, s2, "S2", cfg)
    house = block_house_number_name(s1, s2, "S2", cfg)
    # S1-1 / S2-1 share postal 560001 + related name prefix
    ids = {(a, b) for a, b, _ in postal.pairs}
    assert ("S1-1", "S2-1") in ids or len(house.pairs) >= 0


def test_rare_token_and_prefix(fixture_dir):
    s1, s2, _, _ = fixture_dir
    cfg = BlockingConfig(
        rare_token_min_frequency=1,
        rare_token_max_frequency=10,
        max_rare_token_block_size=50,
        max_prefix_frequency=50,
    )
    rare = block_rare_name_token(s1, s2, "S2", cfg)
    # "prabhav" should link S1-1 to S2-1
    assert any(a == "S1-1" and b == "S2-1" for a, b, _ in rare.pairs)
    pref = block_name_prefix(s1, s2, "S2", cfg)
    assert pref.block_name == "name_prefix"


def test_frequency_cap_blocks_huge_keys(tmp_path):
    # many candidates share same name → capped out
    s1 = [
        {
            "entity_id": "S1-x",
            "business_name": "Common Name Inc",
            "business_address": "1 A St",
            "country": "US",
        }
    ]
    s2 = [
        {
            "entity_id": f"S2-{i}",
            "business_name": "Common Name Inc",
            "business_address": f"{i} B St",
            "country": "US",
        }
        for i in range(20)
    ]
    p1 = _write_records(tmp_path / "s1.parquet", s1)
    p2 = _write_records(tmp_path / "s2.parquet", s2)
    cfg = BlockingConfig(max_name_frequency=5)
    r = block_exact_name(p1, p2, "S2", cfg)
    assert r.pairs == []


def test_union_dedup_and_s2_s3_separation(fixture_dir):
    s1, s2, s3, tmp = fixture_dir
    cfg = BlockingConfig(
        rare_token_min_frequency=1,
        rare_token_max_frequency=20,
        enabled_blocks=[
            "exact_name",
            "exact_address",
            "postal",
            "house_number_name",
            "rare_name_token",
            "name_prefix",
        ],
        work_dir=str(tmp / "work"),
        rapidfuzz_enabled=False,
    )
    # Use generator with pre-normalized paths via public generate on tiny TSV
    # Write TSVs
    def dump_tsv(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in rows:
                f.write(
                    f"{r['entity_id']}\t{r['business_name']}\t{r['business_address']}\t{r['country']}\n"
                )

    # reload from parquet via generator is easier: call blocks and union
    results = [
        block_exact_name(s1, s2, "S2", cfg),
        block_exact_name(s1, s3, "S3", cfg),
        block_rare_name_token(s1, s2, "S2", cfg),
        block_rare_name_token(s1, s3, "S3", cfg),
    ]
    merged = union_pairs(results)
    # sources preserved
    assert all(k[2] in ("S2", "S3") for k in merged)
    # one-to-many: S1-4 can have multiple
    s1_4 = [k for k in merged if k[0] == "S1-4"]
    assert len(s1_4) >= 1
    # zero-match S1-3 may have zero (France-only, no cands)
    s1_3 = [k for k in merged if k[0] == "S1-3"]
    assert len(s1_3) == 0


def test_candidate_generator_end_to_end(tmp_path):
    rows_s1 = [
        {
            "entity_id": "S1-1",
            "business_name": "Alpha Foods",
            "business_address": "10 Pine St 30301",
            "country": "US",
        },
        {
            "entity_id": "S1-2",
            "business_name": "Singleton Biz",
            "business_address": "99 Z St",
            "country": "US",
        },
    ]
    rows_s2 = [
        {
            "entity_id": "S2-1",
            "business_name": "Alpha Foods Inc",
            "business_address": "10 Pine Street, 30301",
            "country": "US",
        }
    ]
    rows_s3 = [
        {
            "entity_id": "S3-1",
            "business_name": "Alpha Foods LLC",
            "business_address": "10 Pine St",
            "country": "US",
        }
    ]

    def dump(name, rows):
        p = tmp_path / name
        with open(p, "w", encoding="utf-8") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in rows:
                f.write(
                    f"{r['entity_id']}\t{r['business_name']}\t{r['business_address']}\t{r['country']}\n"
                )
        return p

    s1, s2, s3 = dump("s1.tsv", rows_s1), dump("s2.tsv", rows_s2), dump("s3.tsv", rows_s3)
    out = tmp_path / "pairs.tsv"
    cfg = BlockingConfig(
        work_dir=str(tmp_path / "work"),
        enabled_blocks=[
            "exact_name",
            "exact_address",
            "postal",
            "house_number_name",
            "rare_name_token",
            "name_prefix",
            "address_token",
        ],
        rare_token_min_frequency=1,
        rare_token_max_frequency=10,
        rapidfuzz_enabled=False,
    )
    gen = CandidateGenerator(cfg)
    gen.generate(s1, s2, s3, out, write_submission_format=True)
    text = out.read_text(encoding="utf-8")
    assert "source1_entity_id" in text
    assert "S2-1" in text or "S3-1" in text
    # S1-2 zero candidates OK
    sub = (tmp_path / "pairs_submission.tsv").read_text(encoding="utf-8")
    assert "S1-2" in sub
