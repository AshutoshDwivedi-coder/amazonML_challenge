"""Error analysis for true matches missed by blocking."""
from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from pathlib import Path

import duckdb

from .normalize import (
    extract_house_number,
    extract_postal_code,
    normalize_address,
    normalize_name,
)

logger = logging.getLogger(__name__)


def _categorize(
    s1_name: str,
    cand_name: str,
    s1_addr: str,
    cand_addr: str,
) -> str:
    n1, n2 = normalize_name(s1_name), normalize_name(cand_name)
    a1, a2 = normalize_address(s1_addr), normalize_address(cand_addr)

    if not (s1_name or "").strip() or not (cand_name or "").strip():
        return "MISSING_NAME"
    if not (s1_addr or "").strip() or not (cand_addr or "").strip():
        if n1 == n2 and n1:
            return "MISSING_ADDRESS"

    if n1 == n2 and a1 != a2:
        return "ADDRESS_VARIATION"
    if a1 == a2 and a1 and n1 != n2:
        # finer name categories
        if _only_suffix_diff(s1_name, cand_name):
            return "LEGAL_SUFFIX"
        if _punct_only(s1_name, cand_name):
            return "PUNCTUATION"
        if _abbrev_diff(n1, n2):
            return "ABBREVIATION"
        if _word_order(n1, n2):
            return "WORD_ORDER"
        if _likely_typo(n1, n2):
            return "TYPO"
        if _transliterationish(s1_name, cand_name):
            return "TRANSLITERATION"
        return "NAME_VARIATION"

    p1, p2 = extract_postal_code(s1_addr), extract_postal_code(cand_addr)
    if (p1 or p2) and p1 != p2:
        return "POSTAL_MISSING"
    h1, h2 = extract_house_number(s1_addr), extract_house_number(cand_addr)
    if (h1 or h2) and h1 != h2:
        return "HOUSE_NUMBER_MISSING"

    if _likely_typo(n1, n2):
        return "TYPO"
    if _abbrev_diff(n1, n2):
        return "ABBREVIATION"
    if _word_order(n1, n2):
        return "WORD_ORDER"
    if _punct_only(s1_name, cand_name):
        return "PUNCTUATION"
    if _transliterationish(s1_name, cand_name):
        return "TRANSLITERATION"
    if n1 != n2:
        return "NAME_VARIATION"
    if a1 != a2:
        return "ADDRESS_VARIATION"
    return "OTHER"


def _tokens(s: str) -> list[str]:
    return [t for t in (s or "").split() if t]


def _only_suffix_diff(a: str, b: str) -> bool:
    from .normalize import LEGAL_SUFFIXES, normalize_name

    return normalize_name(a) == normalize_name(b) and (
        set(_tokens(a.lower())) - set(_tokens(b.lower()))
        or set(_tokens(b.lower())) - set(_tokens(a.lower()))
    ).issubset(LEGAL_SUFFIXES | {t + "s" for t in LEGAL_SUFFIXES})


def _punct_only(a: str, b: str) -> bool:
    clean = lambda s: re.sub(r"[^\w\s]", "", (s or "").lower())
    return clean(a) == clean(b) and a != b


def _abbrev_diff(n1: str, n2: str) -> bool:
    t1, t2 = set(_tokens(n1)), set(_tokens(n2))
    if not t1 or not t2:
        return False
    # one side has prefix tokens of the other
    shorter, longer = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    return shorter < longer or any(
        any(x.startswith(y) or y.startswith(x) for y in longer) for x in shorter
    )


def _word_order(n1: str, n2: str) -> bool:
    t1, t2 = _tokens(n1), _tokens(n2)
    return t1 != t2 and sorted(t1) == sorted(t2)


def _likely_typo(n1: str, n2: str) -> bool:
    if not n1 or not n2:
        return False
    if abs(len(n1) - len(n2)) > 2:
        return False
    # simple Hamming / edit distance bound
    try:
        from rapidfuzz.distance import Levenshtein

        return 0 < Levenshtein.distance(n1, n2) <= 2
    except Exception:
        return False


def _transliterationish(a: str, b: str) -> bool:
    from unidecode import unidecode

    return unidecode(a or "").lower() != (a or "").lower() or unidecode(
        b or ""
    ).lower() != (b or "").lower()


def write_blocking_misses(
    ground_truth: dict[str, set[str]],
    candidates: dict[str, set[str]],
    s1_parquet: str | Path,
    s2_parquet: str | Path,
    s3_parquet: str | Path,
    output_path: str | Path,
    *,
    max_rows: int = 50_000,
) -> Counter:
    """Write missed true pairs with failure categories."""
    con = duckdb.connect(":memory:")

    def _view(name: str, path: str | Path) -> None:
        escaped = str(path).replace("'", "''")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')")

    _view("s1", s1_parquet)
    _view("s2", s2_parquet)
    _view("s3", s3_parquet)

    def lookup(table: str, eid: str) -> tuple[str, str, str]:
        row = con.execute(
            f"SELECT business_name, business_address, country FROM {table} WHERE entity_id = ?",
            [eid],
        ).fetchone()
        if not row:
            return "", "", ""
        return row[0] or "", row[1] or "", row[2] or ""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cats: Counter = Counter()
    written = 0

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "source1_entity_id",
                "true_candidate_id",
                "source",
                "s1_name",
                "candidate_name",
                "s1_address",
                "candidate_address",
                "country",
                "failed_blocks",
                "failure_category",
            ]
        )
        for s1, truth in ground_truth.items():
            cands = candidates.get(s1, set())
            misses = truth - cands
            if not misses:
                continue
            s1_name, s1_addr, country = lookup("s1", s1)
            for mid in misses:
                source = "S2" if mid.startswith("S2-") else "S3" if mid.startswith("S3-") else "?"
                table = "s2" if source == "S2" else "s3"
                cname, caddr, _c = lookup(table, mid)
                cat = _categorize(s1_name, cname, s1_addr, caddr)
                cats[cat] += 1
                w.writerow(
                    [
                        s1,
                        mid,
                        source,
                        s1_name,
                        cname,
                        s1_addr,
                        caddr,
                        country,
                        "ALL",
                        cat,
                    ]
                )
                written += 1
                if written >= max_rows:
                    break
            if written >= max_rows:
                break

    logger.info(
        "Wrote %s misses → %s; categories=%s",
        f"{written:,}",
        output_path,
        dict(cats.most_common()),
    )
    return cats
