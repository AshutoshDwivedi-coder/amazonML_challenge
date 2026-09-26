"""Robust name/address normalization for entity blocking.

Preserves original fields; derived fields are additive only.
No external lookups / geocoding / registries.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from unidecode import unidecode

# ---------------------------------------------------------------------------
# Legal / organizational suffixes (safe to strip for matching keys)
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "llc",
        "llp",
        "lp",
        "ltd",
        "limited",
        "co",
        "company",
        "plc",
        "gmbh",
        "ag",
        "sa",
        "sas",
        "sarl",
        "bv",
        "nv",
        "oy",
        "ab",
        "pty",
        "pvt",
        "private",
        "pc",
        "pllc",
        "lc",
        "lllp",
    }
)

# Safe abbreviation expansions (token-level)
NAME_ABBREVS = {
    "intl": "international",
    "int'l": "international",
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "tech": "technology",
    "mgmt": "management",
    "dept": "department",
    "svcs": "services",
    "svc": "service",
    "assoc": "associates",
    "bros": "brothers",
    "bros.": "brothers",
}

STREET_ABBREVS = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "boul": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "pkwy": "parkway",
    "hwy": "highway",
    "fwy": "freeway",
    "expy": "expressway",
    "tpke": "turnpike",
    "ter": "terrace",
    "trl": "trail",
    "way": "way",
    "aly": "alley",
    "sq": "square",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "rm": "room",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "mt": "mount",
    "ft": "fort",
}

UNIT_PATTERN = re.compile(
    r"\b(?:apt|apartment|suite|ste|unit|fl|floor|rm|room|bldg|building|#)\s*[\w\-]+\b",
    re.IGNORECASE,
)
PO_BOX_PATTERN = re.compile(r"\b(?:p\.?\s*o\.?\s*box|post\s*office\s*box)\s*\w+\b", re.IGNORECASE)

# Country-agnostic postal patterns (open-set; no country hard-coding)
POSTAL_PATTERNS = [
    # US ZIP / ZIP+4
    re.compile(r"\b(\d{5})(?:-\d{4})?\b"),
    # India PIN
    re.compile(r"\b(\d{6})\b"),
    # France / generic 5-digit already covered; alphanumeric (CA/UK-like)
    re.compile(r"\b([A-Z]\d[A-Z]\s?\d[A-Z]\d)\b", re.IGNORECASE),
    re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b", re.IGNORECASE),
]

HOUSE_NUMBER_PATTERN = re.compile(
    r"(?:^|[\s,])(\d{1,6}[A-Za-z]?(?:/\d{1,4})?(?:-\d{1,4})?)(?=\s|,|$)"
)
# Prefer leading street number
LEADING_HOUSE_PATTERN = re.compile(r"^\s*(\d{1,6}[A-Za-z]?(?:/\d{1,4})?)")

PUNCT_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
MULTISPACE = re.compile(r"\s+")
AND_PATTERN = re.compile(r"\s&\s|&")


def _nfkc_lower(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    return text.lower().strip()


def _transliterate(text: str) -> str:
    """Best-effort ASCII fold without external lookups."""
    try:
        return unidecode(text)
    except Exception:
        return text.encode("ascii", "ignore").decode("ascii")


def basic_clean(text: str, *, transliterate: bool = True) -> str:
    text = _nfkc_lower(text)
    if transliterate:
        text = _transliterate(text).lower()
    text = AND_PATTERN.sub(" and ", text)
    # Join possessive / contraction apostrophes before wiping punctuation
    text = re.sub(r"[''`]", "", text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = MULTISPACE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t for t in text.split() if t]


def expand_abbreviations(tokens: Iterable[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(t, t) for t in tokens]


def strip_legal_suffixes(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens
    out = list(tokens)
    while out and out[-1] in LEGAL_SUFFIXES:
        out.pop()
    # also strip leading LLC-style if whole name is "LLC Foo"
    while out and out[0] in LEGAL_SUFFIXES and len(out) > 1:
        out.pop(0)
    return out


def normalize_name(name: str) -> str:
    cleaned = basic_clean(name)
    tokens = tokenize(cleaned)
    tokens = expand_abbreviations(tokens, NAME_ABBREVS)
    tokens = strip_legal_suffixes(tokens)
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    cleaned = basic_clean(address)
    cleaned = PO_BOX_PATTERN.sub(" ", cleaned)
    cleaned = UNIT_PATTERN.sub(" ", cleaned)
    tokens = tokenize(cleaned)
    tokens = expand_abbreviations(tokens, STREET_ABBREVS)
    return MULTISPACE.sub(" ", " ".join(tokens)).strip()


def normalize_country(country: str) -> str:
    """Open-set country key: NFKC + lower + strip. Never hard-code a country list."""
    return _nfkc_lower(country)


def extract_postal_code(address: str, country: str | None = None) -> str:
    """Extract a postal/PIN-like token. Country unused for branching (open-set)."""
    del country  # open-set: do not special-case by country name
    if not address:
        return ""
    text = unicodedata.normalize("NFKC", str(address))
    # Prefer more specific patterns first
    for pattern in POSTAL_PATTERNS:
        m = pattern.search(text)
        if m:
            return re.sub(r"\s+", "", m.group(1)).upper()
    return ""


def extract_house_number(address: str) -> str:
    if not address:
        return ""
    text = unicodedata.normalize("NFKC", str(address))
    m = LEADING_HOUSE_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    # fallback: first standalone number-like token in address
    m = HOUSE_NUMBER_PATTERN.search(text)
    if m:
        return m.group(1).upper()
    return ""


def name_prefix(name_norm: str, length: int) -> str:
    compact = re.sub(r"\s+", "", name_norm or "")
    if len(compact) < length:
        return compact if compact else ""
    return compact[:length]


def char_ngrams(text: str, n: int) -> list[str]:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


def rare_tokens(
    tokens: list[str],
    df: dict[str, int],
    *,
    min_freq: int,
    max_freq: int,
    min_length: int,
    limit: int,
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for t in tokens:
        if len(t) < min_length:
            continue
        if t in LEGAL_SUFFIXES or t in STREET_ABBREVS.values() or t in STREET_ABBREVS:
            continue
        freq = df.get(t, 0)
        if min_freq <= freq <= max_freq:
            scored.append((freq, t))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [t for _, t in scored[:limit]]


def normalize_record(
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str,
) -> dict:
    """Return original + derived fields for one record."""
    name_raw = "" if business_name is None else str(business_name)
    addr_raw = "" if business_address is None else str(business_address)
    country_raw = "" if country is None else str(country)

    name_norm = normalize_name(name_raw)
    address_norm = normalize_address(addr_raw)
    country_norm = normalize_country(country_raw)
    name_tokens = tokenize(name_norm)
    address_tokens = tokenize(address_norm)
    postal = extract_postal_code(addr_raw, country_raw)
    house = extract_house_number(addr_raw)

    return {
        "entity_id": entity_id,
        "business_name": name_raw,
        "business_address": addr_raw,
        "country": country_raw,
        "name_norm": name_norm,
        "address_norm": address_norm,
        "name_tokens": name_tokens,
        "address_tokens": address_tokens,
        "country_norm": country_norm,
        "postal_code": postal,
        "house_number": house,
        "name_prefix_4": name_prefix(name_norm, 4),
        "name_prefix_6": name_prefix(name_norm, 6),
        "name_prefix_8": name_prefix(name_norm, 8),
    }
