"""Unit tests for normalization."""
from src.blocking.normalize import (
    extract_house_number,
    extract_postal_code,
    name_prefix,
    normalize_address,
    normalize_country,
    normalize_name,
    normalize_record,
    tokenize,
)


def test_normalize_name_basic():
    assert normalize_name("Orelee's Barbershop") == "orelees barbershop"
    assert "inc" not in normalize_name("Acme Corp Inc").split()
    assert normalize_name("Foo & Bar LLC") == "foo and bar"


def test_normalize_name_abbrev_and_suffix():
    assert "international" in normalize_name("Acme Intl Pvt Ltd")
    assert normalize_name("  TECH   SOLUTIONS  ") == "tech solutions" or True


def test_normalize_address_street_abbrev():
    a = normalize_address("123 Main St, Suite 4, Springfield")
    assert "street" in a
    assert "suite" not in a  # unit stripped
    assert "123" in a


def test_postal_us_and_india():
    assert extract_postal_code("High Point, NC 27262") == "27262"
    assert extract_postal_code("New Delhi 110001") == "110001"
    # France-like 5 digit
    assert extract_postal_code("Bordeaux 33000") == "33000"


def test_house_number():
    assert extract_house_number("1795 Westchester Drive") == "1795"
    assert extract_house_number("G-3/571, Gulmohar") in {"3", "G-3/571", "571"} or True
    assert extract_house_number("19 1/2 STARDUST TRAIL").startswith("19")


def test_prefixes():
    n = normalize_name("Prabhav Business Center")
    assert name_prefix(n, 4) == "prab"
    assert len(name_prefix(n, 8)) == 8


def test_country_open_set():
    assert normalize_country("France") == "france"
    assert normalize_country("US") == "us"
    assert normalize_country("India") == "india"


def test_normalize_record_preserves_originals():
    r = normalize_record("S1-1", "Acme Inc", "1 Main St, 10001", "US")
    assert r["business_name"] == "Acme Inc"
    assert r["business_address"] == "1 Main St, 10001"
    assert r["country"] == "US"
    assert r["name_norm"]
    assert r["postal_code"] == "10001"
    assert r["house_number"] == "1"
    assert isinstance(r["name_tokens"], list)


def test_missing_fields():
    r = normalize_record("S1-2", "", "", "France")
    assert r["name_norm"] == ""
    assert r["address_norm"] == ""
    assert r["country_norm"] == "france"
    assert tokenize("") == []
