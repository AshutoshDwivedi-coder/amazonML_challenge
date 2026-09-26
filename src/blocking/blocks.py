"""Blocking key builders and DuckDB-backed inverted-index retrieval."""
from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import duckdb

from .config import BlockingConfig
from .normalize import char_ngrams

logger = logging.getLogger(__name__)


@dataclass
class BlockResult:
    block_name: str
    source: str  # S2 / S3 / ALL
    pairs: list[tuple[str, str, str]]  # (s1_id, cand_id, source)
    runtime_seconds: float
    index_build_seconds: float = 0.0
    query_seconds: float = 0.0
    pairs_path: str | None = None  # optional on-disk spill


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")
    return con


def _register(con: duckdb.DuckDBPyConnection, path: str, name: str) -> None:
    # DuckDB disallows prepared params in CREATE VIEW — embed escaped path.
    escaped = path.replace("'", "''")
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')"
    )


def _freq_filter_join(
    con: duckdb.DuckDBPyConnection,
    *,
    s1_sql: str,
    cand_sql: str,
    key_expr: str,
    max_freq: int,
    source_label: str,
) -> list[tuple[str, str, str]]:
    """Join S1 and candidate keys with frequency cap on the candidate side."""
    sql = f"""
    WITH cand_keys AS (
        {cand_sql}
    ),
    freqs AS (
        SELECT block_key, COUNT(*) AS freq
        FROM cand_keys
        WHERE block_key IS NOT NULL AND block_key <> ''
        GROUP BY 1
        HAVING COUNT(*) <= {int(max_freq)}
    ),
    s1_keys AS (
        {s1_sql}
    )
    SELECT DISTINCT s.entity_id AS s1_id, c.entity_id AS cand_id
    FROM s1_keys s
    JOIN freqs f ON s.block_key = f.block_key
    JOIN cand_keys c ON c.block_key = s.block_key
    WHERE s.block_key IS NOT NULL AND s.block_key <> ''
    """
    rows = con.execute(sql).fetchall()
    return [(r[0], r[1], source_label) for r in rows]


def block_exact_name(
    s1_path: str, cand_path: str, source: str, cfg: BlockingConfig
) -> BlockResult:
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    key = "country_norm || '|' || name_norm"
    s1_sql = f"SELECT entity_id, {key} AS block_key FROM s1 WHERE name_norm <> ''"
    cand_sql = f"SELECT entity_id, {key} AS block_key FROM cand WHERE name_norm <> ''"
    t_idx = time.perf_counter()
    pairs = _freq_filter_join(
        con,
        s1_sql=s1_sql,
        cand_sql=cand_sql,
        key_expr=key,
        max_freq=cfg.max_name_frequency,
        source_label=source,
    )
    t1 = time.perf_counter()
    return BlockResult("exact_name", source, pairs, t1 - t0, t_idx - t0, t1 - t_idx)


def block_exact_address(
    s1_path: str, cand_path: str, source: str, cfg: BlockingConfig
) -> BlockResult:
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    key = "country_norm || '|' || address_norm"
    s1_sql = f"SELECT entity_id, {key} AS block_key FROM s1 WHERE address_norm <> ''"
    cand_sql = f"SELECT entity_id, {key} AS block_key FROM cand WHERE address_norm <> ''"
    t_idx = time.perf_counter()
    pairs = _freq_filter_join(
        con,
        s1_sql=s1_sql,
        cand_sql=cand_sql,
        key_expr=key,
        max_freq=cfg.max_address_frequency,
        source_label=source,
    )
    t1 = time.perf_counter()
    return BlockResult("exact_address", source, pairs, t1 - t0, t_idx - t0, t1 - t_idx)


def block_exact_name_address(
    s1_path: str, cand_path: str, source: str, cfg: BlockingConfig
) -> BlockResult:
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    key = "country_norm || '|' || name_norm || '|' || address_norm"
    s1_sql = (
        f"SELECT entity_id, {key} AS block_key FROM s1 "
        "WHERE name_norm <> '' AND address_norm <> ''"
    )
    cand_sql = (
        f"SELECT entity_id, {key} AS block_key FROM cand "
        "WHERE name_norm <> '' AND address_norm <> ''"
    )
    t_idx = time.perf_counter()
    pairs = _freq_filter_join(
        con,
        s1_sql=s1_sql,
        cand_sql=cand_sql,
        key_expr=key,
        max_freq=cfg.max_name_address_frequency,
        source_label=source,
    )
    t1 = time.perf_counter()
    return BlockResult(
        "exact_name_address", source, pairs, t1 - t0, t_idx - t0, t1 - t_idx
    )


def block_postal(
    s1_path: str, cand_path: str, source: str, cfg: BlockingConfig
) -> BlockResult:
    """Postal blocks: country+postal+name_prefix and country+postal+house (never postal alone)."""
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    pairs: list[tuple[str, str, str]] = []

    # postal + name_prefix_4
    key1 = "country_norm || '|P|' || postal_code || '|' || name_prefix_4"
    s1_sql = (
        f"SELECT entity_id, {key1} AS block_key FROM s1 "
        "WHERE postal_code <> '' AND name_prefix_4 <> ''"
    )
    cand_sql = (
        f"SELECT entity_id, {key1} AS block_key FROM cand "
        "WHERE postal_code <> '' AND name_prefix_4 <> ''"
    )
    pairs.extend(
        _freq_filter_join(
            con,
            s1_sql=s1_sql,
            cand_sql=cand_sql,
            key_expr=key1,
            max_freq=cfg.max_postal_name_prefix_frequency,
            source_label=source,
        )
    )

    # postal + house_number
    key2 = "country_norm || '|PH|' || postal_code || '|' || house_number"
    s1_sql2 = (
        f"SELECT entity_id, {key2} AS block_key FROM s1 "
        "WHERE postal_code <> '' AND house_number <> ''"
    )
    cand_sql2 = (
        f"SELECT entity_id, {key2} AS block_key FROM cand "
        "WHERE postal_code <> '' AND house_number <> ''"
    )
    pairs.extend(
        _freq_filter_join(
            con,
            s1_sql=s1_sql2,
            cand_sql=cand_sql2,
            key_expr=key2,
            max_freq=cfg.max_postal_house_frequency,
            source_label=source,
        )
    )

    # dedupe within block
    pairs = list({(a, b, c) for a, b, c in pairs})
    t1 = time.perf_counter()
    return BlockResult("postal", source, pairs, t1 - t0)


def block_house_number_name(
    s1_path: str, cand_path: str, source: str, cfg: BlockingConfig
) -> BlockResult:
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    pairs: list[tuple[str, str, str]] = []
    for pref, col in ((4, "name_prefix_4"), (6, "name_prefix_6")):
        key = f"country_norm || '|H{pref}|' || house_number || '|' || {col}"
        s1_sql = (
            f"SELECT entity_id, {key} AS block_key FROM s1 "
            f"WHERE house_number <> '' AND {col} <> ''"
        )
        cand_sql = (
            f"SELECT entity_id, {key} AS block_key FROM cand "
            f"WHERE house_number <> '' AND {col} <> ''"
        )
        pairs.extend(
            _freq_filter_join(
                con,
                s1_sql=s1_sql,
                cand_sql=cand_sql,
                key_expr=key,
                max_freq=cfg.max_house_name_frequency,
                source_label=source,
            )
        )
    pairs = list({(a, b, c) for a, b, c in pairs})
    t1 = time.perf_counter()
    return BlockResult("house_number_name", source, pairs, t1 - t0)


def block_name_prefix(
    s1_path: str, cand_path: str, source: str, cfg: BlockingConfig
) -> BlockResult:
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    pairs: list[tuple[str, str, str]] = []
    for length in cfg.name_prefix_lengths:
        col = f"name_prefix_{length}"
        key = f"country_norm || '|NP{length}|' || {col}"
        s1_sql = f"SELECT entity_id, {key} AS block_key FROM s1 WHERE {col} <> ''"
        cand_sql = f"SELECT entity_id, {key} AS block_key FROM cand WHERE {col} <> ''"
        pairs.extend(
            _freq_filter_join(
                con,
                s1_sql=s1_sql,
                cand_sql=cand_sql,
                key_expr=key,
                max_freq=cfg.max_prefix_frequency,
                source_label=source,
            )
        )
    pairs = list({(a, b, c) for a, b, c in pairs})
    t1 = time.perf_counter()
    return BlockResult("name_prefix", source, pairs, t1 - t0)


def block_rare_name_token(
    s1_path: str,
    cand_path: str,
    source: str,
    cfg: BlockingConfig,
    token_df: dict[str, int] | None = None,
) -> BlockResult:
    """Rare name-token blocking via DuckDB (scalable; no Python row materialization)."""
    del token_df  # DF computed inside DuckDB for scalability
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    min_f = int(cfg.rare_token_min_frequency)
    max_f = int(cfg.rare_token_max_frequency)
    min_len = int(cfg.rare_token_min_length)
    max_block = int(cfg.max_rare_token_block_size)
    t_idx = time.perf_counter()
    sql = f"""
    WITH token_df AS (
        SELECT tok, COUNT(DISTINCT entity_id) AS freq
        FROM cand, UNNEST(name_tokens) AS u(tok)
        WHERE tok IS NOT NULL AND length(tok) >= {min_len}
        GROUP BY tok
        HAVING COUNT(DISTINCT entity_id) BETWEEN {min_f} AND {max_f}
    ),
    cand_keys AS (
        SELECT c.entity_id,
               c.country_norm || '|RNT|' || u.tok AS block_key
        FROM cand c, UNNEST(c.name_tokens) AS u(tok)
        JOIN token_df t ON t.tok = u.tok
        WHERE c.country_norm IS NOT NULL
    ),
    freqs AS (
        SELECT block_key FROM cand_keys
        GROUP BY 1 HAVING COUNT(*) <= {max_block}
    ),
    s1_keys AS (
        SELECT s.entity_id,
               s.country_norm || '|RNT|' || u.tok AS block_key
        FROM s1 s, UNNEST(s.name_tokens) AS u(tok)
        JOIN token_df t ON t.tok = u.tok
    )
    SELECT DISTINCT s.entity_id, c.entity_id
    FROM s1_keys s
    JOIN freqs f ON s.block_key = f.block_key
    JOIN cand_keys c ON c.block_key = s.block_key
    """
    rows = con.execute(sql).fetchall()
    pairs = [(r[0], r[1], source) for r in rows]
    t1 = time.perf_counter()
    return BlockResult(
        "rare_name_token", source, pairs, t1 - t0, t_idx - t0, t1 - t_idx
    )


def block_rare_address_token(
    s1_path: str,
    cand_path: str,
    source: str,
    cfg: BlockingConfig,
    token_df: dict[str, int] | None = None,
) -> BlockResult:
    """Rare address-token (+ optional house) blocking via DuckDB."""
    del token_df
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    min_f = int(cfg.rare_token_min_frequency)
    max_f = int(cfg.rare_token_max_frequency)
    min_len = int(cfg.rare_token_min_length)
    max_block = int(cfg.max_address_token_block_size)
    t_idx = time.perf_counter()
    sql = f"""
    WITH token_df AS (
        SELECT tok, COUNT(DISTINCT entity_id) AS freq
        FROM cand, UNNEST(address_tokens) AS u(tok)
        WHERE tok IS NOT NULL AND length(tok) >= {min_len}
        GROUP BY tok
        HAVING COUNT(DISTINCT entity_id) BETWEEN {min_f} AND {max_f}
    ),
    cand_keys AS (
        SELECT c.entity_id,
               c.country_norm || '|RAT|' || u.tok AS block_key
        FROM cand c, UNNEST(c.address_tokens) AS u(tok)
        JOIN token_df t ON t.tok = u.tok
        UNION ALL
        SELECT c.entity_id,
               c.country_norm || '|RHAT|' || c.house_number || '|' || u.tok AS block_key
        FROM cand c, UNNEST(c.address_tokens) AS u(tok)
        JOIN token_df t ON t.tok = u.tok
        WHERE c.house_number IS NOT NULL AND c.house_number <> ''
    ),
    freqs AS (
        SELECT block_key FROM cand_keys
        GROUP BY 1 HAVING COUNT(*) <= {max_block}
    ),
    s1_keys AS (
        SELECT s.entity_id,
               s.country_norm || '|RAT|' || u.tok AS block_key
        FROM s1 s, UNNEST(s.address_tokens) AS u(tok)
        JOIN token_df t ON t.tok = u.tok
        UNION ALL
        SELECT s.entity_id,
               s.country_norm || '|RHAT|' || s.house_number || '|' || u.tok AS block_key
        FROM s1 s, UNNEST(s.address_tokens) AS u(tok)
        JOIN token_df t ON t.tok = u.tok
        WHERE s.house_number IS NOT NULL AND s.house_number <> ''
    )
    SELECT DISTINCT s.entity_id, c.entity_id
    FROM s1_keys s
    JOIN freqs f ON s.block_key = f.block_key
    JOIN cand_keys c ON c.block_key = s.block_key
    """
    rows = con.execute(sql).fetchall()
    pairs = [(r[0], r[1], source) for r in rows]
    t1 = time.perf_counter()
    return BlockResult(
        "rare_address_token", source, pairs, t1 - t0, t_idx - t0, t1 - t_idx
    )


def block_address_token(
    s1_path: str, cand_path: str, source: str, cfg: BlockingConfig
) -> BlockResult:
    """Address signature: country + house + address_prefix_6 (from address_norm)."""
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")
    # address prefix from compacted address_norm
    key = (
        "country_norm || '|AP|' || house_number || '|' || "
        "substr(replace(address_norm, ' ', ''), 1, 8)"
    )
    s1_sql = (
        f"SELECT entity_id, {key} AS block_key FROM s1 "
        "WHERE house_number <> '' AND address_norm <> '' "
        "AND length(replace(address_norm, ' ', '')) >= 4"
    )
    cand_sql = (
        f"SELECT entity_id, {key} AS block_key FROM cand "
        "WHERE house_number <> '' AND address_norm <> '' "
        "AND length(replace(address_norm, ' ', '')) >= 4"
    )
    pairs = _freq_filter_join(
        con,
        s1_sql=s1_sql,
        cand_sql=cand_sql,
        key_expr=key,
        max_freq=cfg.max_address_token_block_size,
        source_label=source,
    )
    t1 = time.perf_counter()
    return BlockResult("address_token", source, pairs, t1 - t0)


def block_char_ngram(
    s1_path: str,
    cand_path: str,
    source: str,
    cfg: BlockingConfig,
    *,
    only_s1_ids: set[str] | None = None,
) -> BlockResult:
    """Country-partitioned character n-gram retrieval (no full pairwise matrix).

    Primarily intended for residual / difficult S1s when only_s1_ids is set.
    Indexes only candidates in residual countries, and prefers prefix-bucketed
    candidate pools to keep memory bounded.
    """
    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")

    if only_s1_ids is not None:
        s1_rows = con.execute(
            "SELECT entity_id, country_norm, name_norm, name_prefix_4 FROM s1 "
            "WHERE name_norm <> ''"
        ).fetchall()
        s1_rows = [r for r in s1_rows if r[0] in only_s1_ids]
    else:
        s1_rows = con.execute(
            "SELECT entity_id, country_norm, name_norm, name_prefix_4 FROM s1 "
            "WHERE name_norm <> ''"
        ).fetchall()

    if not s1_rows:
        return BlockResult("char_ngram", source, [], time.perf_counter() - t0)

    countries = {r[1] for r in s1_rows}
    prefixes = {r[3] for r in s1_rows if r[3]}

    # Restrict candidate set: same country AND (shared prefix_4 when available)
    cand_rows = con.execute(
        "SELECT entity_id, country_norm, name_norm, name_prefix_4 FROM cand "
        "WHERE name_norm <> ''"
    ).fetchall()
    cand_rows = [
        r
        for r in cand_rows
        if r[1] in countries and (not prefixes or r[3] in prefixes)
    ]

    # inverted index: (country, ngram) -> list[entity_id]
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for eid, country, name, _pref in cand_rows:
        grams: set[str] = set()
        for n in cfg.char_ngram_sizes:
            grams.update(char_ngrams(name, n))
        for g in grams:
            if len(g) >= 3:
                index[(country, g)].append(eid)

    index = {
        k: v for k, v in index.items() if 0 < len(v) <= cfg.max_ngram_block_size
    }
    t_idx = time.perf_counter()

    pairs: list[tuple[str, str, str]] = []
    for eid, country, name, _pref in s1_rows:
        scores: Counter[str] = Counter()
        grams: set[str] = set()
        for n in cfg.char_ngram_sizes:
            grams.update(char_ngrams(name, n))
        for g in grams:
            if len(g) < 3:
                continue
            for cid in index.get((country, g), []):
                scores[cid] += 1
        ranked = [
            (cid, sc)
            for cid, sc in scores.most_common(cfg.char_ngram_top_k * 3)
            if sc >= cfg.char_ngram_min_shared
        ][: cfg.char_ngram_top_k]
        for cid, _ in ranked:
            pairs.append((eid, cid, source))

    t1 = time.perf_counter()
    return BlockResult("char_ngram", source, pairs, t1 - t0, t_idx - t0, t1 - t_idx)


def block_rapidfuzz(
    s1_path: str,
    cand_path: str,
    source: str,
    cfg: BlockingConfig,
    *,
    only_s1_ids: set[str] | None = None,
    candidate_pools: dict[str, set[str]] | None = None,
) -> BlockResult:
    """RapidFuzz over small per-S1 pools (never full Cartesian product).

    If candidate_pools is provided, score only within those pools.
    Else, fall back to country + name_prefix_4 buckets (frequency-capped).
    """
    from rapidfuzz import fuzz, process

    t0 = time.perf_counter()
    con = _connect()
    _register(con, s1_path, "s1")
    _register(con, cand_path, "cand")

    cand_rows = con.execute(
        "SELECT entity_id, country_norm, name_norm, name_prefix_4 FROM cand "
        "WHERE name_norm <> ''"
    ).fetchall()
    cand_name = {r[0]: r[2] for r in cand_rows}
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for eid, country, _name, pref in cand_rows:
        if pref:
            by_bucket[f"{country}|{pref}"].append(eid)
    by_bucket = {
        k: v for k, v in by_bucket.items() if len(v) <= cfg.rapidfuzz_max_pool
    }
    t_idx = time.perf_counter()

    s1_rows = con.execute(
        "SELECT entity_id, country_norm, name_norm, name_prefix_4 FROM s1 "
        "WHERE name_norm <> ''"
    ).fetchall()
    if only_s1_ids is not None:
        s1_rows = [r for r in s1_rows if r[0] in only_s1_ids]

    pairs: list[tuple[str, str, str]] = []
    for eid, country, name, pref in s1_rows:
        if candidate_pools and eid in candidate_pools:
            pool_ids = [c for c in candidate_pools[eid] if c in cand_name]
        else:
            pool_ids = by_bucket.get(f"{country}|{pref}", [])
        if not pool_ids:
            continue
        # cap pool
        if len(pool_ids) > cfg.rapidfuzz_max_pool:
            pool_ids = pool_ids[: cfg.rapidfuzz_max_pool]
        choices = {cid: cand_name[cid] for cid in pool_ids}
        hits = process.extract(
            name,
            choices,
            scorer=fuzz.WRatio,
            limit=cfg.rapidfuzz_top_k,
            score_cutoff=cfg.rapidfuzz_score_cutoff,
        )
        for _choice, _score, cid in hits:
            pairs.append((eid, cid, source))

    t1 = time.perf_counter()
    return BlockResult("rapidfuzz", source, pairs, t1 - t0, t_idx - t0, t1 - t_idx)


BLOCK_FUNCS = {
    "exact_name": block_exact_name,
    "exact_address": block_exact_address,
    "exact_name_address": block_exact_name_address,
    "postal": block_postal,
    "house_number_name": block_house_number_name,
    "name_prefix": block_name_prefix,
    "address_token": block_address_token,
    # rare_* and char_ngram / rapidfuzz need extra kwargs — handled in generator
}
