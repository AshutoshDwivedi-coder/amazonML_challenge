"""I/O helpers: TSV → normalized Parquet via chunked processing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .normalize import normalize_record

logger = logging.getLogger(__name__)

SOURCE_SCHEMA = pa.schema(
    [
        ("entity_id", pa.string()),
        ("business_name", pa.string()),
        ("business_address", pa.string()),
        ("country", pa.string()),
        ("name_norm", pa.string()),
        ("address_norm", pa.string()),
        ("name_tokens", pa.list_(pa.string())),
        ("address_tokens", pa.list_(pa.string())),
        ("country_norm", pa.string()),
        ("postal_code", pa.string()),
        ("house_number", pa.string()),
        ("name_prefix_4", pa.string()),
        ("name_prefix_6", pa.string()),
        ("name_prefix_8", pa.string()),
    ]
)


def iter_tsv_rows(path: str | Path, chunk_size: int = 200_000) -> Iterator[list[dict]]:
    path = Path(path)
    batch: list[dict] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        header = f.readline()
        if not header:
            return
        cols = [c.strip() for c in header.rstrip("\n").split("\t")]
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            # pad / truncate to header length
            while len(parts) < len(cols):
                parts.append("")
            row = {cols[i]: parts[i] for i in range(len(cols))}
            batch.append(row)
            if len(batch) >= chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch


def normalize_source_to_parquet(
    tsv_path: str | Path,
    parquet_path: str | Path,
    *,
    chunk_size: int = 200_000,
) -> int:
    """Normalize a source TSV into a Parquet file. Returns row count."""
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total = 0
    for batch in iter_tsv_rows(tsv_path, chunk_size=chunk_size):
        records = [
            normalize_record(
                row.get("entity_id", ""),
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", ""),
            )
            for row in batch
        ]
        table = pa.Table.from_pylist(records, schema=SOURCE_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, SOURCE_SCHEMA, compression="zstd")
        writer.write_table(table)
        total += len(records)
        if total % (chunk_size * 2) == 0 or total == len(records):
            logger.info("Normalized %s rows → %s", f"{total:,}", parquet_path.name)
    if writer is not None:
        writer.close()
    else:
        # empty file
        pq.write_table(pa.Table.from_pylist([], schema=SOURCE_SCHEMA), parquet_path)
    logger.info("Done normalizing %s (%s rows)", parquet_path.name, f"{total:,}")
    return total


def ensure_normalized(
    tsv_path: str | Path,
    work_dir: str | Path,
    label: str,
    *,
    chunk_size: int = 200_000,
    force: bool = False,
) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{label}_normalized.parquet"
    tsv_mtime = Path(tsv_path).stat().st_mtime
    if out.exists() and not force:
        if out.stat().st_mtime >= tsv_mtime:
            logger.info("Reusing cached %s", out)
            return out
    normalize_source_to_parquet(tsv_path, out, chunk_size=chunk_size)
    return out
