#!/usr/bin/env python3
"""Seed ghw_vector_events with deterministic 384-d embeddings (no API key required)."""

from __future__ import annotations

import hashlib
import math
import os
import struct
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2.extras import Json, execute_values

DIM = 384
ROWS = [
    ("sensor-a", "floor", "temperature", "Temperature spike on floor sensor during peak load"),
    ("sensor-a", "floor", "temperature", "Cooling cycle restored; readings back in normal band"),
    ("sensor-b", "ceiling", "humidity", "Humidity climbed above comfort threshold near ceiling duct"),
    ("sensor-b", "ceiling", "humidity", "Dehumidifier engaged; relative humidity trending down"),
    ("sensor-c", "floor", "vibration", "Unusual vibration pattern detected on compressor mount"),
    ("sensor-c", "floor", "vibration", "Maintenance ticket: inspect bearing assembly on pump line"),
    ("sensor-d", "ceiling", "air_quality", "CO2 levels elevated during occupied hours"),
    ("sensor-d", "ceiling", "air_quality", "Ventilation increased; CO2 returned to baseline"),
    ("sensor-a", "floor", "power", "Brief power dip logged; UPS switched to battery"),
    ("sensor-b", "ceiling", "leak", "Moisture sensor triggered under rooftop HVAC unit"),
    ("sensor-c", "floor", "noise", "Acoustic anomaly correlated with nearby construction"),
    ("sensor-d", "ceiling", "filter", "Air filter differential pressure approaching replace threshold"),
]


def embedding_from_text(text: str, dim: int = DIM) -> list[float]:
    """Hash text into a unit-normalized vector (same model for insert + query)."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = hashlib.sha256(digest)
    values: list[float] = []
    while len(values) < dim:
        block = rng.digest()
        for i in range(0, len(block), 4):
            if len(values) >= dim:
                break
            chunk = block[i : i + 4]
            if len(chunk) < 4:
                break
            u = struct.unpack(">I", chunk.ljust(4, b"\0"))[0]
            values.append((u / 0xFFFFFFFF) * 2.0 - 1.0)
        rng = hashlib.sha256(block)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def halfvec_literal(vec: list[float]) -> str:
    inner = ",".join(f"{v:.8f}" for v in vec)
    return f"[{inner}]"


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    service = os.environ.get("TIGER_SERVICE", "stillbench")
    try:
        out = subprocess.check_output(
            ["tiger", "db", "uri", service, "--with-password"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            "Set DATABASE_URL or fix Tiger CLI auth "
            f"(tiger db save-password {service}).\n"
            f"{exc}"
        ) from exc
    return out.strip()


def main() -> None:
    now = datetime.now(timezone.utc)
    payload = []
    for idx, (sensor_id, location, event_type, note) in enumerate(ROWS):
        t = now - timedelta(hours=idx * 6)
        emb = embedding_from_text(note)
        payload.append(
            (
                t,
                sensor_id,
                location,
                event_type,
                note,
                Json({"source": "seed_embeddings.py", "idx": idx}),
                halfvec_literal(emb),
            )
        )

    conn = psycopg2.connect(database_url())
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("TRUNCATE ghw_vector_events;")
        execute_values(
            cur,
            """
            INSERT INTO ghw_vector_events
              (time, sensor_id, location, event_type, note, metadata, embedding)
            VALUES %s
            """,
            payload,
            template="(%s, %s, %s, %s, %s, %s, %s::halfvec(384))",
        )
        cur.execute("DROP INDEX IF EXISTS ghw_vector_events_embedding_hnsw_idx;")
        cur.execute(
            """
            CREATE INDEX ghw_vector_events_embedding_hnsw_idx
              ON ghw_vector_events
              USING hnsw (embedding halfvec_cosine_ops)
              WITH (m = 16, ef_construction = 64);
            """
        )
    conn.close()
    print(f"Inserted {len(payload)} rows and built HNSW index on ghw_vector_events.")


if __name__ == "__main__":
    main()
