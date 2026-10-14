#!/usr/bin/env python3
"""Hybrid semantic + time + metadata search against ghw_vector_events."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import psycopg2

from seed_embeddings import embedding_from_text, halfvec_literal


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    service = os.environ.get("TIGER_SERVICE", "stillbench")
    out = subprocess.check_output(
        ["tiger", "db", "uri", service, "--with-password"],
        text=True,
    )
    return out.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid vector search demo")
    parser.add_argument("query", help="Natural-language search text")
    parser.add_argument("--location", default="floor", help="Metadata filter")
    parser.add_argument("--hours", type=int, default=168, help="Time window (hours)")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    query_vec = halfvec_literal(embedding_from_text(args.query))

    sql = """
    SET hnsw.ef_search = 100;
    SET hnsw.iterative_scan = relaxed_order;

    SELECT
      time,
      sensor_id,
      location,
      event_type,
      note,
      embedding <=> %s::halfvec(384) AS cosine_distance
    FROM ghw_vector_events
    WHERE location = %s
      AND time >= now() - (%s || ' hours')::interval
    ORDER BY embedding <=> %s::halfvec(384)
    LIMIT %s;
    """

    conn = psycopg2.connect(database_url())
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (query_vec, args.location, str(args.hours), query_vec, args.limit),
        )
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No matches (check seed data or widen --hours).")
        sys.exit(1)

    print(f"Query: {args.query!r}  |  location={args.location!r}  |  last {args.hours}h\n")
    for time, sensor_id, location, event_type, note, dist in rows:
        print(f"{time.isoformat()}  {sensor_id}  {location}  {event_type}  dist={dist:.4f}")
        print(f"  {note}\n")


if __name__ == "__main__":
    main()
