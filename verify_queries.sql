-- Good MLH submission screenshots: extensions, hypertable, hybrid vector search

SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector', 'vectorscale', 'timescaledb')
ORDER BY extname;

SELECT hypertable_schema, hypertable_name, num_dimensions
FROM timescaledb_information.hypertables
WHERE hypertable_name = 'ghw_vector_events';

SELECT count(*) AS row_count FROM ghw_vector_events;

SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'ghw_vector_events'
ORDER BY indexname;

-- Hybrid RAG: semantic nearest neighbors + time + metadata filters
SET hnsw.ef_search = 100;
SET hnsw.iterative_scan = relaxed_order;

WITH query AS (
  SELECT embedding AS q
  FROM ghw_vector_events
  WHERE note ILIKE '%vibration%'
  ORDER BY time DESC
  LIMIT 1
)
SELECT
  e.time,
  e.sensor_id,
  e.location,
  e.event_type,
  e.note,
  e.embedding <=> q.q AS cosine_distance
FROM ghw_vector_events e
CROSS JOIN query q
WHERE e.location = 'floor'
  AND e.time >= now() - interval '14 days'
ORDER BY e.embedding <=> q.q
LIMIT 5;
