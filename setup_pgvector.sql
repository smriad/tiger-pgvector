-- MLH GHW: Implement AI-Powered Vector Search with pgvector
-- Tiger Cloud service: stillbench (extends prior GHW time-series work)
-- Docs: https://www.tigerdata.com/docs/learn/search/key-vector-database-concepts-for-understanding-pgvector

CREATE EXTENSION IF NOT EXISTS vector;

-- Optional on Tiger Cloud; safe if already enabled
DO $$
BEGIN
  CREATE EXTENSION IF NOT EXISTS vectorscale;
EXCEPTION
  WHEN OTHERS THEN
    RAISE NOTICE 'vectorscale not available on this service (%). pgvector HNSW still works.', SQLERRM;
END $$;

DROP TABLE IF EXISTS ghw_vector_events;

-- Time-series + semantic embeddings in one hypertable (hybrid RAG pattern)
CREATE TABLE ghw_vector_events (
  time        TIMESTAMPTZ NOT NULL,
  sensor_id   TEXT        NOT NULL,
  location    TEXT        NOT NULL,
  event_type  TEXT        NOT NULL,
  note        TEXT        NOT NULL,
  metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
  embedding   halfvec(384) NOT NULL
) WITH (
  tsdb.hypertable,
  tsdb.segmentby = 'sensor_id',
  tsdb.orderby = 'time DESC'
);

CREATE INDEX ghw_vector_events_location_idx
  ON ghw_vector_events (location);

CREATE INDEX ghw_vector_events_event_type_idx
  ON ghw_vector_events (event_type);

-- Build after bulk load (see seed_embeddings.py)
-- CREATE INDEX ghw_vector_events_embedding_hnsw_idx
--   ON ghw_vector_events
--   USING hnsw (embedding halfvec_cosine_ops)
--   WITH (m = 16, ef_construction = 64);
