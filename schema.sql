-- Enable the pgvector extension to allow storing and searching vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Chat History Table: Stores messages for short-term context.
-- We use BIGINT for IDs because Discord snowflake IDs are 64-bit integers.
CREATE TABLE IF NOT EXISTS chat_history (
    message_id BIGINT PRIMARY KEY,
    channel_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    username VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for retrieving recent channel history quickly
CREATE INDEX IF NOT EXISTS idx_chat_history_channel_msg ON chat_history (channel_id, message_id DESC);

-- 2. Message Embeddings Table: Stores 768-dimensional embeddings for semantic search.
-- We use text-embedding-004 which outputs 768-dimensional vectors.
CREATE TABLE IF NOT EXISTS message_embeddings (
    id SERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES chat_history(message_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL
);

-- Index for fast cosine similarity search (using IVFFlat or HNSW index on pgvector)
-- We will use a standard vector index (cosine distance) for similarity search
CREATE INDEX IF NOT EXISTS idx_message_embeddings_vector ON message_embeddings USING hnsw (embedding vector_cosine_ops);

-- 3. User Facts Table: Stores long-term memories/facts about users.
CREATE TABLE IF NOT EXISTS user_facts (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    fact TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, fact)
);

-- Index to quickly look up facts by user ID
CREATE INDEX IF NOT EXISTS idx_user_facts_user_id ON user_facts (user_id);

-- 4. Database Function for Semantic Similarity Search (RPC)
-- This function computes the cosine similarity and filters by channel ID.
CREATE OR REPLACE FUNCTION match_message_embeddings(
  query_embedding VECTOR(768),
  match_threshold FLOAT,
  match_count INT,
  channel_id_filter BIGINT
)
RETURNS TABLE (
  id INT,
  message_id BIGINT,
  content TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    me.id,
    me.message_id,
    me.content,
    1 - (me.embedding <=> query_embedding) AS similarity
  FROM message_embeddings me
  JOIN chat_history ch ON me.message_id = ch.message_id
  WHERE ch.channel_id = channel_id_filter
    AND 1 - (me.embedding <=> query_embedding) > match_threshold
  ORDER BY me.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- 5. Grant permissions to service_role to ensure our backend bot can access the tables and functions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;

