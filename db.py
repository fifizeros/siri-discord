import os
import logging
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")
        self.client: Client = create_client(url, key)

    def save_message(self, message_id: str, channel_id: str, user_id: str, username: str, content: str):
        """Saves a message to the chat_history table and prunes messages older than 30 days."""
        try:
            data = {
                "message_id": message_id,
                "channel_id": channel_id,
                "user_id": user_id,
                "username": username,
                "content": content
            }
            self.client.table("chat_history").insert(data).execute()
            
            # Auto-prune messages older than 30 days to prevent storage bloat
            try:
                import datetime
                threshold = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)).isoformat()
                self.client.table("chat_history").delete().lt("created_at", threshold).execute()
            except Exception as prune_err:
                logger.error(f"Error pruning old chat history: {prune_err}")
                
        except Exception as e:
            logger.error(f"Error saving message {message_id}: {e}")

    def delete_user_fact(self, user_id: str, fact: str):
        """Deletes a specific learned fact about a user."""
        try:
            self.client.table("user_facts").delete().eq("user_id", user_id).eq("fact", fact).execute()
        except Exception as e:
            logger.error(f"Error deleting fact for user {user_id}: {e}")

    def delete_all_user_facts(self, user_id: str):
        """Deletes all facts/memories stored for a specific user."""
        try:
            self.client.table("user_facts").delete().eq("user_id", user_id).execute()
        except Exception as e:
            logger.error(f"Error deleting all facts for user {user_id}: {e}")

    def reset_channel_history(self, channel_id: str):
        """Deletes all chat history and associated embeddings for a channel."""
        try:
            self.client.table("chat_history").delete().eq("channel_id", channel_id).execute()
        except Exception as e:
            logger.error(f"Error resetting history for channel {channel_id}: {e}")


    def get_recent_history(self, channel_id: str, limit: int = 50):
        """Fetches recent message history for a channel, ordered chronologically."""
        try:
            # We sort by message_id because Snowflake IDs are time-ordered.
            response = self.client.table("chat_history") \
                .select("username, content") \
                .eq("channel_id", channel_id) \
                .order("message_id", desc=True) \
                .limit(limit) \
                .execute()
            
            # Since we fetched descending (latest first) for limiting, we reverse it to chronological order.
            messages = response.data or []
            messages.reverse()
            return messages
        except Exception as e:
            logger.error(f"Error fetching history for channel {channel_id}: {e}")
            return []

    def save_embedding(self, message_id: str, content: str, embedding: list[float]):
        """Saves a vector embedding for a specific message."""
        try:
            data = {
                "message_id": message_id,
                "content": content,
                "embedding": embedding
            }
            self.client.table("message_embeddings").insert(data).execute()
        except Exception as e:
            logger.error(f"Error saving embedding for message {message_id}: {e}")

    def search_semantic_history(self, channel_id: str, query_embedding: list[float], limit: int = 5, threshold: float = 0.5):
        """Performs a semantic similarity search using pgvector via Supabase RPC."""
        try:
            # Call the custom RPC function defined in our schema.sql
            params = {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit,
                "channel_id_filter": channel_id
            }
            response = self.client.rpc("match_message_embeddings", params).execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error performing semantic search in channel {channel_id}: {e}")
            return []

    def save_user_fact(self, user_id: str, fact: str):
        """Saves a learned fact about a user."""
        try:
            data = {
                "user_id": user_id,
                "fact": fact
            }
            # upsert or ignore duplicates
            self.client.table("user_facts").upsert(data, on_conflict="user_id, fact").execute()
        except Exception as e:
            logger.error(f"Error saving fact for user {user_id}: {e}")

    def get_user_facts(self, user_id: str):
        """Fetches all facts/memories stored for a specific user."""
        try:
            response = self.client.table("user_facts") \
                .select("fact") \
                .eq("user_id", user_id) \
                .execute()
            return [row["fact"] for row in response.data or []]
        except Exception as e:
            logger.error(f"Error fetching facts for user {user_id}: {e}")
            return []
