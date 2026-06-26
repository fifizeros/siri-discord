# tools_config.py
"""
This module holds the tool definitions (function declarations) exposed to the Gemini API.
The docstrings and type annotations here are automatically converted by the google-genai SDK into tool schemas.
"""

def web_search(query: str) -> str:
    """Searches the web for real-time information on a topic.
    
    Args:
        query: The search query.
    """
    return ""

def add_reaction(emoji: str) -> str:
    """Adds a reaction emoji to the user's current message.
    
    Args:
        emoji: The exact emoji character to react with (e.g. '👍', '🔥', '🎉').
    """
    return ""

def pin_message() -> str:
    """Pins the user's current message to the channel."""
    return ""

def create_thread(name: str) -> str:
    """Creates a new public thread conversation starting from the user's current message.
    
    Args:
        name: The name of the thread to create.
    """
    return ""

def send_dm(content: str) -> str:
    """Sends a private Direct Message (DM) to the user who sent the current message.
    
    Args:
        content: The text content of the private message.
    """
    return ""

# Export the list of tools for the Gemini client config
BOT_TOOLS = [web_search, add_reaction, pin_message, create_thread, send_dm]
