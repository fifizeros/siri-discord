# tools_config.py
from google.genai import types

"""
This module holds the manual Tool Declarations exposed to the Gemini API.
Manually defining schemas using types.FunctionDeclaration is 100% robust and ensures
that Gemma 4 and Gemini Flash models correctly trigger function calling.
"""

# 1. web_search
web_search_decl = types.FunctionDeclaration(
    name="web_search",
    description="Searches the web for real-time information on a topic, weather, news, or facts you don't know.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "query": types.Schema(
                type="STRING",
                description="The search query to look up on the web (e.g. 'weather in Chiang Mai')."
            )
        },
        required=["query"]
    )
)

# 2. add_reaction
add_reaction_decl = types.FunctionDeclaration(
    name="add_reaction",
    description="Adds a reaction emoji to the user's current message.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "emoji": types.Schema(
                type="STRING",
                description="The exact emoji character to react with (e.g. '👍', '🔥', '🎉')."
            )
        },
        required=["emoji"]
    )
)

# 3. pin_message
pin_message_decl = types.FunctionDeclaration(
    name="pin_message",
    description="Pins the user's current message to the channel.",
    parameters=types.Schema(
        type="OBJECT",
        properties={}
    )
)

# 4. create_thread
create_thread_decl = types.FunctionDeclaration(
    name="create_thread",
    description="Creates a new public thread conversation starting from the user's current message.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "name": types.Schema(
                type="STRING",
                description="The name of the thread to create."
            )
        },
        required=["name"]
    )
)

# 5. send_dm
send_dm_decl = types.FunctionDeclaration(
    name="send_dm",
    description="Sends a private Direct Message (DM) to the user who sent the current message.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "content": types.Schema(
                type="STRING",
                description="The text content of the private message."
            )
        },
        required=["content"]
    )
)

# Export the list of tools wrapped in types.Tool for the Gemini client config
BOT_TOOLS = [
    types.Tool(
        function_declarations=[
            web_search_decl,
            add_reaction_decl,
            pin_message_decl,
            create_thread_decl,
            send_dm_decl
        ]
    )
]
