# tools_config.py
from google.genai import types

"""
This module holds the manual Tool Declarations exposed to the Gemini API.
Manually defining schemas using types.FunctionDeclaration is 100% robust and ensures
that Gemma 4 and Gemini Flash models correctly trigger function calling.
"""

# 1. tavily_search
tavily_search_decl = types.FunctionDeclaration(
    name="tavily_search",
    description="Searches the web for real-time facts, news, or weather. Allows selecting basic/advanced depth, news topic, and time ranges.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "query": types.Schema(
                type="STRING",
                description="The search query to look up on the web."
            ),
            "search_depth": types.Schema(
                type="STRING",
                enum=["basic", "advanced"],
                description="The depth of search. Use 'advanced' for complex queries requiring deep information and 'basic' for quick facts."
            ),
            "topic": types.Schema(
                type="STRING",
                enum=["general", "news"],
                description="Search category. Use 'news' for current events and recent articles."
            ),
            "time_range": types.Schema(
                type="STRING",
                enum=["day", "week", "month", "year"],
                description="Optionally filter news/search results to a specific timeframe."
            )
        },
        required=["query"]
    )
)

# 2. tavily_extract
tavily_extract_decl = types.FunctionDeclaration(
    name="tavily_extract",
    description="Extracts clean markdown content from specific web page URLs to read full article/document content.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "urls": types.Schema(
                type="ARRAY",
                items=types.Schema(type="STRING"),
                description="A list of specific web page URLs to extract text from."
            )
        },
        required=["urls"]
    )
)

# 3. tavily_crawl
tavily_crawl_decl = types.FunctionDeclaration(
    name="tavily_crawl",
    description="Crawls a website starting from a root URL and extracts content from multiple subpages.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "url": types.Schema(
                type="STRING",
                description="The root URL of the website to crawl."
            ),
            "limit": types.Schema(
                type="INTEGER",
                description="Maximum number of subpages to crawl (default is 3)."
            )
        },
        required=["url"]
    )
)

# 4. tavily_research
tavily_research_decl = types.FunctionDeclaration(
    name="tavily_research",
    description="Runs an autonomous multi-step deep research task and returns a comprehensive report. Use for complex research questions.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "query": types.Schema(
                type="STRING",
                description="The complex research query to investigate."
            ),
            "model": types.Schema(
                type="STRING",
                enum=["mini", "pro"],
                description="The research model. Use 'pro' for comprehensive analysis and 'mini' for faster research."
            )
        },
        required=["query"]
    )
)

# 4. add_reaction
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

# 5. pin_message
pin_message_decl = types.FunctionDeclaration(
    name="pin_message",
    description="Pins the user's current message to the channel.",
    parameters=types.Schema(
        type="OBJECT",
        properties={}
    )
)

# 6. create_thread
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

# 7. send_dm
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

# 8. disconnect_voice
disconnect_voice_decl = types.FunctionDeclaration(
    name="disconnect_voice",
    description="Disconnects/kicks a member from their current voice channel. Requires the bot to have 'Move Members' permission.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "user_id": types.Schema(
                type="STRING",
                description="Optional Discord User ID of the member to disconnect. If omitted, it will disconnect the user who sent the command."
            )
        }
    )
)

# Export the list of tools wrapped in types.Tool for the Gemini client config
BOT_TOOLS = [
    types.Tool(
        function_declarations=[
            tavily_search_decl,
            tavily_extract_decl,
            tavily_crawl_decl,
            tavily_research_decl,
            add_reaction_decl,
            pin_message_decl,
            create_thread_decl,
            send_dm_decl,
            disconnect_voice_decl
        ]
    )
]

