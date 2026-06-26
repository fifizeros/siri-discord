import os
import asyncio
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import discord
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DiscordBot")

# Import our managers, SDK types, prompts builder, and tools config
from google.genai import types
from db import DatabaseManager
from ai import AIManager
from prompts import build_system_instruction
from tools_config import BOT_TOOLS

# Initialize clients
try:
    db = DatabaseManager()
    ai = AIManager()
except Exception as e:
    logger.critical(f"Failed to initialize database or AI manager: {e}")
    db = None
    ai = None

# Initialize discord client with gateway intents
intents = discord.Intents.default()
intents.message_content = True  # Crucial: Allows bot to read message content
intents.guilds = True
intents.guild_messages = True
intents.dm_messages = True

client = discord.Client(intents=intents)

async def async_save_message(message: discord.Message):
    """Saves message metadata to Supabase asynchronously in a background thread."""
    if not db:
        return
    
    # Extract IDs as strings to prevent float precision loss
    msg_id = str(message.id)
    chan_id = str(message.channel.id)
    user_id = str(message.author.id)
    username = f"{message.author.name}#{message.author.discriminator}" if message.author.discriminator != "0" else message.author.name
    content = message.content
    
    # Save the message in a thread pool to avoid blocking the event loop
    await asyncio.to_thread(db.save_message, msg_id, chan_id, user_id, username, content)

async def async_embed_and_save(message_id: str, content: str):
    """Generates and saves the vector embedding in the background."""
    if not db or not ai or not content.strip():
        return
    try:
        # Only embed substantial messages to save API calls
        if len(content) < 5:
            return
            
        embedding = await asyncio.to_thread(ai.get_embedding, content)
        if embedding:
            await asyncio.to_thread(db.save_embedding, message_id, content, embedding)
            logger.info(f"Successfully saved embedding for message {message_id}")
    except Exception as e:
        logger.error(f"Failed to process embedding for message {message_id}: {e}")

def should_respond(message: discord.Message) -> bool:
    """Step 3: Should Respond? checks."""
    # 1. Never respond to self
    if message.author == client.user:
        return False
    
    # 2. Respond if it is a Direct Message (DM)
    if message.guild is None:
        return True
    
    # 3. Respond if the bot is mentioned directly
    if client.user in message.mentions:
        return True
    
    # 4. Respond if it is a reply to the bot's message
    if message.reference and message.reference.resolved:
        resolved_msg = message.reference.resolved
        if isinstance(resolved_msg, discord.Message) and resolved_msg.author == client.user:
            return True
            
    # 5. Respond if prefixed with a command keyword
    if message.content.startswith("!bot "):
        return True

    return False

@client.event
async def on_ready():
    logger.info(f"Bot logged in as {client.user} (ID: {client.user.id})")
    print(f"Bot connected: {client.user}")

@client.event
async def on_message(message: discord.Message):
    # Step 2: Persist message asynchronously (Non-blocking)
    if db:
        asyncio.create_task(async_save_message(message))
        # Embed the message for semantic search in the background (only if it's not the bot's own message)
        if message.author != client.user:
            asyncio.create_task(async_embed_and_save(str(message.id), message.content))

    # Step 3: Should Respond?
    if not should_respond(message):
        return

    # Trigger typing indicator to show the bot is thinking
    async with message.channel.typing():
        try:
            user_id_str = str(message.author.id)
            channel_id_str = str(message.channel.id)
            user_query = message.content
            
            # Step 4: Context Building
            # A. Retrieve Short-term context (recent 50 messages)
            recent_history = []
            if db:
                raw_history = await asyncio.to_thread(db.get_recent_history, channel_id_str, limit=30)
                recent_history = [f"{msg['username']}: {msg['content']}" for msg in raw_history]
            
            # B. Retrieve Long-term context (Semantic Search top 5)
            semantic_context = []
            if db and ai and len(user_query.strip()) > 5:
                query_emb = await asyncio.to_thread(ai.get_embedding, user_query)
                if query_emb:
                    raw_semantic = await asyncio.to_thread(db.search_semantic_history, channel_id_str, query_emb, limit=5)
                    semantic_context = [row['content'] for row in raw_semantic]

            # C. Retrieve User Memory / Facts
            user_facts = []
            if db:
                user_facts = await asyncio.to_thread(db.get_user_facts, user_id_str)

            # Step 6: Harness Assembly (Structured 8-layered prompt using prompts.py)
            system_instruction = build_system_instruction(
                user_facts=user_facts,
                semantic_context=semantic_context
            )

            # Assemble conversation history and user query
            history_text = "\n".join(recent_history)
            final_prompt = (
                f"Conversation history of the channel:\n{history_text}\n\n"
                f"User's current message: {user_query}\n"
                "Reply to the user directly and naturally."
            )

            # Initialize conversation history with user message
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=final_prompt)]
                )
            ]

            # Tool Execution Handler
            async def execute_tool(name: str, args: dict) -> str:
                logger.info(f"Executing tool '{name}' with args: {args}")
                if name == "web_search":
                    query = args.get("query")
                    if not query:
                        return "Error: Missing query argument."
                    return await asyncio.to_thread(ai.perform_search, query=query)


                elif name == "add_reaction":
                    emoji = args.get("emoji")
                    if not emoji:
                        return "Error: Missing emoji argument."
                    try:
                        await message.add_reaction(emoji)
                        return f"Successfully added reaction: {emoji}"
                    except Exception as e:
                        return f"Error adding reaction: {e}"
                        
                elif name == "pin_message":
                    try:
                        await message.pin()
                        return "Successfully pinned the message."
                    except Exception as e:
                        return f"Error pinning message: {e}"
                        
                elif name == "create_thread":
                    thread_name = args.get("name", "AI Conversation Thread")
                    try:
                        thread = await message.create_thread(name=thread_name)
                        return f"Successfully created thread: {thread.name}"
                    except Exception as e:
                        return f"Error creating thread: {e}"
                        
                elif name == "send_dm":
                    dm_content = args.get("content")
                    if not dm_content:
                        return "Error: Missing content argument."
                    try:
                        await message.author.send(dm_content)
                        return "Successfully sent private Direct Message to the user."
                    except Exception as e:
                        return f"Error sending DM (user may have DMs closed): {e}"
                        
                return f"Error: Unknown tool '{name}'."

            # Step 7: AI Execution
            response = await asyncio.to_thread(ai.generate_reply, system_instruction, contents, BOT_TOOLS)
            
            if not response:
                await message.reply("ขออภัย เกิดข้อผิดพลาดในการประมวลผลข้อความของคุณ")
                return

            # Check if Gemma decided to call one or more tools
            if response.function_calls:
                # 1. Append original assistant tool call content
                contents.append(response.candidates[0].content)
                
                # 2. Execute all tool calls asynchronously
                tool_parts = []
                for call in response.function_calls:
                    tool_result = await execute_tool(call.name, call.args)
                    part = types.Part.from_function_response(
                        name=call.name,
                        response={"result": tool_result}
                    )
                    tool_parts.append(part)
                
                # 3. Append tool responses to content history
                contents.append(types.Content(role="tool", parts=tool_parts))
                
                # 4. Request final response based on tool results
                final_response = await asyncio.to_thread(ai.generate_reply, system_instruction, contents, BOT_TOOLS)
                if final_response and final_response.text:
                    reply = final_response.text
                else:
                    reply = "ดำเนินการตามคำสั่งของท่านเสร็จสิ้นแล้ว"
            else:
                reply = response.text or "ขออภัย ฉันไม่สามารถหาข้อมูลเพื่อตอบคุณในขณะนี้ได้"

            # Discord message limit is 2000 characters
            if len(reply) > 2000:
                for i in range(0, len(reply), 1950):
                    await message.reply(reply[i:i+1950])
            else:
                await message.reply(reply)

            # Memory extraction shortcut helper
            if user_query.startswith("!remember ") and db:
                fact_to_save = user_query[10:].strip()
                if fact_to_save:
                    await asyncio.to_thread(db.save_user_fact, user_id_str, fact_to_save)
                    await message.reply(f"บันทึกความจำแล้ว: \"{fact_to_save}\"")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.reply("ขออภัย เกิดข้อผิดพลาดในการประมวลผลข้อความของคุณ")

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")
        
    def log_message(self, format, *args):
        # Suppress logging every GET request to prevent console spam
        return

def run_health_server():
    port = int(os.getenv("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    logger.info(f"Starting background health check server on port {port}")
    server.serve_forever()

if __name__ == "__main__":
    # Start background health server for Render port binding (prevents deployment timeout)
    threading.Thread(target=run_health_server, daemon=True).start()

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.critical("DISCORD_TOKEN is missing from environment variables!")
    else:
        client.run(token)
