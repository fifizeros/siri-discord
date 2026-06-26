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

# Active tasks tracking for steering/cancellation
# Key: f"{channel_id}:{user_id}", Value: asyncio.Task
active_tasks = {}

# Caching for tool execution results to prevent redundant API calls
# Key: f"{channel_id}:{user_id}", Value: dict (containing queries and results)
tool_cache = {}


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

async def consolidate_memory(user_id_str: str, new_fact: str) -> list[str]:
    """Asks Gemini to check if the new fact contradicts/overrides any existing facts, and deletes them."""
    existing_facts = await asyncio.to_thread(db.get_user_facts, user_id_str)
    if not existing_facts:
        return []
        
    facts_str = "\n".join([f"- {f}" for f in existing_facts])
    prompt = (
        "คุณคือระบบผู้ช่วยจัดการความทรงจำ หน้าที่ของคุณคือเปรียบเทียบข้อมูลเก่ากับข้อมูลใหม่เพื่อหาข้อมูลที่ขัดแย้ง ซ้ำซ้อน หรือล้าสมัย\n\n"
        f"นี่คือรายการข้อมูลเก่าที่คุณจำเกี่ยวกับผู้ใช้คนนี้:\n{facts_str}\n\n"
        f"นี่คือข้อมูลใหม่ที่ผู้ใช้เพิ่งบอก:\n- {new_fact}\n\n"
        "คำแนะนำ:\n"
        "1. หาข้อความเก่าที่มีเนื้อหาขัดแย้ง ซ้ำซ้อน หรือถูกทดแทนด้วยข้อมูลใหม่นี้โดยสิ้นเชิง\n"
        "2. หากมีข้อมูลเก่าที่ควรถูกลบออก ให้ลิสต์ข้อความเก่าเหล่านั้นออกมาตรงๆ บรรทัดละข้อ โดยต้องตรงกับข้อความเก่าทุกตัวอักษร ห้ามสรุปความ ห้ามใส่ตัวเลขหรือสัญลักษณ์นำหน้า\n"
        "3. หากไม่มีข้อความเก่าใดที่ขัดแย้งหรือซ้ำซ้อนเลย ให้ตอบคำว่า 'None' เท่านั้น\n"
        "ตอบ:"
    )
    
    response = await asyncio.to_thread(
        ai.generate_reply,
        system_instruction="คุณคือระบบวิเคราะห์เปรียบเทียบข้อมูล ตอบเฉพาะข้อความที่ต้องการลบตรงๆ หรือ 'None' เท่านั้น ห้ามตอบนอกเหนือจากนี้",
        contents=[prompt]
    )
    
    deleted_facts = []
    if response and response.text:
        lines = [line.strip().strip("-*•").strip() for line in response.text.split("\n") if line.strip()]
        for line in lines:
            if line == "None" or not line:
                continue
            for old_fact in existing_facts:
                if old_fact.lower().strip() == line.lower().strip() or line.lower().strip() in old_fact.lower().strip():
                    await asyncio.to_thread(db.delete_user_fact, user_id_str, old_fact)
                    deleted_facts.append(old_fact)
    return deleted_facts

async def match_fact_to_forget(user_id_str: str, text_to_forget: str) -> str:
    """Asks Gemini to identify which of the existing facts matches the user's forget request."""
    existing_facts = await asyncio.to_thread(db.get_user_facts, user_id_str)
    if not existing_facts:
        return None
        
    facts_str = "\n".join([f"- {f}" for f in existing_facts])
    prompt = (
        "คุณคือระบบผู้ช่วยค้นหาข้อความที่จะลบ\n\n"
        f"นี่คือรายการข้อความที่คุณจำเกี่ยวกับผู้ใช้:\n{facts_str}\n\n"
        f"ผู้ใช้ต้องการลบความทรงจำเรื่อง: '{text_to_forget}'\n\n"
        "โปรดเลือกข้อความเก่าจากรายการข้างต้นที่ตรงหรือใกล้เคียงที่สุดกับความต้องการที่จะลบของผู้ใช้\n"
        "ตอบข้อความเก่านั้นตรงๆ ทุกตัวอักษร ห้ามสรุปความ ห้ามใส่ตัวเลข หากไม่พบข้อความใดที่ตรงหรือใกล้เคียงเลย ให้ตอบคำว่า 'None' เท่านั้น\n"
        "ตอบ:"
    )
    
    response = await asyncio.to_thread(
        ai.generate_reply,
        system_instruction="ตอบเฉพาะข้อความเก่าจากรายการที่เลือก หรือ 'None' เท่านั้น ห้ามตอบอธิบายเพิ่มเด็ดขาด",
        contents=[prompt]
    )
    
    if response and response.text:
        ans = response.text.strip().strip("-*•").strip()
        if ans != "None" and ans:
            for old_fact in existing_facts:
                if old_fact.lower().strip() == ans.lower().strip() or ans.lower().strip() in old_fact.lower().strip():
                    return old_fact
    return None

def parse_trigger(content: str) -> tuple[bool, str]:
    """Checks if message starts with triggers like 'สิริ', 'siri', 'ai' (with or without '!', case-insensitive, with/without space).
    Returns (has_trigger, cleaned_query).
    """
    content_stripped = content.strip()
    content_lower = content_stripped.lower()
    
    # 1. Check Thai trigger "สิริ" and its prefixed version "!สิริ"
    for pref in ("!สิริ", "สิริ"):
        if content_lower.startswith(pref):
            return True, content_stripped[len(pref):].strip()
            
    # 2. Check English triggers
    for pref in ("!siri", "siri", "!ai", "ai"):
        if content_lower == pref:
            return True, ""
        if content_lower.startswith(pref + " ") or content_lower.startswith(pref + "\n"):
            return True, content_stripped[len(pref):].strip()
        
        # Check if starts with prefix and the next char is non-alphanumeric or non-ASCII (e.g. Thai character or punctuation)
        if content_lower.startswith(pref):
            next_char = content_lower[len(pref):len(pref)+1]
            if next_char and not (next_char.isalnum() and next_char.isascii()):
                return True, content_stripped[len(pref):].strip()
                
    return False, ""

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
    has_trigger, _ = parse_trigger(message.content)
    if has_trigger:
        return True

    # 6. Respond directly to memory/reset commands
    content_stripped = message.content.strip()
    if content_stripped.startswith(("!remember ", "!forget ", "!forgetall", "!reset")):
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

    user_id_str = str(message.author.id)
    channel_id_str = str(message.channel.id)
    task_key = f"{channel_id_str}:{user_id_str}"

    # Cancel previous active task if it exists for this user in this channel (Interactive Steering)
    if task_key in active_tasks:
        old_task = active_tasks[task_key]
        if not old_task.done():
            logger.info(f"Cancelling active task for user {user_id_str} in channel {channel_id_str} due to steering/interruption.")
            old_task.cancel()

    # Define the core processing coroutine
    async def process_and_reply():
        try:
            user_query = message.content.strip()
            
            # Clean up the command prefix using parse_trigger
            has_trigger, cmd_query = parse_trigger(user_query)
            if not has_trigger:
                cmd_query = user_query
            else:
                # Match command routing by adding "!" back if it starts with memory commands
                if cmd_query.startswith(("remember ", "forget ", "forgetall", "reset")):
                    cmd_query = "!" + cmd_query

            # Intercept commands
            if cmd_query.startswith("!remember ") and db:
                fact_to_save = cmd_query[10:].strip()
                if fact_to_save:
                    # Run consolidation check
                    deleted = await consolidate_memory(user_id_str, fact_to_save)
                    await asyncio.to_thread(db.save_user_fact, user_id_str, fact_to_save)
                    if deleted:
                        deleted_list = ", ".join([f'"{d}"' for d in deleted])
                        await message.reply(f"บันทึกความทรงจำใหม่: \"{fact_to_save}\" เรียบร้อยแล้วครับ! (และลบความจำเดิมที่ขัดแย้งออก: {deleted_list}) 🧠✨")
                    else:
                        await message.reply(f"บันทึกความทรงจำแล้ว: \"{fact_to_save}\" 🧠✨")
                else:
                    await message.reply("โปรดระบุข้อความหลังคำสั่ง !remember เช่น `!remember ผมชอบกินกาแฟดำ`")
                return

            elif cmd_query.startswith("!forget ") and db:
                text_to_forget = cmd_query[8:].strip()
                if text_to_forget:
                    matched_fact = await match_fact_to_forget(user_id_str, text_to_forget)
                    if matched_fact:
                        await asyncio.to_thread(db.delete_user_fact, user_id_str, matched_fact)
                        await message.reply(f"ลบความทรงจำเรื่อง: \"{matched_fact}\" เรียบร้อยแล้วครับ! 🧹")
                    else:
                        await message.reply(f"ไม่พบความทรงจำที่ใกล้เคียงกับ \"{text_to_forget}\" ในระบบครับ")
                else:
                    await message.reply("โปรดระบุสิ่งที่ต้องการให้ลืมหลังคำสั่ง !forget เช่น `!forget เรื่องชอบกาแฟดำ`")
                return

            elif cmd_query == "!forgetall" and db:
                await asyncio.to_thread(db.delete_all_user_facts, user_id_str)
                await message.reply("ลบความทรงจำทั้งหมดเกี่ยวกับคุณเรียบร้อยแล้วครับ! 🧹✨")
                return

            elif cmd_query == "!reset" and db:
                await asyncio.to_thread(db.reset_channel_history, channel_id_str)
                await message.reply("ล้างประวัติการคุยและฐานความรู้ในช่องแชทนี้เรียบร้อยแล้วครับ! เริ่มต้นบทสนทนาใหม่ได้เลย 🧹✨")
                return

            # Trigger typing indicator since it is not a direct command
            async with message.channel.typing():
                # Step 4: Context Building
                # A. Retrieve Short-term context (recent 30 messages)
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

                # Tool Execution Handler with Caching logic to prevent redundant calls
                async def execute_tool(name: str, args: dict) -> str:
                    logger.info(f"Executing tool '{name}' with args: {args}")
                    
                    # Read caching for search & research tools
                    if name in ("tavily_search", "tavily_research"):
                        query = args.get("query")
                        if query:
                            cached = tool_cache.get(task_key)
                            if cached and cached.get("tool") == name and cached.get("query") == query:
                                logger.info(f"Steering optimization: Reusing cached tool result for '{name}' (query: '{query}')")
                                return cached.get("result")

                    if name == "tavily_search":
                        query = args.get("query")
                        if not query:
                            return "Error: Missing query argument."
                        search_depth = args.get("search_depth", "basic")
                        topic = args.get("topic", "general")
                        time_range = args.get("time_range")
                        res = await asyncio.to_thread(
                            ai.perform_search,
                            query=query,
                            search_depth=search_depth,
                            topic=topic,
                            time_range=time_range
                        )
                        # Save to cache
                        tool_cache[task_key] = {"tool": name, "query": query, "result": res}
                        return res

                    elif name == "tavily_extract":
                        urls = args.get("urls")
                        if not urls:
                            return "Error: Missing urls argument."
                        if isinstance(urls, str):
                            urls = [urls]
                        return await asyncio.to_thread(ai.perform_extract, urls=urls)

                    elif name == "tavily_crawl":
                        url = args.get("url")
                        if not url:
                            return "Error: Missing url argument."
                        limit = args.get("limit", 3)
                        return await asyncio.to_thread(ai.perform_crawl, url=url, limit=limit)

                    elif name == "tavily_research":
                        query = args.get("query")
                        if not query:
                            return "Error: Missing query argument."
                        model = args.get("model", "mini")
                        res = await asyncio.to_thread(ai.perform_research, query=query, model=model)
                        # Save to cache
                        tool_cache[task_key] = {"tool": name, "query": query, "result": res}
                        return res

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

                # Multi-turn Tool Calling Loop (up to 5 rounds)
                round_limit = 5
                current_round = 0
                
                while response and response.function_calls and current_round < round_limit:
                    current_round += 1
                    logger.info(f"Tool execution round {current_round}/{round_limit}")
                    
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
                    
                    # 4. Request next response/calls based on tool results
                    response = await asyncio.to_thread(ai.generate_reply, system_instruction, contents, BOT_TOOLS)
                
                if response and response.text:
                    reply = response.text
                else:
                    reply = "ดำเนินการตามคำสั่งของท่านเสร็จสิ้นแล้ว"

                # Discord message limit is 2000 characters
                if len(reply) > 2000:
                    for i in range(0, len(reply), 1950):
                        await message.reply(reply[i:i+1950])
                else:
                    await message.reply(reply)

                # Clear cache on successful generation completion
                tool_cache.pop(task_key, None)

        except asyncio.CancelledError:
            logger.info(f"Task for key {task_key} was cancelled due to steering/interruption.")
            # Keep tool_cache intact so the next steered task can reuse already completed search results
            raise
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.reply("ขออภัย เกิดข้อผิดพลาดในการประมวลผลข้อความของคุณ")
        finally:
            # Safely remove this task from active_tasks registry when finished
            if active_tasks.get(task_key) == asyncio.current_task():
                active_tasks.pop(task_key, None)

    # Register and start the asyncio Task
    new_task = asyncio.create_task(process_and_reply())
    active_tasks[task_key] = new_task


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
