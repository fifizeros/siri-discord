import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def print_status(service: str, status: str, detail: str = ""):
    icon = "[OK]" if status == "success" else "[FAIL]"
    detail_str = f" ({detail})" if detail else ""
    print(f"{icon} {service}: {status.upper()}{detail_str}")

def check_env():
    print("--- 1. Checking Environment Variables ---")
    env_file_exists = os.path.exists(".env")
    if not env_file_exists:
        print("[WARN] .env file not found. Creating a blank .env from .env.example...")
        try:
            with open(".env.example", "r") as src, open(".env", "w") as dst:
                dst.write(src.read())
            print("[OK] Created .env file. Please fill in your API keys in the .env file before running this again.")
        except Exception as e:
            print(f"[FAIL] Could not copy .env.example to .env: {e}")
        return False

    required_keys = ["DISCORD_TOKEN", "SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"]
    all_set = True
    for key in required_keys:
        val = os.getenv(key)
        if not val or "your_" in val:
            print(f"[FAIL] {key} is missing or placeholder in .env")
            all_set = False
        else:
            print(f"[OK] {key} is configured")
    return all_set

def check_supabase():
    print("\n--- 2. Checking Supabase Connection ---")
    try:
        from db import DatabaseManager
        db = DatabaseManager()
        # Try a direct query to verify if the tables exist in the schema
        db.client.table("user_facts").select("fact").limit(1).execute()
        print_status("Supabase Client", "success", "Connected & verified schema")
        return db
    except Exception as e:
        print_status("Supabase Client", "failed", f"Database schema is missing or invalid: {str(e)}")
        return None

def check_ai():
    print("\n--- 3. Checking Gemini & Tavily API ---")
    try:
        from ai import AIManager
        ai = AIManager()
        
        # Test 1: Embedding model
        print("Testing gemini-embedding-001...")
        emb = ai.get_embedding("Test connectivity")
        if emb and len(emb) == 768:
            print_status("Gemini Embedding API", "success", f"Generated 768-dim vector")
        else:
            print_status("Gemini Embedding API", "failed", "Invalid vector response")

        # Test 2: Chat model
        print("Testing gemma-4-31b-it...")
        response = ai.generate_reply(
            system_instruction="You are a helper. Answer in one word.",
            contents=["Say Hello"]
        )
        reply = response.text if response else ""
        print_status("Gemma Chat API", "success", f"Response: {reply.strip()}")

        # Test 3: Tavily Search
        if ai.tavily:
            print("Testing Tavily Search...")
            search_res = ai.perform_search("current weather in Bangkok")
            if "Error" not in search_res and "No search results" not in search_res:
                print_status("Tavily Search API", "success", "Search returns valid results")
            else:
                print_status("Tavily Search API", "failed", search_res)
        else:
            print_status("Tavily Search API", "skipped", "No key configured")

        return ai
    except Exception as e:
        print_status("AI Manager", "failed", str(e))
        return None

def main():
    print("=========================================")
    print("       Discord Bot Pipeline Checker      ")
    print("=========================================\n")
    
    if not check_env():
        print("\n[STOP] Please configure your .env file first!")
        sys.exit(1)
        
    db = check_supabase()
    ai = check_ai()
    
    print("\n=========================================")
    if db and ai:
        print("SUCCESS: All core integrations are OK!")
        print("You can now start your bot using: python bot.py")
    else:
        print("WARNING: Some checks failed. Please resolve the errors above.")
    print("=========================================")

if __name__ == "__main__":
    main()
