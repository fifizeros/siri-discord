import os
import logging
from google import genai
from google.genai import types
from tavily import TavilyClient

logger = logging.getLogger(__name__)

class AIManager:
    def __init__(self):
        gemini_key = os.getenv("GEMINI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY must be set in environment variables.")
        
        self.ai = genai.Client(api_key=gemini_key)
        self.tavily = TavilyClient(api_key=tavily_key) if tavily_key else None
        
        # Primary models as requested
        self.chat_model = "gemma-4-31b-it"
        self.embedding_model = "gemini-embedding-001"

    def get_embedding(self, text: str) -> list[float]:
        """Generates 768-dimensional vector embedding for the input text."""
        try:
            response = self.ai.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=768
                )
            )
            # Response contains a list of embeddings. We extract the first one.
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def perform_search(self, query: str) -> str:
        """Searches the web using Tavily API and returns a formatted summary of findings."""
        if not self.tavily:
            return "Web search is currently disabled (TAVILY_API_KEY is not configured)."
        
        try:
            logger.info(f"Performing Tavily web search for: '{query}'")
            # Get simple search results with basic depth
            response = self.tavily.search(query=query, search_depth="basic", max_results=5)
            results = response.get("results", [])
            answer = response.get("answer")
            
            if not results and not answer:
                return "No search results found."
            
            formatted_results = []
            if answer:
                formatted_results.append(f"### Direct Answer:\n{answer}\n")
                
            if results:
                formatted_results.append("### Search Results:")
                for i, res in enumerate(results, 1):
                    title = res.get("title", "No Title")
                    url = res.get("url", "")
                    content = res.get("content", "")
                    formatted_results.append(f"[{i}] {title}\nURL: {url}\nContent: {content}\n")
                    
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return f"Error executing search: {str(e)}"

    def generate_reply(self, system_instruction: str, contents: list, tools: list = None) -> any:
        """Generates text reply or function calls from Gemma/Gemini based on system instruction, contents, and tools."""
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
            if tools:
                config.tools = tools
                
            response = self.ai.models.generate_content(
                model=self.chat_model,
                contents=contents,
                config=config
            )
            return response
        except Exception as e:
            logger.error(f"Error generating LLM reply: {e}")
            return None

