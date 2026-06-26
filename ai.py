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
        self.chat_model = "gemini-3.1-flash-lite"
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

    def perform_search(self, query: str, search_depth: str = "basic", topic: str = "general", time_range: str = None) -> str:
        """Searches the web using Tavily API and returns a formatted summary of findings."""
        if not self.tavily:
            return "Web search is currently disabled (TAVILY_API_KEY is not configured)."
        
        try:
            logger.info(f"Performing Tavily web search for: '{query}' (depth: {search_depth}, topic: {topic}, time_range: {time_range})")
            response = self.tavily.search(
                query=query,
                search_depth=search_depth,
                topic=topic,
                time_range=time_range,
                max_results=5
            )
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

    def perform_extract(self, urls: list[str]) -> str:
        """Extracts cleaned markdown content from a list of web URLs using Tavily Extract."""
        if not self.tavily:
            return "Web extraction is currently disabled (TAVILY_API_KEY is not configured)."
        try:
            logger.info(f"Performing Tavily extract for URLs: {urls}")
            response = self.tavily.extract(urls=urls)
            results = response.get("results", [])
            failed = response.get("failed_results", [])
            
            formatted_results = []
            for res in results:
                url = res.get("url")
                title = res.get("title", "No Title")
                content = res.get("raw_content", "")
                formatted_results.append(f"### Extracted URL: {url}\nTitle: {title}\nContent:\n{content}\n")
            
            for f in failed:
                formatted_results.append(f"### Failed to extract: {f.get('url')} (Error: {f.get('error')})")
                
            if not formatted_results:
                return "No content could be extracted."
                
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error performing extract: {e}")
            return f"Error executing extract: {str(e)}"

    def perform_crawl(self, url: str, limit: int = 3) -> str:
        """Crawls a website and extracts content from multiple pages using Tavily Crawl."""
        if not self.tavily:
            return "Web crawling is currently disabled (TAVILY_API_KEY is not configured)."
        try:
            logger.info(f"Performing Tavily crawl for: {url} (limit: {limit})")
            response = self.tavily.crawl(url=url, max_depth=1, limit=limit)
            results = response.get("results", [])
            
            formatted_results = []
            for res in results:
                page_url = res.get("url")
                content = res.get("raw_content", "")
                formatted_results.append(f"### Crawled URL: {page_url}\nContent:\n{content}\n")
                
            if not formatted_results:
                return "No crawl results found."
                
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error performing crawl: {e}")
            return f"Error executing crawl: {str(e)}"

    def perform_research(self, query: str, model: str = "mini") -> str:
        """Runs a deep research task using Tavily Research API and polls for completion."""
        if not self.tavily:
            return "Web research is currently disabled (TAVILY_API_KEY is not configured)."
        try:
            logger.info(f"Starting Tavily research for: '{query}' (model: {model})")
            task = self.tavily.research(input=query, model=model)
            request_id = task.get("request_id")
            
            # Poll for results
            import time
            for _ in range(12): # Poll for up to 60 seconds (12 * 5 seconds)
                time.sleep(5)
                res = self.tavily.get_research(request_id)
                status = res.get("status")
                if status == "completed":
                    content = res.get("content", "")
                    return f"### Deep Research Report:\n{content}"
                elif status == "failed":
                    return "Error: Deep research task failed on the server."
            
            return f"Timeout: Research is still processing (Task ID: {request_id}). Please check again later."
        except Exception as e:
            logger.error(f"Error performing research: {e}")
            return f"Error executing research: {str(e)}"



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

