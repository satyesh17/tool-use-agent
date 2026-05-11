"""Web search tool — uses Tavily for LLM-friendly search results."""

import os

from tavily import TavilyClient




# Initialize client once at module load (singleton pattern — avoids re-auth on every call)
_client = None


def _get_client() -> TavilyClient:
    """Lazy-initialize the Tavily client. Reads TAVILY_API_KEY from environment."""
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment")
        _client = TavilyClient(api_key=api_key)
    return _client


def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for information.
    
    Args:
        query: The search query.
        max_results: Maximum number of results to return (default 3).
    
    Returns:
        A formatted string with top results, or an error message.
    """
    try:
        client = _get_client()
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # "advanced" is slower but more thorough
        )
        
        results = response.get("results", [])
        if not results:
            return f"No results found for query: '{query}'"
        
        # Format as readable text the LLM can synthesize
        formatted = []
        for i, r in enumerate(results, start=1):
            formatted.append(
                f"Result {i}: {r.get('title', 'untitled')}\n"
                f"  URL: {r.get('url', '?')}\n"
                f"  Content: {r.get('content', '')[:300]}"
            )
        
        return "\n\n".join(formatted)
        
    except Exception as e:
        return f"Search error: {type(e).__name__}: {e}"