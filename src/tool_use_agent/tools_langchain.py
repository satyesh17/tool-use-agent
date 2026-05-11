"""LangChain-wrapped versions of the same tools — same logic, framework-friendly.

The original tools in src/tool_use_agent/tools/ stay framework-agnostic.
This module adapts them for LangGraph's ToolNode by adding the @tool decorator.

Why this layer exists:
- Keeps tools/ portable (works without LangChain installed)
- Lets multiple frameworks adapt the same tools (LangGraph, AutoGen, etc.)
- Isolates LangChain-specific concerns
"""

from langchain_core.tools import tool

from .tools.calculator import calculator as raw_calculator
from .tools.weather import get_weather as raw_get_weather
from .tools.search import search_web as raw_search_web


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the numeric result.
    
    Supports +, -, *, /, **, %, parentheses, and unary minus.
    Use for any arithmetic computation. Pass literal numeric values,
    not variable references.
    
    Args:
        expression: Math expression to evaluate, e.g. '3 + 4 * 2' or '(10/5)**3'
    """
    return raw_calculator(expression)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Returns temperature in Celsius, weather conditions, humidity percentage,
    and wind speed in km/h. Use this for any question about current weather.
    
    Args:
        city: City name. Use the most specific name available, e.g. 'Tokyo',
              'Paris, France', or 'Bengaluru' (instead of ambiguous 'Bangalore')
    """
    return raw_get_weather(city)


@tool
def search_web(query: str) -> str:
    """Search the web for current information.
    
    Returns up to 3 results with title, URL, and content snippet. Use this
    when you need:
    - Up-to-date facts (news, prices, current events)
    - Information about specific people, places, or products
    - Anything that may have changed after the training cutoff
    
    Args:
        query: The search query. Be specific — short queries return better results.
    """
    return raw_search_web(query)