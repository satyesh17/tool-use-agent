"""Tool-use agent in pure Python — Groq + native function calling.

This is the simplest possible agent that uses tools:
- Conversation accumulates as a list of messages
- LLM decides when to use a tool via native function-calling
- Tools execute, results feed back into the conversation
- Loop until the LLM produces a final text answer

No framework. No magic. Just a loop.
"""

import json
import os

from groq import Groq

from .tools.calculator import calculator
from .tools.weather import get_weather
from .tools.search import search_web


# Tool registry — name string → callable function
TOOL_REGISTRY = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_web": search_web,
}


# Tool specs — what the LLM sees describing each tool
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a math expression and return the numeric result. "
                "Supports +, -, *, /, **, %, parentheses, and unary minus. "
                "Use for any arithmetic computation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '3 + 4 * 2'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get the current weather for a city. Returns temperature, conditions, "
                "humidity, and wind speed. Use for questions about current weather."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo' or 'Paris'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current information. Returns a few top results "
                "with title, URL, and content snippet. Use when you need up-to-date "
                "facts, recent events, or any information not in your training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

You have three tools:
- calculator: for math
- get_weather: for current weather
- search_web: for current information

Use tools when they help answer the user's question. Otherwise, answer directly.
After using tools, synthesize the results into a clear, friendly answer."""


class ToolUseAgent:
    """Tool-use agent with native function-calling."""
    
    MAX_ITERATIONS = 10
    
    def __init__(self, model: str = "llama-3.1-8b-instant", verbose: bool = True):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.verbose = verbose
    
    def run(self, user_message: str) -> str:
        """Run the agent on a user query. Returns the final text answer."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        for iteration in range(self.MAX_ITERATIONS):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Ask the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS_SPEC,
                tool_choice="auto",  # let the LLM decide whether to use tools
            )
            
            msg = response.choices[0].message
            
            # Append the assistant message to the conversation
            # (must include tool_calls field if present, for proper threading)
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ],
            })
            
            # No tool calls → final answer, we're done
            if not msg.tool_calls:
                if self.verbose:
                    print(f"  Final answer: {msg.content}")
                return msg.content
            
            # Execute each tool call
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                if self.verbose:
                    print(f"  → Tool call: {tool_name}({tool_args})")
                
                if tool_name not in TOOL_REGISTRY:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    try:
                        result = TOOL_REGISTRY[tool_name](**tool_args)
                    except Exception as e:
                        result = f"Tool execution error: {type(e).__name__}: {e}"
                
                if self.verbose:
                    print(f"  ← Result: {result[:150]}")
                
                # Append the tool result to the conversation, threaded by tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        
        raise RuntimeError(f"Agent did not finish after {self.MAX_ITERATIONS} iterations")