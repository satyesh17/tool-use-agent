"""Tool-use agent built with LangGraph.

Same Groq backend, same three tools, same conversation behavior as agent.py.
The difference is structural: instead of a hand-rolled while-loop, the agent
is modeled as a directed graph with state.

Architectural comparison:
- agent.py:       ~80 lines, sequential loop, manual message threading
- agent_langgraph.py: ~50 lines, graph state machine, framework handles threading

Frameworks rarely shrink code; they restructure it. This file is shorter
because LangGraph's ToolNode and tools_condition replace ~30 lines of manual
dispatch logic from the pure-Python version.
"""

import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from .tools_langchain import calculator, get_weather, search_web


SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

You have three tools:
- calculator: for math
- get_weather: for current weather
- search_web: for current information

Use tools when they help answer the user's question. Otherwise, answer directly.
After using tools, synthesize the results into a clear, friendly answer."""


def build_agent():
    """Build and compile the LangGraph agent.
    
    Returns a compiled graph ready to invoke with `.stream(...)` or `.invoke(...)`.
    """
    # 1. Configure the LLM with tools bound
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
    )
    tools = [calculator, get_weather, search_web]
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Define the "agent" node — what runs when we reach this step
    def agent_node(state: MessagesState):
        """Call the LLM with the current conversation. Returns the new message."""
        messages = state["messages"]
        
        # Prepend system prompt on first call (not yet in conversation)
        if not messages or messages[0].type != "system":
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # 3. Build the graph
    graph = StateGraph(MessagesState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        tools_condition,                          # returns "tools" if tool_calls present, else END
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")              # after tool execution, loop back to agent
    
    # 4. Compile to an executable graph
    return graph.compile()


def run_query(query: str, verbose: bool = True) -> str:
    """Run the agent on a single query. Returns the final text answer.
    
    Uses streaming so we can print each node's output as the graph executes.
    """
    agent = build_agent()
    
    if verbose:
        print(f"\n--- Running through LangGraph ---")
    
    final_state = None
    for chunk in agent.stream({"messages": [("user", query)]}, stream_mode="values"):
        final_state = chunk
        if not verbose:
            continue
        
        last_msg = chunk["messages"][-1]
        msg_type = last_msg.type
        
        if msg_type == "ai":
            if last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    print(f"  → Tool call: {tc['name']}({tc['args']})")
            elif last_msg.content:
                print(f"  Final answer: {last_msg.content}")
        elif msg_type == "tool":
            content_preview = last_msg.content[:150] if last_msg.content else ""
            print(f"  ← Tool result: {content_preview}")
    
    return final_state["messages"][-1].content