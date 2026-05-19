

"""Demo runner for the LangGraph tool-use agent.

Runs the same 5 queries as run_agent.py so we can compare outputs and
behavior side-by-side. The graph version should produce equivalent answers
(LLM non-determinism aside) — the architectural difference is in HOW the
agent is structured, not WHAT it can do.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from tool_use_agent.agent_langgraph import build_agent, run_query


def main():
    queries = [
        # Single-tool: should call calculator
        "What is 73 * 14 plus 219?",
        
        # Single-tool: should call weather
        "What's the weather in Tokyo right now?",
        
        # Single-tool: should call search
        "What's the latest Claude model from Anthropic?",
        
        # Multi-tool: should call weather AND calculator (self-correction expected)
        "What's the temperature in Paris in Celsius? If you convert that to Fahrenheit, what's the value?",
        
        # No tools needed: should answer from training
        "What is the capital of India? Which Indian city has the worst traffic congestion, is it not Bangalore? What is the current weather of the city? What is the current population of that city and what is its forecasted population in 2047?",
    ]
    
    for q in queries:
        print(f"\n{'='*70}")
        print(f"USER: {q}")
        print('='*70)
        answer = run_query(q, verbose=True)
        print(f"\nFINAL ANSWER: {answer}\n")
    
    # Print the graph structure once at the end as a portfolio artifact
    print("\n" + "="*70)
    print("GRAPH STRUCTURE (Mermaid — paste in a .md file to render)")
    print("="*70)
    agent = build_agent()
    print(agent.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
