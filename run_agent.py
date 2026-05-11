"""Demo runner for the pure-Python tool-use agent."""

from src.tool_use_agent.agent import ToolUseAgent


def main():
    agent = ToolUseAgent(verbose=True)
    
    queries = [
        # Single-tool: should call calculator
        "What is 73 * 14 plus 219?",
        
        # Single-tool: should call weather
        "What's the weather in Tokyo right now?",
        
        # Single-tool: should call search
        "What's the latest Claude model from Anthropic?",
        
        # Multi-tool: should call weather AND calculator
        "What's the temperature in Paris in Celsius? If you convert that to Fahrenheit, what's the value?",
        
        # No tools needed: should answer from training
        "What is the capital of India? Which Indian city has the worst traffic congestion, is it not Bangalore? What is the current weather of the city? What is the current population of that city and what is its forecasted population in 2047?",
    ]
    
    for q in queries:
        print(f"\n{'='*70}")
        print(f"USER: {q}")
        print('='*70)
        answer = agent.run(q)
        print(f"\nFINAL ANSWER: {answer}\n")


if __name__ == "__main__":
    main()