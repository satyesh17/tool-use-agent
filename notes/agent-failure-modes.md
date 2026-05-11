# Agent Failure Modes Observed (Day 4)

Three production-class bugs discovered while stress-testing the pure-Python
agent with a real-world multi-part query.

## Test Query

"What is the capital of India? Which Indian city has the worst traffic 
congestion, is it not Bangalore? What is the current weather of the city? 
What is the current population of that city and what is its forecasted 
population in 2047?"

This is a real-world composite question with 4 sub-tasks, requiring 
parallel tool dispatch and cross-tool reasoning.

## Bug 1: Geocoding Disambiguation Failure

**What happened:** Agent called `get_weather('Bangalore')`. Open-Meteo's 
geocoding API returned "Bangalore Town, Pakistan" — a Karachi neighborhood —
instead of Bengaluru, India (population 8.4M).

**Root cause:** The geocoder was called with `count=1`, returning whichever 
result the API ranked first. The Pakistani neighborhood happened to outrank 
the Indian city in that query.

**Architectural class:** API contract user-hostility. What the LLM asked for 
("Bangalore") and what it meant ("the famous Indian tech hub") are different. 
Production agents must always disambiguate ambiguous name lookups.

**Mitigation:** Pass `count=5`, select the result with highest population. 
For city names, the most populous match is almost always the intended one.

## Bug 2: Calculator Receives Natural Language

**What happened:** Agent called `calculator('Bangalore population in 2023 + 
24 * Bangalore population in 2023')`. The AST parser correctly rejected this 
as an invalid expression.

**Root cause:** The LLM treated the calculator as if it shared variable 
context with previous tool calls. It used natural-language symbols 
("Bangalore population in 2023") instead of literal numbers.

**Architectural class:** Tool-context isolation surprise. Each tool runs 
independently — tools don't share variables. The LLM doesn't know this 
without being told explicitly.

**Mitigation:** Either (a) improve calculator's error message to teach the 
LLM about literal-value requirement, or (b) document tool-isolation in the 
tool description so the LLM doesn't try variable references.

**Bonus observation:** In Day 4 Query 4 (Paris/Fahrenheit), the same failure 
occurred and the LLM self-corrected within 1 retry. In this query, the LLM 
gave up after 1 failed call because other tool calls succeeded. 
*Error recovery is conditional on the model's perceived value of recovering.*

## Bug 3: Partial Hallucination + Graceful Abandonment

**What happened:** Agent's final answer included "approximately 14,772,000" 
as Bangalore's 2026 population — a number not visible in any tool result. 
For the 2047 forecast, the agent honestly said "specific numbers could not 
be found."

**Root cause:** When tool results were incomplete, the model partially 
hallucinated a current population from training data and acknowledged the 
forecast gap. Honest but inconsistent — same model both invented and 
disclaimed in adjacent sentences.

**Architectural class:** Mixed grounding. Production agents need explicit 
"only-from-tools" mode if hallucination matters. Otherwise the LLM 
opportunistically uses training data alongside tool results.

**Mitigation:** Stronger system prompt explicitly forbidding facts not 
derived from tool results: "If you do not have a tool result confirming a 
fact, say you don't know rather than inferring from training data."

## What This Teaches About Real-World Testing

Three failure modes from one query. Hand-picked test queries (Q1-Q5 from the 
demo) only hit expected code paths. Real-world queries with ambiguity, 
multi-step reasoning, and partial information surface bugs the demo set 
never would.

**Architectural rule:** Demo queries prove the agent runs. Adversarial 
queries prove the agent is robust. Both required for production readiness.
