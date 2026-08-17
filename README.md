# Agentic Loop — Goal Decomposition

Breaks a high-level goal into concrete, actionable steps through an iterative agentic loop.

## LLM

Meta Llama 3.1 8B Instruct via [OpenRouter](https://openrouter.ai)

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenRouter API key:

```
OPENROUTER_API_KEY=<your_key>
```

## Run

```bash
python main.py
```

Enter any goal when prompted. The agent will decompose it into actionable steps across 2–3 iterations.

## Project Structure

```
agent/
  loop.py           # perceive / reason / act / reflect
  tools.py          # break_goal and refine_steps tool handlers
  prompts.py        # prompt templates
  memory_manager.py # in-memory store with keyword retrieval
  harness.py        # retry, fallback, token budget
  logger.py         # structured JSON step logger
main.py             # entry point
config.yaml         # runtime configuration
```

## Configuration

All parameters are in `config.yaml` — model, max iterations, retry settings, token budget. No hardcoded values in the core loop.
