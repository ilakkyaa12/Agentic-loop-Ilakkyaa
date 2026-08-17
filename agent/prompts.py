# Prompt templates for each step of the agentic loop.
# Each function returns a ready-to-send string.

def reason_prompt(observation: dict, memory: list) -> str:
    """
    Tells the LLM what the current state is and asks it to pick an action.
    Returns a JSON plan: {action, parameters, reasoning}.
    """
    return f"""You are the reasoning engine of a goal-decomposition agent.
Your job: decide the SINGLE best next action given the current state.

=== CURRENT OBSERVATION ===
Goal       : {observation.get("goal")}
Current steps: {observation.get("result", {}).get("steps", []) if isinstance(observation.get("result"), dict) else "none"}
Reflection : {observation.get("reflection", "none")}

=== PAST MEMORY (most relevant) ===
{memory if memory else "No prior memory."}

=== AVAILABLE ACTIONS ===
1. break_goal   – Split the goal into 3-5 high-level steps.
                  Use when no steps exist yet.
2. refine_steps – Break existing broad steps into smaller, actionable tasks
                  (aim for 6+ total). Use when steps are too vague.
                  You MUST include the current steps list in parameters.
3. finish       – The goal is fully decomposed into clear, actionable steps.

=== OUTPUT FORMAT (strict JSON, no markdown) ===
For break_goal:
{{"action": "break_goal", "parameters": {{"goal": "<the goal>"}}, "reasoning": "<why>"}}

For refine_steps (copy the current steps list into parameters):
{{"action": "refine_steps", "parameters": {{"steps": ["<step1>", "<step2>", ...]}}, "reasoning": "<why>"}}

For finish:
{{"action": "finish", "parameters": {{}}, "reasoning": "<why>"}}"""


def break_goal_prompt(goal: str) -> str:
    """
    Asks the LLM to split a goal into 3-5 high-level steps.
    """
    return f"""Break the following goal into 3 to 5 clear, high-level steps.
Each step should be a short, actionable phrase.

Goal: {goal}

Return ONLY a JSON array of strings. Example:
["Step one", "Step two", "Step three"]"""


def refine_steps_prompt(steps: list) -> str:
    """
    Asks the LLM to expand broad steps into smaller sub-tasks.
    """
    steps_text = "\n".join(f"- {s}" for s in steps)
    return f"""The following steps are too broad. Break each one into smaller,
concrete sub-tasks. Return 6 or more total tasks.

Current steps:
{steps_text}

Return ONLY a JSON array of strings."""
