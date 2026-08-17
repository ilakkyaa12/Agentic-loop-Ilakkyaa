import json
from agent.prompts import break_goal_prompt, refine_steps_prompt


def break_goal(parameters: dict, client, model: str) -> dict:
    goal = parameters["goal"]
    prompt = break_goal_prompt(goal)

    response = client.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()
    steps = json.loads(text)
    return {"steps": steps}


def refine_steps(parameters: dict, client, model: str) -> dict:
    steps = parameters["steps"]
    prompt = refine_steps_prompt(steps)

    response = client.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()
    refined = json.loads(text)
    return {"steps": refined}


TOOLS = {
    "break_goal": break_goal,
    "refine_steps": refine_steps,
}

TOOL_DEFINITIONS = {
    "break_goal": {
        "description": "Break a high-level goal into 3-5 actionable steps.",
        "parameters": {
            "goal": {"type": "string", "description": "The goal to decompose"}
        }
    },
    "refine_steps": {
        "description": "Expand broad steps into smaller, concrete sub-tasks.",
        "parameters": {
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of broad steps to refine"
            }
        }
    }
}
