import json
from agent.prompts import reason_prompt
from agent.memory_manager import MemoryManager


def perceive(input_data: str) -> dict:
    """
    Parse raw input into a structured observation dict.
    Extracts the goal and sets intent to 'break_goal'.
    """
    return {
        "goal": input_data,
        "intent": "break_goal",
        "constraints": [],
        "context": {},
        "result": None,
        "reflection": None,
    }


def reason(observation: dict, memory: list, client, model: str) -> dict:
    """
    Call the LLM to decide the next action.
    Returns a plan dict: {action, parameters, reasoning}.
    """
    prompt = reason_prompt(observation, memory)

    response = client.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()
    plan = json.loads(text)
    return plan


def act(plan: dict, tools: dict, client, model: str, last_steps: list = None) -> dict:
    action = plan["action"]
    parameters = plan.get("parameters", {})

    if action == "finish":
        return {"finished": True, "steps": parameters.get("steps", [])}

    # If LLM forgot to pass steps for refine_steps, use last known steps
    if action == "refine_steps" and not parameters.get("steps") and last_steps:
        parameters["steps"] = last_steps

    handler = tools[action]
    return handler(parameters, client, model)


def reflect(result: dict, observation: dict) -> dict:
    """
    Evaluate whether the goal decomposition is complete.
    Done when we have 6+ steps or the action was 'finish'.
    Returns {is_done, quality_score, next_instruction}.
    """
    if result.get("finished"):
        return {
            "is_done": True,
            "quality_score": 10,
            "next_instruction": "Goal fully decomposed.",
        }

    steps = result.get("steps", [])

    if len(steps) >= 6:
        return {
            "is_done": True,
            "quality_score": 9,
            "next_instruction": "Goal has been broken into enough actionable steps.",
        }

    return {
        "is_done": False,
        "quality_score": 5,
        "next_instruction": "Steps are still too broad. Refine them into smaller tasks.",
    }


def run_agent(input_data: str, tools: dict, client, model: str, max_iterations: int = 10) -> dict:
    """
    Main agentic loop: Perceive → Reason → Act → Reflect, repeated until done.

    Memory is read before each reason() call and written after each reflect() call.
    Returns the final result dict.
    """
    memory_manager = MemoryManager()
    observation = perceive(input_data)
    last_reflection = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*40}")
        print(f"Iteration {iteration}")
        print(f"{'='*40}")

        # --- MEMORY READ ---
        memory = memory_manager.retrieve(input_data)
        print(f"[Memory] {len(memory)} relevant entries recalled.")

        # --- REASON ---
        plan = reason(observation, memory, client, model)
        print(f"[Reason] Action: {plan['action']} | {plan.get('reasoning', '')}")

        # --- ACT ---
        result = act(plan, tools, client, model)
        print(f"[Act]    Steps so far: {len(result.get('steps', []))}")

        # --- REFLECT ---
        reflection = reflect(result, observation)
        print(f"[Reflect] Done: {reflection['is_done']} | Score: {reflection['quality_score']}")

        # --- MEMORY WRITE ---
        memory_manager.add({
            "iteration": iteration,
            "action": plan["action"],
            "steps_count": len(result.get("steps", [])),
            "reflection": reflection,
        })

        # --- STUCK DETECTION ---
        if last_reflection and last_reflection == reflection:
            print("[Guard] Identical reflection twice in a row. Stopping (STUCK).")
            result["status"] = "STUCK"
            return result

        last_reflection = reflection

        if reflection["is_done"]:
            print("\n✓ Goal fully decomposed.")
            result["status"] = "DONE"
            return result

        # Feed reflection back into next observation
        observation = {
            "goal": input_data,
            "result": result,
            "reflection": reflection,
        }

    print("\n⚠ Max iterations reached.")
    result["status"] = "PARTIAL"
    return result
