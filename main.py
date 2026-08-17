"""
Entry point for the Goal-Decomposition Agentic Loop.

Run:  python main.py
"""

import os
import time
import yaml
from dotenv import load_dotenv
import openai

from agent.loop import perceive, reason, act, reflect
from agent.tools import TOOLS
from agent.memory_manager import MemoryManager
from agent.harness import safe_reason, safe_act, TokenBudget
from agent.logger import log_step

load_dotenv()


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    openai.api_key = os.environ["OPENROUTER_API_KEY"]
    openai.api_base = "https://openrouter.ai/api/v1"
    client = openai

    model          = cfg["model"]
    max_iterations = cfg["max_iterations"]
    max_retries    = cfg["max_retries"]
    base_delay     = cfg["base_delay"]
    token_limit    = cfg.get("token_budget", 10_000)

    goal = input("Enter your goal: ").strip()
    if not goal:
        goal = "Learn machine learning from scratch"

    memory_manager = MemoryManager()
    budget         = TokenBudget(limit=token_limit)
    observation    = perceive(goal)
    last_reflection = None

    print(f"\nGoal: {goal}")
    print(f"Model: {model} | Max iterations: {max_iterations}\n")

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*50}")
        print(f"  Iteration {iteration}")
        print(f"{'='*50}")

        # ── PERCEIVE (already done; observation is updated at end of loop) ──
        t0 = time.time()
        log_step(iteration, "perceive", goal, str(observation)[:100], (time.time()-t0)*1000)

        # ── REASON ──────────────────────────────────────────────────────────
        memory = memory_manager.retrieve(goal)
        t0 = time.time()
        try:
            plan = safe_reason(reason, observation, memory, client, model, max_retries, base_delay)
        except Exception as e:
            log_step(iteration, "reason", str(observation)[:100], "", 0, error=str(e))
            break
        log_step(iteration, "reason", str(observation)[:100], str(plan)[:100], (time.time()-t0)*1000)
        print(f"[Reason] {plan['action']} — {plan.get('reasoning','')}")

        # ── ACT ─────────────────────────────────────────────────────────────
        last_steps = observation.get("result", {}).get("steps", []) if isinstance(observation.get("result"), dict) else []
        t0 = time.time()
        result = safe_act(act, plan, TOOLS, client, model, max_retries, base_delay, last_steps=last_steps)
        log_step(iteration, "act", str(plan)[:100], str(result)[:100], (time.time()-t0)*1000,
                 error=result.get("error"))
        steps = result.get("steps", [])
        print(f"[Act]    {len(steps)} steps produced")
        if steps:
            for i, s in enumerate(steps, 1):
                print(f"         {i}. {s}")

        # ── REFLECT ─────────────────────────────────────────────────────────
        t0 = time.time()
        reflection = reflect(result, observation)
        log_step(iteration, "reflect", str(result)[:100], str(reflection)[:100], (time.time()-t0)*1000)
        print(f"[Reflect] done={reflection['is_done']} score={reflection['quality_score']}")
        print(f"          {reflection['next_instruction']}")

        # ── MEMORY WRITE ────────────────────────────────────────────────────
        memory_manager.add({
            "iteration": iteration,
            "action": plan["action"],
            "steps_count": len(steps),
            "reflection": reflection,
        })

        # ── STUCK GUARD ─────────────────────────────────────────────────────
        if last_reflection and last_reflection == reflection:
            print("\n[Guard] Stuck — identical reflection twice. Stopping.")
            result["status"] = "STUCK"
            break

        last_reflection = reflection

        if reflection["is_done"]:
            print("\n✓ Goal fully decomposed!")
            result["status"] = "DONE"
            break

        # Feed reflection back into next observation
        observation = {
            "goal": goal,
            "result": result,
            "reflection": reflection,
        }
    else:
        print("\n⚠ Max iterations reached.")
        result["status"] = "PARTIAL"

    print(f"\nFinal status : {result.get('status')}")
    print(f"Total steps  : {len(result.get('steps', []))}")
    return result


if __name__ == "__main__":
    main()
