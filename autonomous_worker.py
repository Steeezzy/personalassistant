"""
autonomous_worker.py — Daemon loop that processes goals from the queue.

Run as a separate process alongside the FastAPI server:
    python autonomous_worker.py

The worker:
1. Dequeues the highest-priority pending goal
2. Plans steps via the LLM
3. Validates + executes steps
4. Evaluates progress (rules first, LLM if ambiguous)
5. Loops (retry / replan) or marks done / failed
6. Sleeps and repeats
"""

import json
import signal
import sys
import time

from shared import (
    MAX_ITERATIONS,
    ask_llm_with_retry,
    deterministic_plan_fallback,
    execute_steps,
    infer_forbidden_actions,
    infer_plan_gaps,
    is_truly_complete,
    load_memory,
    parse_controller_output,
    render_final_prompt,
    save_memory,
    summarize_execution_for_state,
    validate_steps,
)
from goal_queue import dequeue_next, mark_done, mark_failed, update_goal
from goal_state import init_state, update_state, has_errors, has_results
from evaluator import evaluate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 2.0       # How often to check for new goals
MIN_ITERATION_GAP_SECONDS = 2.0   # Rate limit between planning iterations
MAX_WORKER_ITERATIONS = 3         # Per-goal max iterations (matches MAX_ITERATIONS)

# ---------------------------------------------------------------------------
# Graceful Shutdown
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    print(f"\n[WORKER] Received signal {signum}, shutting down gracefully...")
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(msg, goal_id=None):
    """Structured log output."""
    prefix = f"[WORKER][{goal_id[:8]}]" if goal_id else "[WORKER]"
    timestamp = time.strftime("%H:%M:%S")
    print(f"{timestamp} {prefix} {msg}")


# ---------------------------------------------------------------------------
# Core: Process a Single Goal
# ---------------------------------------------------------------------------


def process_goal(goal_obj):
    """
    Process a single goal through the plan → execute → evaluate loop.

    Returns:
        (success: bool, result_or_error: any)
    """
    goal_id = goal_obj["id"]
    goal_text = goal_obj["goal"]

    log(f"Processing goal: '{goal_text}'", goal_id)
    log(f"Attempt {goal_obj.get('attempts', 1)}/{goal_obj.get('max_attempts', 3)}", goal_id)

    # Initialize per-goal state
    memory = load_memory()
    state = init_state(goal_text, memory=memory)

    all_results = []
    previous_steps = []
    raw_outputs = []

    for iteration in range(MAX_WORKER_ITERATIONS):
        if _shutdown:
            log("Shutdown requested, pausing goal", goal_id)
            update_goal(goal_id, status="pending")  # Put it back
            return False, "Worker shutdown"

        state["iteration"] = iteration
        log(f"Iteration {iteration + 1}/{MAX_WORKER_ITERATIONS}", goal_id)

        # Rate limit between iterations
        if iteration > 0:
            time.sleep(MIN_ITERATION_GAP_SECONDS)

        # ---------------------------------------------------------------
        # PLAN: Ask LLM for steps
        # ---------------------------------------------------------------
        prompt = render_final_prompt(goal_text, state, previous_steps)
        llm_output, attempts = ask_llm_with_retry(
            prompt,
            label_prefix=f"WORKER[{goal_id[:8]}] iter={iteration}",
            max_attempts=2,
        )
        raw_outputs.append(attempts)

        if not llm_output:
            log("LLM returned no output, trying fallback", goal_id)
            steps = deterministic_plan_fallback(goal_text)
            if not steps:
                log("No fallback plan available", goal_id)
                state["errors"].append({
                    "iteration": iteration,
                    "error": "LLM failed and no deterministic fallback",
                })
                continue

            controller = {
                "mode": "PLAN",
                "complete": True,
                "confidence": 0.5,
                "steps": steps,
            }
        else:
            controller = parse_controller_output(llm_output)
            if not controller:
                log("Failed to parse LLM output, trying fallback", goal_id)
                steps = deterministic_plan_fallback(goal_text)
                if steps:
                    controller = {
                        "mode": "PLAN",
                        "complete": True,
                        "confidence": 0.5,
                        "steps": steps,
                    }
                else:
                    state["errors"].append({
                        "iteration": iteration,
                        "error": "Invalid controller output, no fallback",
                    })
                    previous_steps = []
                    continue

        mode = controller["mode"]
        complete = controller["complete"]
        confidence = controller["confidence"]
        steps = controller["steps"]

        log(f"Plan: mode={mode}, steps={len(steps)}, confidence={confidence}", goal_id)

        # ---------------------------------------------------------------
        # VALIDATE: Check plan against rules
        # ---------------------------------------------------------------
        valid, reason = validate_steps(steps)
        gaps = infer_plan_gaps(goal_text, steps) if complete else []
        forbidden = infer_forbidden_actions(goal_text, steps)

        if not valid or gaps or forbidden:
            log(f"Validation failed: valid={valid}, gaps={gaps}, forbidden={forbidden}", goal_id)

            # Try deterministic fallback
            fallback_steps = deterministic_plan_fallback(goal_text)
            fallback_valid, _ = validate_steps(fallback_steps)
            fallback_forbidden = infer_forbidden_actions(goal_text, fallback_steps)
            fallback_gaps = infer_plan_gaps(goal_text, fallback_steps)

            if fallback_valid and not fallback_forbidden and not fallback_gaps:
                log("Using deterministic fallback plan", goal_id)
                steps = fallback_steps
                mode = "REPAIR"
                confidence = min(confidence, 0.6)
            else:
                state["errors"].append({
                    "iteration": iteration,
                    "error": reason or "Plan validation failed",
                    "gaps": gaps,
                    "forbidden": forbidden,
                })
                previous_steps = steps
                continue

        # ---------------------------------------------------------------
        # EXECUTE: Run the validated steps
        # ---------------------------------------------------------------
        log(f"Executing {len(steps)} steps", goal_id)
        execution_result = execute_steps(steps)

        if isinstance(execution_result, dict) and execution_result.get("error"):
            log(f"Execution error: {execution_result.get('error')}", goal_id)
            update_state(state, controller, execution_result)
            previous_steps = steps
            continue

        # Record results
        for item in execution_result:
            all_results.append({
                "iteration": iteration,
                "mode": mode,
                "confidence": confidence,
                "step": item["step"],
                "action": item["action"],
                "result": item["result"],
            })

        # Update state with results
        update_state(state, controller, execution_result)
        previous_steps = steps

        log(f"Execution complete: {len(execution_result)} results", goal_id)

        # ---------------------------------------------------------------
        # EVALUATE: Determine if goal is complete
        # ---------------------------------------------------------------
        decision = evaluate(goal_text, state, goal_obj)

        log(f"Evaluation: {decision}", goal_id)

        if decision["complete"]:
            log(f"Goal COMPLETE: {decision['reason']}", goal_id)
            save_memory(goal_text, state["completed_actions"])
            return True, all_results

        if decision["abort"]:
            log(f"Goal ABORTED: {decision['reason']}", goal_id)
            return False, {
                "error": decision["reason"],
                "results": all_results,
            }

        if decision["retry"]:
            log(f"RETRYING: {decision['reason']}", goal_id)
            continue

        if decision["replan"]:
            log(f"REPLANNING: {decision['reason']}", goal_id)
            previous_steps = steps  # Give LLM context about what failed
            continue

    # Exhausted all iterations
    log("All iterations exhausted", goal_id)

    # If we have results, consider it a partial success
    if all_results:
        save_memory(goal_text, state["completed_actions"])
        return True, all_results

    return False, {
        "error": "Max iterations reached without completion",
        "state_summary": state.get("last_summary", ""),
    }


# ---------------------------------------------------------------------------
# Main Daemon Loop
# ---------------------------------------------------------------------------


def main():
    """
    Continuously poll the goal queue and process goals.
    Runs until SIGINT or SIGTERM.
    """
    log("Autonomous worker started")
    log(f"Poll interval: {POLL_INTERVAL_SECONDS}s")
    log(f"Max iterations per goal: {MAX_WORKER_ITERATIONS}")

    while not _shutdown:
        try:
            goal = dequeue_next()

            if not goal:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            goal_id = goal["id"]
            log(f"Dequeued goal: {goal_id}", goal_id)

            # Process the goal
            success, result = process_goal(goal)

            if success:
                mark_done(goal_id, result)
                log("Goal marked DONE", goal_id)
            else:
                # Check if we should retry
                attempts = goal.get("attempts", 1)
                max_attempts = goal.get("max_attempts", 3)

                if attempts < max_attempts:
                    log(f"Goal failed, will retry (attempt {attempts}/{max_attempts})", goal_id)
                    update_goal(goal_id, status="pending")
                else:
                    error_info = result if isinstance(result, dict) else {"error": str(result)}
                    mark_failed(goal_id, error_info)
                    log(f"Goal marked FAILED after {attempts} attempts", goal_id)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Unexpected error: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)

    log("Worker shut down cleanly")


if __name__ == "__main__":
    main()
