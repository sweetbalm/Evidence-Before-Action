import datetime
import json
import logging
import os
import time

from lib_results_logger import log_task_completion
from mm_agents.response_parser import extract_evidence, extract_visual_anchor

logger = logging.getLogger("desktopenv.experiment")


def _is_terminal_action(action):
    """Return True when an action terminates the episode."""
    return isinstance(action, str) and action in {"DONE", "FAIL"}


def _normalize_actions(actions):
    """Normalize agent output while preserving multi-action turns."""
    if actions is None:
        return []
    if isinstance(actions, (str, dict)):
        return [actions]
    return list(actions)


def _replace_last_action_history(agent, actions):
    """Keep PromptAgent history consistent with actions actually executed."""
    action_history = getattr(agent, "actions", None)
    if isinstance(action_history, list) and action_history:
        action_history[-1] = actions


def _is_screenshot_action(action):
    return isinstance(action, dict) and action.get("evidence_type") == "SCREENSHOT"


def _save_response_evidence(result_dir, step_num, response, controller):
    """Capture evidence before the action changes the referenced screen.

    Use OSWorld's existing screenshot/crop interface without adding an
    environment action. Missing or invalid anchors still preserve the text.
    """
    evidence_text = extract_evidence(response)
    region = extract_visual_anchor(response)
    if evidence_text is None and region is None:
        return
    evidence_dir = os.path.join(result_dir, "evidence", f"step_{step_num}")
    os.makedirs(evidence_dir, exist_ok=True)
    with open(os.path.join(evidence_dir, "evidence.txt"), "w", encoding="utf-8") as file:
        file.write((evidence_text or "") + "\n")
    if region is None:
        logger.warning("Step %s has no valid visual_anchor; saved evidence text only", step_num)
        return
    try:
        screenshot = controller.get_screenshot(region=region)
    except Exception:
        logger.exception("Failed to capture evidence for step %s", step_num)
        return
    if not screenshot:
        logger.warning("No evidence screenshot returned for step %s", step_num)
        return
    with open(os.path.join(evidence_dir, "screenshot.png"), "wb") as file:
        file.write(screenshot)


def _save_solver_evidence(result_dir, step_num, evidence_text, actions, screenshot):
    """Save solver evidence when its first action is a screenshot capture.

    The text inside ``<evidence>`` is optional: a solver can provide a
    screenshot as the evidence by itself.  Therefore the presence of text
    must not gate creation of the evidence directory.
    """
    if not actions or not _is_screenshot_action(actions[0]) or not screenshot:
        logger.info(
            "Skipping solver evidence for step %s: actions=%s, first_is_screenshot=%s, screenshot=%s",
            step_num,
            bool(actions),
            bool(actions and _is_screenshot_action(actions[0])),
            bool(screenshot),
        )
        return
    logger.info(
        "Saving solver evidence for step %s (text=%s, actions=%d, screenshot_bytes=%d)",
        step_num,
        bool(evidence_text),
        len(actions),
        len(screenshot),
    )
    evidence_dir = os.path.join(result_dir, "evidence", f"step_{step_num}")
    os.makedirs(evidence_dir, exist_ok=True)
    with open(os.path.join(evidence_dir, "evidence.txt"), "w", encoding="utf-8") as file:
        file.write((evidence_text or "") + "\n")
    with open(os.path.join(evidence_dir, "actions.json"), "w", encoding="utf-8") as file:
        json.dump(actions, file, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(evidence_dir, "screenshot.png"), "wb") as file:
        file.write(screenshot)


def _write_dual_trajectory(
    result_dir,
    role,
    round_num,
    action_index,
    action,
    response,
    obs,
    reward,
    done,
    info,
):
    action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
    screenshot_file = (
        f"round_{round_num + 1}_{role}_{action_index + 1}_{action_timestamp}.png"
    )
    screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
    if screenshot is not None:
        with open(os.path.join(result_dir, screenshot_file), "wb") as image_file:
            image_file.write(screenshot)

    with open(
        os.path.join(result_dir, "traj.jsonl"), "a", encoding="utf-8"
    ) as traj_file:
        traj_file.write(
            json.dumps(
                {
                    "role": role,
                    "round_num": round_num + 1,
                    "action_index": action_index + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "screenshot_file": screenshot_file
                    if screenshot is not None
                    else None,
                },
                default=str,
            )
        )
        traj_file.write("\n")


def _reset_agent(agent, runtime_logger, vm_ip):
    """Reset agents with compatibility for PromptAgent-like reset methods."""
    try:
        agent.reset(runtime_logger, vm_ip=vm_ip)
    except TypeError:
        try:
            agent.reset(vm_ip=vm_ip)
        except TypeError:
            agent.reset()


def configure_dual_agent_roles(solver, judge):
    """Add role-specific instructions to the two agents."""
    solver.system_message = (
        "You are the SOLVER. Execute the task directly. You may return multiple "
        "actions in one turn; do not assume another agent shares your memory. "
        "DONE or FAIL is treated as a request that the judge must verify.\n"
        + solver.system_message
    )
    judge.system_message = (
        "You are the JUDGE and corrective executor. Inspect the current state "
        "after the solver's entire action sequence, then return one or more "
        "actions to verify or correct the task. Use DONE only after verification.\n"
        + judge.system_message
    )


def run_single_example_single(
    agent,
    env,
    example,
    max_steps,
    instruction,
    args,
    example_result_dir,
    scores,
):
    """Run a task with the original single-agent execution loop.

    This path is selected when ``run.py --use_judge`` is false.  It still
    supports batched actions and solver-declared screenshot evidence.
    """
    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    _reset_agent(agent, runtime_logger, getattr(env, "vm_ip", None))

    time.sleep(60)
    obs = env._get_obs()
    done = False
    env.controller.start_recording()

    for step_num in range(max_steps):
        response, predicted_actions = agent.predict(instruction, obs)
        _save_response_evidence(
            example_result_dir, step_num + 1, response, env.controller,
        )
        actions = _normalize_actions(predicted_actions)
        if not actions:
            logger.warning("solver returned no actions; executing WAIT")
            actions = ["WAIT"]

        executed_actions = []
        for action_index, action in enumerate(actions):
            logger.info("Step %d action %d: %s", step_num + 1, action_index + 1, action)
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            executed_actions.append(action)

            if action_index == 0 and _is_screenshot_action(action):
                _save_solver_evidence(
                    example_result_dir,
                    step_num + 1,
                    getattr(agent, "last_evidence", None),
                    actions,
                    getattr(env, "last_action_screenshot", None),
                )

            _write_dual_trajectory(
                example_result_dir,
                "solver",
                step_num,
                action_index,
                action,
                response,
                obs,
                reward,
                done,
                info,
            )
            if done:
                break

        _replace_last_action_history(agent, executed_actions)
        if done:
            break

    time.sleep(20)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as result_file:
        result_file.write(f"{result}\n")
    log_task_completion(example, result, example_result_dir, args)
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def run_single_example_dual(
    solver,
    judge,
    env,
    example,
    max_steps,
    instruction,
    args,
    example_result_dir,
    scores,
):
    """Run one task with solver and judge alternating by prediction turn."""
    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)
    _reset_agent(solver, runtime_logger, getattr(env, "vm_ip", None))
    _reset_agent(judge, runtime_logger, getattr(env, "vm_ip", None))

    time.sleep(60)
    obs = env._get_obs()
    done = False
    env.controller.start_recording()

    for round_num in range(max_steps):
        solver_response, predicted_solver_actions = solver.predict(instruction, obs)
        # Each model prediction is a step; solver and judge must not overwrite
        # one another's evidence within a round.
        solver_step = 2 * round_num + 1
        judge_step = solver_step + 1
        _save_response_evidence(
            example_result_dir, solver_step, solver_response, env.controller,
        )
        solver_actions = _normalize_actions(predicted_solver_actions)
        if not solver_actions:
            logger.warning("solver returned no actions; executing WAIT")
            solver_actions = ["WAIT"]

        solver_terminal_request = None
        executed_solver_actions = []
        for action_index, action in enumerate(solver_actions):
            if _is_terminal_action(action):
                solver_terminal_request = action
                logger.info(
                    "Round %d solver requested termination with %s; deferring to judge",
                    round_num + 1,
                    action,
                )
                break

            logger.info(
                "Round %d solver action %d: %s",
                round_num + 1,
                action_index + 1,
                action,
            )
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            if action_index == 0 and _is_screenshot_action(action):
                _save_solver_evidence(
                    example_result_dir,
                    solver_step,
                    getattr(solver, "last_evidence", None),
                    solver_actions,
                    getattr(env, "last_action_screenshot", None),
                )
            executed_solver_actions.append(action)
            _write_dual_trajectory(
                example_result_dir,
                "solver",
                round_num,
                action_index,
                action,
                solver_response,
                obs,
                reward,
                done,
                info,
            )
            if done:
                break

        _replace_last_action_history(solver, executed_solver_actions)
        if done:
            break

        judge_instruction = (
            f"{instruction}\n\n"
            "You are the judge. The solver has completed its full action sequence. "
            "Inspect the CURRENT environment, verify the result, and correct any "
            "errors. You may return multiple actions. Use DONE only after verifying "
            "the task is complete.\n\n"
            f"Solver response:\n{solver_response}\n\n"
            f"Solver predicted actions: {solver_actions}\n"
            f"Solver executed actions: {executed_solver_actions}\n"
            f"Solver termination request: {solver_terminal_request}"
        )
        judge_response, predicted_judge_actions = judge.predict(judge_instruction, obs)
        _save_response_evidence(
            example_result_dir, judge_step, judge_response, env.controller,
        )
        judge_actions = _normalize_actions(predicted_judge_actions)
        if not judge_actions:
            logger.warning("judge returned no actions; executing WAIT")
            judge_actions = ["WAIT"]

        executed_judge_actions = []
        for action_index, action in enumerate(judge_actions):
            logger.info(
                "Round %d judge action %d: %s",
                round_num + 1,
                action_index + 1,
                action,
            )
            obs, reward, done, info = env.step(action, args.sleep_after_execution)
            executed_judge_actions.append(action)
            _write_dual_trajectory(
                example_result_dir,
                "judge",
                round_num,
                action_index,
                action,
                judge_response,
                obs,
                reward,
                done,
                info,
            )
            if done:
                break

        _replace_last_action_history(judge, executed_judge_actions)
        if done:
            break

    time.sleep(20)
    result = env.evaluate()
    logger.info("Result: %.2f", result)
    scores.append(result)
    with open(
        os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8"
    ) as result_file:
        result_file.write(f"{result}\n")
    log_task_completion(example, result, example_result_dir, args)
    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(
        logging.FileHandler(os.path.join(example_result_dir, "runtime.log"))
    )
    return runtime_logger
