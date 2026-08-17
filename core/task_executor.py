"""
core.task_executor
~~~~~~~~~~~~~~~~~~

Runs a :class:`~core.task_plan.Task` produced by the planner.

Responsibilities
----------------
* Walk the plan's steps in order.
* Before each step: ask the confirmation manager if the step is risky.
  Denial on a HIGH_RISK step aborts the whole mission; denial on a
  regular confirmation just marks the step skipped.
* Translate the step into a router command string and dispatch it
  through the existing ``CommandRouter``. We do NOT bypass the
  router - all of arvis's tools, deterministic matchers, LLM agent
  fallback, and routine plumbing stay intact.
* Verify the result via :class:`~core.verification.Verifier`.
* On failure: retry up to ``Config.task_max_retries``; on retry
  exhaustion, replan once (max ``Config.task_max_replans``); on
  replan failure, mark FAILED and speak the error.
* On cancellation: stop immediately between steps. The voice loop
  uses ``cancel()`` from the mic listener when it detects "stop" or
  barge-in.
* Broadcast every state transition through the optional
  ``state_hub`` callback so the web dashboard / SocketIO can render
  a live mission panel.

Threading model
---------------
One daemon thread per task. The executor NEVER blocks the
microphone loop - it runs in the background and only calls back
into TTS / STT for explicit confirmation prompts (which the mic
loop is not using at the moment). Cancellation is cooperative
through a ``threading.Event`` checked between steps.

Limits (capped by config)
-------------------------
* ``task_max_steps`` - hard cap on steps the executor will walk
* ``task_max_duration_s`` - hard cap on wall-clock time
* ``task_max_retries`` - per-step retry attempts
* ``task_max_replans`` - one-shot replanning budget per task
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from core.base import ToolResult
from core.config import get_config
from core.confirmation import ConfirmationManager, RiskLevel
from core.logger import get_logger
from core.planner import TaskPlanner
from core.router import CommandRouter
from core.speech import speak, stop_speaking
from core.task_context import TaskContext
from core.task_plan import Task, TaskStep, hint_to_router_text
from core.task_state import TaskState, TaskStateInfo, TaskStepInfo
from core.verification import Verifier, VerifyOutcome

log = get_logger(__name__)

# Callback signature for state broadcasts.
StateCallback = Callable[[TaskStateInfo], None]


def _default_state_callback(info: TaskStateInfo) -> None:
    """Logging fallback when no dashboard is wired in."""
    if info.state in {TaskState.COMPLETED.value, TaskState.FAILED.value, TaskState.CANCELLED.value}:
        log.info("[TASK] %s state=%s error=%s", info.goal, info.state, info.error)
    elif info.state == TaskState.EXECUTING.value and info.steps:
        # Only log the in-progress transition for the step that just
        # became active to avoid spamming on every broadcast.
        cur = next((s for s in info.steps if s.status == "in_progress"), None)
        if cur is not None and cur.message:
            log.info("[TASK] step %d/%d: %s", info.current_step, info.total_steps, cur.message)


class TaskExecutor:
    """Stateful executor. One instance per session is plenty."""

    def __init__(
        self,
        router: CommandRouter,
        ctx: TaskContext,
        confirmation: Optional[ConfirmationManager] = None,
        verifier: Optional[Verifier] = None,
        planner: Optional[TaskPlanner] = None,
        state_callback: Optional[StateCallback] = None,
    ) -> None:
        self._router = router
        self._ctx = ctx
        self._confirmation = confirmation or ConfirmationManager()
        self._verifier = verifier or Verifier()
        self._planner = planner or TaskPlanner(router=router)
        self._state_cb = state_callback or _default_state_callback

        self._cancel_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_task: Optional[Task] = None
        self._snapshot_lock = threading.Lock()
        self._snapshot = TaskStateInfo()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run_async(self, task: Task) -> threading.Thread:
        """Start *task* on a daemon thread. Returns the thread (already started)."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("[TASK] run_async called while another task is running; cancelling previous")
            self.cancel()
            self._thread.join(timeout=2)
        self._cancel_event = threading.Event()
        self._current_task = task
        thread = threading.Thread(
            target=self._run_task,
            args=(task,),
            name=f"arvis-task-{int(time.time())}",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return thread

    def cancel(self) -> None:
        """Request cancellation. Returns immediately; the executor checks
        between steps."""
        log.info("[TASK] cancel requested")
        self._cancel_event.set()
        # Also stop any TTS currently happening so the user gets
        # silence to speak into.
        try:
            stop_speaking()
        except Exception:
            pass

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the current task finishes (or *timeout* seconds).

        Returns ``True`` if it finished cleanly, ``False`` on timeout.
        """
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def state_snapshot(self) -> TaskStateInfo:
        with self._snapshot_lock:
            # Return a shallow copy so callers cannot mutate our state.
            from copy import deepcopy
            return deepcopy(self._snapshot)

    @property
    def current_task(self) -> Optional[Task]:
        return self._current_task

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def _run_task(self, task: Task) -> None:
        cfg = get_config()
        task.state = TaskState.EXECUTING
        task.started_at = time.time()
        replans_used = 0
        try:
            while True:
                if self._check_cancel(task, reason="cancelled by user"):
                    return
                self._check_limits(task)

                # Process any pending steps; if none, we're done.
                step = task.pending_step()
                if step is None:
                    self._finalize(task, TaskState.COMPLETED, error=None)
                    return

                # Move step to in_progress.
                step.status = "in_progress"
                step.retries = 0
                task.current_step = step.id
                self._broadcast(task, message=f"Working on {step.description}.")

                # Risk check.
                risk = self._confirmation.classify(step)
                if risk is not RiskLevel.SAFE:
                    self._broadcast_state(task, TaskState.WAITING_CONFIRMATION)
                    speak(self._confirmation_prompt(step, risk))
                    if risk is RiskLevel.HIGH_RISK:
                        # Extra pause for HIGH_RISK to deter accidental yes.
                        time.sleep(cfg.confirmation_high_risk_pause_s)
                    if not self._confirmation.ask(step):
                        step.status = "failed"
                        step.error = "user declined confirmation"
                        if risk is RiskLevel.HIGH_RISK:
                            self._finalize(task, TaskState.CANCELLED, error=step.error)
                            return
                        # Otherwise just skip and keep going.
                        speak(f"Skipping {step.description}.")
                        self._broadcast(task)
                        continue
                    self._broadcast_state(task, TaskState.EXECUTING)

                # Execute with retry.
                outcome = self._execute_with_retries(step, task)
                if outcome == "ok":
                    # Re-check cancel before moving to the next step
                    # so a mid-step cancel surfaces as a CANCELLED
                    # state instead of silently completing.
                    if self._check_cancel(task, reason="cancelled between steps"):
                        return
                    continue  # Move to next pending step.
                if outcome == "retry_replan":
                    if replans_used >= cfg.task_max_replans:
                        self._finalize(task, TaskState.FAILED, error="replans exhausted")
                        return
                    replans_used += 1
                    self._broadcast_state(task, TaskState.RECOVERING)
                    speak("Replanning.")
                    new_task = self._replan(task, step)
                    if new_task is None or not new_task.steps:
                        self._finalize(task, TaskState.FAILED, error="replan produced no steps")
                        return
                    # Splice the replanned steps in starting from the failed one.
                    self._splice_replan(task, step, new_task)
                    continue
                # outcome == "failed"
                self._finalize(task, TaskState.FAILED, error=step.error or "step failed")
                return

        except _TaskLimitExceeded as exc:
            self._finalize(task, TaskState.FAILED, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("[TASK] executor crashed")
            self._finalize(task, TaskState.FAILED, error=f"executor error: {exc!r}")

    # ------------------------------------------------------------------
    # Step execution with retries
    # ------------------------------------------------------------------
    def _execute_with_retries(self, step: TaskStep, task: Task) -> str:
        cfg = get_config()
        last_result: Optional[ToolResult] = None
        last_error = ""
        for attempt in range(cfg.task_max_retries + 1):
            if self._check_cancel(task, reason="cancelled mid-step"):
                return "ok"  # Will be caught at the top of the outer loop.
            try:
                self._broadcast_state(task, TaskState.EXECUTING)
                command = hint_to_router_text(step)
                log.info("[TASK] step=%d attempt=%d cmd=%r", step.id, attempt + 1, command)
                result = self._router.dispatch(command, context={"task": task, "ctx": self._ctx})
            except Exception as exc:
                log.warning("[TASK] dispatch raised: %s", exc)
                result = ToolResult(success=False, message=f"dispatch raised {exc!r}")
            last_result = result
            # Fast-fail: if the router itself said "no tool handled this
            # command" there is nothing a retry can fix.  Retrying just
            # wastes audio ("Retrying." * 3) and looks broken.  Bail out
            # immediately and let the outer loop surface the error to
            # the user / planner.
            if (
                not result.success
                and result.message
                and "am not sure how to handle" in result.message.lower()
            ):
                step.status = "failed"
                step.error = "unhandled_command"
                log.info(
                    "[TASK] step=%d unhandled command %r - skipping retries",
                    step.id, command,
                )
                self._broadcast(task)
                return "failed"
            step.result = result
            self._ctx.record_tool(step.tool_hint or "unknown", result, step.arguments)
            step.message = self._summarize_result(result)

            # Verification.
            self._broadcast_state(task, TaskState.VERIFYING)
            verify_outcome = self._verifier.verify(step, result)
            log.info(
                "[TASK] step=%d verified=%s detail=%s",
                step.id,
                verify_outcome.verified,
                verify_outcome.detail,
            )

            if verify_outcome.verified:
                step.status = "done"
                step.error = None
                self._broadcast(task)
                if step.message:
                    speak(step.message)
                return "ok"

            # Verification failed.
            step.retries += 1
            last_error = verify_outcome.detail or "verification failed"
            if attempt < cfg.task_max_retries:
                speak("Retrying.")
                time.sleep(min(2.0, 0.5 + attempt * 0.5))
                continue
            break

        # Out of retries.
        step.status = "failed"
        step.error = last_error
        last_msg = last_result.message if last_result is not None else last_error
        speak(f"Could not complete {step.description}: {last_msg}")
        self._broadcast(task)
        # Decide whether to replan or just fail.
        if self._should_replan(step):
            return "retry_replan"
        return "failed"

    def _should_replan(self, step: TaskStep) -> bool:
        # Replan when the failure looks like a wrong tool / bad
        # arguments rather than a missing capability. Concretely: if
        # the router returned success=False AND the verifier added a
        # note that the process is missing, replanning with a different
        # tool_hint may help.
        if step.result is None:
            return False
        if step.result.success:
            return False
        haystack = " ".join(
            str(x).lower()
            for x in (step.error, step.result.message, step.result.data.get("detail") if isinstance(step.result.data, dict) else None)
            if x
        )
        return (
            "no matching process" in haystack
            or "not found" in haystack
            or "could not" in haystack
        )

    # ------------------------------------------------------------------
    # Replanning
    # ------------------------------------------------------------------
    def _replan(self, task: Task, failed_step: TaskStep) -> Optional[Task]:
        """Ask the planner to come up with a new sub-plan starting at
        *failed_step*.

        We keep the goal the same; the new plan will likely skip the
        failing step or substitute a different tool_hint for it.
        """
        try:
            ctx_snapshot = self._ctx.snapshot() if hasattr(self._ctx, "snapshot") else {}
            prompt = (
                f"The previous plan for goal '{task.goal}' failed at step "
                f"{failed_step.id} ({failed_step.description}). "
                f"Error: {failed_step.error or 'unknown'}. "
                f"Please return a fresh plan (using one of the three JSON "
                f"shapes from the system prompt) that accomplishes the goal "
                f"without repeating the failing step."
            )
            new_task = self._planner.plan(prompt, ctx=self._ctx)
            return new_task
        except Exception as exc:
            log.warning("[TASK] replan failed: %s", exc)
            return None

    def _splice_replan(self, task: Task, failed_step: TaskStep, new_task: Task) -> None:
        """Replace the failed step with the new task's steps, renumber ids."""
        idx = next((i for i, s in enumerate(task.steps) if s.id == failed_step.id), None)
        if idx is None:
            return
        # Mark already-completed steps as done.
        for s in task.steps:
            if s.status not in {"done", "skipped"}:
                s.status = "failed"
                s.error = (s.error or "aborted for replan")
        replacement = list(new_task.steps)
        # Renumber ids so the broadcast stays consistent.
        for i, s in enumerate(replacement):
            s.id = idx + i + 1
        task.steps = task.steps[:idx] + replacement

    # ------------------------------------------------------------------
    # Cancellation / limits
    # ------------------------------------------------------------------
    def _check_cancel(self, task: Task, *, reason: str = "") -> bool:
        if not self._cancel_event.is_set():
            return False
        log.info("[TASK] cancelling: %s", reason)
        task.state = TaskState.CANCELLED
        task.error = reason or "cancelled"
        self._finalize(task, TaskState.CANCELLED, error=task.error)
        return True

    def _check_limits(self, task: Task) -> None:
        cfg = get_config()
        if task.started_at and (time.time() - task.started_at) > cfg.task_max_duration_s:
            raise _TaskLimitExceeded(f"task exceeded max duration ({cfg.task_max_duration_s}s)")
        if len(task.steps) > cfg.task_max_steps:
            raise _TaskLimitExceeded(f"task has more than {cfg.task_max_steps} steps")

    # ------------------------------------------------------------------
    # Finalisation + broadcasting
    # ------------------------------------------------------------------
    def _finalize(self, task: Task, state: TaskState, error: Optional[str]) -> None:
        task.state = state
        task.error = error
        task.finished_at = time.time()
        self._broadcast_state(task, state)
        if state == TaskState.COMPLETED:
            speak(self._completion_message(task))
        elif state == TaskState.FAILED:
            speak(f"I could not complete the task. {error or ''}".strip())
        elif state == TaskState.CANCELLED:
            speak("Task cancelled.")

    def _broadcast(self, task: Task, *, message: Optional[str] = None) -> None:
        with self._snapshot_lock:
            self._snapshot = self._build_snapshot(task, TaskState.EXECUTING)
        if message is not None and self._snapshot.steps:
            # Stamp the message onto the currently in_progress step so
            # the dashboard can surface it without recomputing.
            for s in self._snapshot.steps:
                if s.id == task.current_step:
                    s.message = message
                    break
        try:
            self._state_cb(self.state_snapshot())
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("[TASK] state callback raised: %s", exc)

    def _broadcast_state(self, task: Task, state: TaskState) -> None:
        task.state = state
        with self._snapshot_lock:
            self._snapshot = self._build_snapshot(task, state)
        try:
            self._state_cb(self.state_snapshot())
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("[TASK] state callback raised: %s", exc)

    def _build_snapshot(self, task: Task, state: TaskState) -> TaskStateInfo:
        steps: List[TaskStepInfo] = []
        for s in task.steps:
            steps.append(
                TaskStepInfo(
                    id=s.id,
                    description=s.description,
                    tool_hint=s.tool_hint or "",
                    status=s.status,
                    retries=s.retries,
                    error=s.error,
                    message=getattr(s, "message", None),
                )
            )
        return TaskStateInfo(
            active=not state.is_terminal(),
            goal=task.goal,
            state=state.value,
            current_step=task.current_step,
            total_steps=len(task.steps),
            steps=steps,
            error=task.error,
            started_at=task.started_at,
            finished_at=task.finished_at,
        )

    # ------------------------------------------------------------------
    # Speech helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _confirmation_prompt(step: TaskStep, risk: RiskLevel) -> str:
        desc = step.description or step.tool_hint or "perform this action"
        if risk is RiskLevel.HIGH_RISK:
            return (
                f"Warning, sir. I am about to {desc}. "
                f"This is a high-risk action. Please say yes to continue."
            )
        return f"I am about to {desc}. Should I continue?"

    @staticmethod
    def _summarize_result(result: ToolResult) -> str:
        if result is None:
            return "No result."
        if not result.success:
            return f"Failed: {result.message or 'unknown error'}"
        msg = (result.message or "").strip()
        if msg:
            return msg
        return "Done."

    @staticmethod
    def _completion_message(task: Task) -> str:
        done = sum(1 for s in task.steps if s.status == "done")
        total = len(task.steps)
        if done == total:
            return f"Task complete. All {total} steps succeeded."
        return f"Task finished. {done} of {total} steps succeeded."


class _TaskLimitExceeded(Exception):
    """Internal sentinel raised when the executor's step / time budget is exceeded."""


__all__ = ["TaskExecutor", "StateCallback"]
