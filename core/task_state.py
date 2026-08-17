"""
core.task_state
~~~~~~~~~~~~~~~

State machine for the autonomous task executor.

The enum is the public vocabulary the planner, the executor, and the
web API all speak. ``TaskStateInfo`` is the snapshot dataclass
broadcast over SocketIO so the dashboard can render a live progress
panel.

The states form a directed graph:

    IDLE
      └─ PLANNING
           └─ EXECUTING
                ├─ WAITING_CONFIRMATION
                │     └─ EXECUTING (or CANCELLED on denial)
                ├─ VERIFYING
                │     └─ EXECUTING (or RECOVERING on verify fail)
                ├─ RECOVERING
                │     └─ EXECUTING (or FAILED on replan exhaustion)
                └─ COMPLETED | FAILED | CANCELLED
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class TaskState(str, enum.Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    WAITING_CONFIRMATION = "waiting_confirmation"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        return self in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}


@dataclass
class TaskStepInfo:
    """Public, JSON-serialisable view of a single step.

    Mirrors :class:`core.task_plan.TaskStep` but with only the fields
    the web dashboard needs. We do NOT reuse the dataclass directly so
    the dashboard payload stays decoupled from internal fields.
    """

    id: int
    description: str
    tool_hint: str
    status: str
    retries: int = 0
    error: Optional[str] = None
    message: Optional[str] = None  # last spoken summary

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskStateInfo:
    """Snapshot of the task layer for the web API."""

    active: bool = False
    goal: Optional[str] = None
    state: str = TaskState.IDLE.value
    current_step: int = 0
    total_steps: int = 0
    steps: List[TaskStepInfo] = field(default_factory=list)
    error: Optional[str] = None
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "goal": self.goal,
            "state": self.state,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


__all__ = ["TaskState", "TaskStateInfo", "TaskStepInfo"]
