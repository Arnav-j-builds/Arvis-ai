"""
Routines package.

Stores user-defined multi-step workflows. Each routine is a list of
:class:`~core.base.RoutineAction` instances; the manager handles
persistence, the tool (:mod:`routines.commands`) handles voice I/O.
"""

from routines.commands import InteractiveRoutineBuilder, RoutinesTool, register_routines_tool
from routines.manager import Routine, RoutineManager

__all__ = [
    "InteractiveRoutineBuilder",
    "Routine",
    "RoutineManager",
    "RoutinesTool",
    "register_routines_tool",
]
