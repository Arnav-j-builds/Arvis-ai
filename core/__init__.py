"""
Core package.

Provides the foundational building blocks that every feature module uses:

* :mod:`core.config`  - central configuration loaded from environment / .env
* :mod:`core.logger`  - shared logger factory
* :mod:`core.base`    - abstract base classes every tool must implement
* :mod:`core.router`  - command router that wires feature modules into the agent
* :mod:`core.speech`  - thin helpers to speak a string (does not duplicate TTS)

Nothing in this package depends on the vision / communication / routines
sub-packages. Feature modules depend on `core`, never the other way around.
"""

from core.base import BaseTool, RoutineAction, ToolResult
from core.config import Config, get_config
from core.logger import get_logger
from core.router import CommandRouter, get_router

__all__ = [
    "BaseTool",
    "RoutineAction",
    "ToolResult",
    "Config",
    "get_config",
    "get_logger",
    "CommandRouter",
    "get_router",
]
