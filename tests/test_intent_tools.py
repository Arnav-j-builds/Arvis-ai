"""
Focused tests for the four new ARVIS capabilities:

* Universal Screen Context (cached)
* Visual Action Engine
* Skill Builder (create / run / variables)
* Browser Agent (search / ordinal / research)

These tests are intentionally hermetic: they monkey-patch pyautogui,
mss, vision.capture, and duckduckgo_search so no real screen / network
calls happen.  Each test exercises one capability in isolation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_test_state: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakePyAutoGUI:
    def __init__(self) -> None:
        self.size = lambda: (1920, 1080)
        self.clicks: List[Dict[str, Any]] = []
        self.scrolls: List[int] = []
        self.typed: List[str] = []
        self.fail_next = False

    def click(self, x, y, *args, **kwargs):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("boom")
        self.clicks.append({"x": x, "y": y})

    def doubleClick(self, x, y):
        self.clicks.append({"x": x, "y": y, "double": True})

    def rightClick(self, x, y):
        self.clicks.append({"x": x, "y": y, "right": True})

    def scroll(self, amount):
        self.scrolls.append(int(amount))

    def hscroll(self, amount):
        self.scrolls.append(int(amount))

    def typewrite(self, text):
        self.typed.append(text)


class FakeCapture:
    def __init__(self, path: str = "fake.png", width: int = 1920, height: int = 1080) -> None:
        self.path = path
        self.width = width
        self.height = height


@pytest.fixture
def fake_pyautogui(monkeypatch):
    fake = FakePyAutoGUI()
    sys.modules["pyautogui"] = fake  # type: ignore[assignment]
    return fake


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_caches(monkeypatch):
    """Reset the module-level screen + browser caches between tests so
    cached frames from a previous test do not leak into the next one."""
    from core import context_engine

    context_engine._screen_cache = None
    context_engine._browser_cache = None
    yield
    context_engine._screen_cache = None
    context_engine._browser_cache = None


@pytest.fixture
def fake_capture(monkeypatch):
    """Patch the vision capture helpers so no real screen is touched."""
    from core import visual_actions

    def _capture_primary(destination=None):
        return SimpleNamespace(path=Path("fake.png"), width=1920, height=1080, source="screen")

    def _capture_active(destination=None):
        return SimpleNamespace(path=Path("fake_active.png"), width=1920, height=1080, source="active_window")

    monkeypatch.setattr(visual_actions, "capture_primary_monitor", _capture_primary)
    monkeypatch.setattr(visual_actions, "capture_active_window", _capture_active)
    return _capture_primary


@pytest.fixture
def fake_ocr(monkeypatch):
    """Patch OCR to return a fixed snippet."""
    from core import visual_actions, context_engine
    from dataclasses import dataclass

    @dataclass
    class _OCR:
        text: str
        engine: str = "tesseract"
        confidence: float = 0.91
        language: str = "eng"

    text = "VS Code\napp.py\nTraceback (most recent call last)\nFileNotFoundError: no such file"

    def _extract(path, languages=None):
        return _OCR(text=text)

    monkeypatch.setattr(visual_actions, "extract_text", _extract)
    return text


# ---------------------------------------------------------------------------
# ScreenContext
# ---------------------------------------------------------------------------
def test_screen_context_is_cached(fake_capture, fake_ocr) -> None:
    from core.context_engine import get_screen_cache
    from core.visual_actions import build_screen_context

    a = build_screen_context()
    b = build_screen_context()
    assert a is b
    assert get_screen_cache().get() is a


def test_screen_context_force_recaptures(fake_capture, fake_ocr) -> None:
    from core.visual_actions import build_screen_context

    a = build_screen_context()
    time.sleep(0.01)
    b = build_screen_context(force=True)
    assert a.timestamp != b.timestamp


def test_intent_tool_screen_summary(fake_pyautogui, fake_capture, fake_ocr) -> None:
    from core.intent_tools import IntentTool

    tool = IntentTool()
    out = tool.execute("what is on my screen")
    assert out.success is True
    assert "VS Code" in out.message or "FileNotFoundError" in out.message or "open" in out.message.lower()


def test_intent_tool_screen_error(fake_pyautogui, fake_capture, fake_ocr) -> None:
    from core.intent_tools import IntentTool

    tool = IntentTool()
    out = tool.execute("explain this error")
    assert out.success is True
    # The error text should be present in the data dict.
    assert out.data is not None
    assert "context" in out.data
    assert "FileNotFoundError" in out.data["context"]["ocr_text"]


# ---------------------------------------------------------------------------
# VisualAction
# ---------------------------------------------------------------------------
def test_parse_target_click() -> None:
    from core.visual_actions import parse_target

    t = parse_target('click "Download"')
    assert t.action == "click"
    assert t.text.lower() == "download"


def test_parse_target_double_click() -> None:
    from core.visual_actions import parse_target

    t = parse_target("double click on Settings")
    assert t.action == "double_click"
    assert "settings" in t.text


def test_parse_target_scroll() -> None:
    from core.visual_actions import parse_target

    t = parse_target("scroll down")
    assert t.action == "scroll"
    assert t.extra == "down"


def test_parse_target_ordinal() -> None:
    from core.visual_actions import parse_target

    t = parse_target("click the second result")
    assert t.ordinal == 1
    assert "result" in t.text


def test_parse_target_colour_and_type() -> None:
    from core.visual_actions import parse_target

    t = parse_target("click the green button")
    assert t.color == "green"
    assert t.target_type == "button"


def test_visual_execute_uses_ocr_target(fake_pyautogui, fake_capture, fake_ocr) -> None:
    from core.intent_tools import IntentTool

    tool = IntentTool()
    out = tool.execute('click "VS Code"')
    assert out.success is True
    assert fake_pyautogui.clicks, "pyautogui.click should have been called"
    click = fake_pyautogui.clicks[0]
    assert 0 < click["x"] < 1920
    assert 0 < click["y"] < 1080


def test_visual_scroll(fake_pyautogui, fake_capture, fake_ocr) -> None:
    from core.intent_tools import IntentTool

    tool = IntentTool()
    out = tool.execute("scroll down")
    assert out.success is True
    assert fake_pyautogui.scrolls


def test_visual_click_unknown_target(fake_pyautogui, fake_capture, fake_ocr) -> None:
    from core.intent_tools import IntentTool

    tool = IntentTool()
    out = tool.execute('click "NonExistentButton"')
    assert out.success is False
    assert not fake_pyautogui.clicks


# ---------------------------------------------------------------------------
# Skill Builder
# ---------------------------------------------------------------------------
def _make_skill_manager(tmp_path) -> Any:
    from core.skill_manager import SkillManager
    from core.context_engine import SkillRecord

    return SkillManager(path=tmp_path / "skills.json")


def test_skill_create_and_run(tmp_path, monkeypatch) -> None:
    # Stub tools.opener before the router tries to import it.
    import types
    if "tools.opener" not in sys.modules:
        stub = types.ModuleType("tools.opener")
        stub._launch_app = lambda cmd: (True, f"Launched {cmd}")
        stub._open_url = lambda url: None
        stub.APP_SHORTCUTS = {"vscode": ("Code.exe", "all"), "code": ("Code.exe", "all")}
        stub.google_search = lambda q: "google"
        stub.youtube_search = lambda q: "youtube"
        sys.modules["tools.opener"] = stub

    from core.base import BaseTool, ToolResult
    from core.intent_tools import IntentTool
    from core.router import CommandRouter

    router = CommandRouter()

    class OpenAppTool(BaseTool):
        name = "open_app_test"
        description = "fake open app"

        def can_handle(self, command, context=None):
            return (command or "").lower().startswith("open app ")

        def execute(self, command, context=None):
            return ToolResult(success=True, message=f"Launched {command}")

    router.register(OpenAppTool(), keywords=("open app ",), priority=10)

    class WaitTool(BaseTool):
        name = "wait_test"
        description = "fake wait"

        def can_handle(self, command, context=None):
            return (command or "").lower().startswith("wait ")

        def execute(self, command, context=None):
            return ToolResult(success=True, message="waited")

    class SayTool(BaseTool):
        name = "say_test"
        description = "fake say"

        def can_handle(self, command, context=None):
            return (command or "").lower().startswith("say ")

        def execute(self, command, context=None):
            return ToolResult(success=True, message="said")

    router.register(WaitTool(), keywords=("wait ",), priority=10)
    router.register(SayTool(), keywords=("say ",), priority=10)
    sm = _make_skill_manager(tmp_path)
    sm.attach_router(router)
    from core.intent_tools import register_intent_tool
    register_intent_tool(router, skill_manager=sm)
    tool = IntentTool(skill_manager=sm)

    out = tool.execute("create a skill called coding setup")
    assert out.success is True
    assert sm.get("coding setup") is not None

    out = tool.execute("run skill coding setup")
    assert out.success is True
    # Step 1 of the seed skill opens vscode.
    assert any("Code.exe" in m or "Launched" in m or "open app vscode" in m for m in (out.data or {}).get("results", []))


def test_skill_variable_substitution(tmp_path) -> None:
    from core.skill_manager import SkillManager
    from core.context_engine import SkillRecord

    sm = SkillManager(path=tmp_path / "skills.json")
    rec = SkillRecord(
        name="open_project",
        description="Open a project at a given path",
        steps=[
            {"action": "open_app", "args": {"app": "vscode"}},
            {"action": "say", "args": {"text": "Opening {{PROJECT_PATH}}"}},
        ],
    )
    rec.variables = SkillManager.extract_variables(rec.steps)
    sm.upsert(rec)

    rendered = SkillManager.render("Opening {{PROJECT_PATH}}", {"PROJECT_PATH": "F:\\Nova"})
    assert "F:\\Nova" in rendered


def test_skill_list_and_delete(tmp_path) -> None:
    from core.intent_tools import IntentTool
    from core.context_engine import SkillRecord

    sm = _make_skill_manager(tmp_path)
    sm.upsert(SkillRecord(name="alpha", steps=[{"action": "say", "args": {"text": "hi"}}]))
    sm.upsert(SkillRecord(name="beta", steps=[{"action": "say", "args": {"text": "yo"}}]))

    tool = IntentTool(skill_manager=sm)
    out = tool.execute("list my skills")
    assert out.success is True
    assert "alpha" in out.message and "beta" in out.message

    out = tool.execute("delete skill alpha")
    assert out.success is True
    assert sm.get("alpha") is None


def test_skill_rename(tmp_path) -> None:
    from core.intent_tools import IntentTool
    from core.context_engine import SkillRecord

    sm = _make_skill_manager(tmp_path)
    sm.upsert(SkillRecord(name="old", steps=[{"action": "say", "args": {"text": "x"}}]))

    tool = IntentTool(skill_manager=sm)
    out = tool.execute("rename skill old to new")
    assert out.success is True
    assert sm.get("new") is not None
    assert sm.get("old") is None


# ---------------------------------------------------------------------------
# Browser Agent
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_ddg(monkeypatch):
    """Patch the duckduckgo_search library so we do not hit the network."""

    class FakeDDGS:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, **kwargs):
            return [
                {
                    "title": f"Result 1 for {query}",
                    "href": "https://example.com/1",
                    "body": "First snippet",
                },
                {
                    "title": f"Result 2 for {query}",
                    "href": "https://example.com/2",
                    "body": "Second snippet",
                },
                {
                    "title": f"Result 3 for {query}",
                    "href": "https://example.com/3",
                    "body": "Third snippet",
                },
            ]

    # Stub the existing tools.duckduckgo module too (it imports
    # langchain and would otherwise fail in this venv).
    import types

    mod = types.ModuleType("duckduckgo_search")
    mod.DDGS = FakeDDGS
    sys.modules["duckduckgo_search"] = mod

    stub = types.ModuleType("tools.duckduckgo")
    stub.duckduckgo_search_tool = lambda query: "stubbed"
    sys.modules["tools.duckduckgo"] = stub
    return mod


def test_browser_search_records_cache(fake_ddg) -> None:
    from core.intent_tools import IntentTool
    from core.context_engine import get_browser_cache

    tool = IntentTool()
    out = tool.execute("search for MediaPipe hand tracking")
    assert out.success is True
    cache = get_browser_cache().get()
    assert cache.search_query == "MediaPipe hand tracking"
    assert len(cache.results) >= 1


def test_browser_ordinal_lookup(fake_ddg, monkeypatch) -> None:
    # Patch the browser agent's open_url to record what gets opened.
    from core.intent_tools import IntentTool
    from core import browser_agent

    opened: List[str] = []

    def fake_open_url(url):
        opened.append(url)
        browser_agent.get_browser_cache().set_page(url, title=url)
        return True, "ok"

    monkeypatch.setattr(browser_agent, "open_url", fake_open_url)

    tool = IntentTool()
    tool.execute("search for MediaPipe")
    out = tool.execute("open the second result")
    assert out.success is True
    assert opened and "example.com/2" in opened[0]


def test_browser_research(fake_ddg) -> None:
    from core.intent_tools import IntentTool

    tool = IntentTool()
    out = tool.execute("research MediaPipe hand tracking")
    assert out.success is True
    assert "sources" in (out.data or {})
    assert "Result 1" in out.message


# ---------------------------------------------------------------------------
# Cross-feature integration
# ---------------------------------------------------------------------------
def test_existing_voice_unchanged(fake_pyautogui, fake_capture, fake_ocr) -> None:
    """Adding the new tool must not break the existing screen tool."""
    from core.intent_tools import IntentTool

    tool = IntentTool()
    out = tool.execute("what is on my screen")
    assert out.success is True
    # Should not raise; should return a useful summary.
    assert len(out.message) > 0
