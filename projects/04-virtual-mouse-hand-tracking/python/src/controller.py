"""Safe preview and explicitly enabled operating-system mouse controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .errors import ControllerError, DependencyError


class MouseController(Protocol):
    @property
    def screen_size(self) -> tuple[int, int]: ...

    def move_to(self, x: int, y: int) -> None: ...

    def click(self, button: str) -> None: ...

    def mouse_down(self, button: str = "left") -> None: ...

    def mouse_up(self, button: str = "left") -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class PreviewMouseController:
    """A no-side-effect controller used by default and by tests."""

    width: int = 1920
    height: int = 1080
    actions: list[tuple[str, int | str | None, int | None]] = field(default_factory=list)
    _left_down: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("preview screen dimensions must be positive")

    @property
    def screen_size(self) -> tuple[int, int]:
        return self.width, self.height

    def move_to(self, x: int, y: int) -> None:
        self.actions.append(("move", int(x), int(y)))

    def click(self, button: str) -> None:
        self.actions.append(("click", str(button), None))

    def mouse_down(self, button: str = "left") -> None:
        self._left_down = True
        self.actions.append(("down", str(button), None))

    def mouse_up(self, button: str = "left") -> None:
        if self._left_down:
            self.actions.append(("up", str(button), None))
        self._left_down = False

    def close(self) -> None:
        self.mouse_up("left")


class PyAutoGuiMouseController:
    """Thin, fail-safe wrapper loaded only after explicit control opt-in."""

    def __init__(self, pyautogui_module=None) -> None:
        if pyautogui_module is None:
            try:
                import pyautogui as pyautogui_module  # type: ignore[no-redef]
            except (ImportError, OSError) as exc:
                raise DependencyError(
                    "PyAutoGUI is unavailable. Install python/requirements.txt or use preview mode."
                ) from exc
        self._pyautogui = pyautogui_module
        self._left_down = False
        try:
            self._pyautogui.FAILSAFE = True
            self._pyautogui.PAUSE = 0.0
            size = self._pyautogui.size()
            self._screen_size = (int(size[0]), int(size[1]))
        except Exception as exc:
            raise ControllerError(f"could not initialize OS mouse control: {exc}") from exc
        if min(self._screen_size) <= 0:
            raise ControllerError("operating system returned an invalid screen size")

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    def _call(self, operation: str, *args, **kwargs) -> None:
        try:
            getattr(self._pyautogui, operation)(*args, **kwargs)
        except Exception as exc:
            raise ControllerError(f"mouse {operation} failed: {exc}") from exc

    def move_to(self, x: int, y: int) -> None:
        self._call("moveTo", int(x), int(y), duration=0, _pause=False)

    def click(self, button: str) -> None:
        self._call("click", button=button, _pause=False)

    def mouse_down(self, button: str = "left") -> None:
        self._call("mouseDown", button=button, _pause=False)
        self._left_down = True

    def mouse_up(self, button: str = "left") -> None:
        if self._left_down:
            self._call("mouseUp", button=button, _pause=False)
        self._left_down = False

    def close(self) -> None:
        self.mouse_up("left")
