"""Perception helpers: turning a raw frame into facts the agent can act on.

Every module here is standard-library only and imports nothing outside this package.
The Python-tool sandbox cannot import project code, so these sources are pasted into
its bootstrap and run in one flat namespace; ``python_tool_sandbox`` strips the
intra-package imports on the way in. Keep that constraint in mind when editing: no
third-party dependencies, and no imports from elsewhere in the project.
"""

from inference.perception.background import identify_background
from inference.perception.crop import crop_ascii, window_ascii
from inference.perception.diffing import diff_grids
from inference.perception.effects import summarize_action_effects
from inference.perception.hud import detect_hud
from inference.perception.objects import match_objects
from inference.perception.pathfinding import find_path
from inference.perception.symmetry import detect_symmetry

__all__ = [
    "crop_ascii",
    "detect_hud",
    "detect_symmetry",
    "diff_grids",
    "find_path",
    "identify_background",
    "match_objects",
    "summarize_action_effects",
    "window_ascii",
]
