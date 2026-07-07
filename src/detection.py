"""Pattern detectors for the simulation.

Two independent detectors live here:

* :class:`StabilityDetector` -- recognises when the simulation has settled into
  a still life, an empty board or a short-period oscillator so the caller can
  stop the run (requirement *h*).
* :class:`GliderDetector` -- locates gliders and logs them through the Hydra
  colorised logger (requirement *i*).
"""
import logging
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
def _glider_phases() -> List[np.ndarray]:
    """Return every 3x3 glider phase in all four diagonal orientations.

    A glider cycles through four shapes as it travels; rotating those four base
    shapes by 0/90/180/270 degrees covers gliders heading in any diagonal
    direction.  Duplicate shapes are removed.
    """
    base = [
        np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.uint8),
        np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=np.uint8),
    ]
    phases: List[np.ndarray] = []
    for shape in base:
        for k in range(4):
            rotated = np.rot90(shape, k)
            if not any(np.array_equal(rotated, seen) for seen in phases):
                phases.append(rotated)
    return phases


@typechecked
class StabilityDetector:
    """Detect that the board no longer changes (or cycles with a short period).

    The last ``history`` boards are hashed; if a freshly computed board matches
    a recently seen one the simulation is considered stable.
    """

    def __init__(self, history: int = 3) -> None:
        self.history: int = max(1, history)
        self._recent: Deque[int] = deque(maxlen=self.history)

    def update(self, board: np.ndarray) -> Tuple[bool, Optional[int]]:
        """Register ``board`` and report stability.

        Returns ``(is_stable, period)`` where ``period`` is the number of
        generations after which the pattern repeats (``0`` for a still life),
        or ``None`` when the board is not yet known to be stable.
        """
        digest = hash(board.tobytes())
        period: Optional[int] = None
        if digest in self._recent:
            # Distance from the end of the deque gives the cycle period.
            recent = list(self._recent)
            period = len(recent) - recent.index(digest)
        self._recent.append(digest)
        return (period is not None, period)


@typechecked
class GliderDetector:
    """Find gliders on the board and log each detection in colour."""

    def __init__(self) -> None:
        self._phases: List[np.ndarray] = _glider_phases()

    def detect(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """Return the ``(row, col)`` top-left corner of every glider found."""
        rows, cols = board.shape
        found: List[Tuple[int, int]] = []
        for r in range(rows - 2):
            for c in range(cols - 2):
                window = board[r:r + 3, c:c + 3]
                # A glider occupies exactly five cells; skip cheaply otherwise.
                if window.sum() != 5:
                    continue
                if any(
                        np.array_equal(window, phase)
                        for phase in self._phases):
                    found.append((r, c))
        return found

    def log_detections(self, iteration: int, board: np.ndarray) -> int:
        """Detect gliders and log them; return how many were found."""
        gliders = self.detect(board)
        for (r, c) in gliders:
            log.info(
                "Glider detected at row=%d, col=%d "
                "(iteration %d)", r, c, iteration)
        return len(gliders)
