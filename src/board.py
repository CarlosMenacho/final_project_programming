"""Helpers to create, load and persist Game-of-Life boards.

A *board* is a 2D :class:`numpy.ndarray` of ``uint8`` where ``1`` marks a live
cell and ``0`` a dead one.  The canonical way to seed a simulation is the
``initial.pkl`` file required by the assignment (requirement *f*).
"""
import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
def make_glider(top: int, left: int, rows: int, cols: int) -> np.ndarray:
    """Return an empty ``rows x cols`` board containing a single glider.

    The glider is placed with its bounding box top-left corner at
    ``(top, left)``.  Gliders travel diagonally and are therefore handy to
    demonstrate the glider detector (requirement *i*).
    """
    board = np.zeros((rows, cols), dtype=np.uint8)
    # Canonical south-east travelling glider (phase 0).
    glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.uint8)
    board[top:top + 3, left:left + 3] = glider
    return board


@typechecked
def random_board(rows: int,
                 cols: int,
                 density: float,
                 seed: Optional[int] = None) -> np.ndarray:
    """Return a random board where ``density`` is the fraction of live cells."""
    rng = np.random.default_rng(seed)
    return (rng.random((rows, cols)) < density).astype(np.uint8)


@typechecked
def default_initial_state(rows: int,
                          cols: int,
                          seed: Optional[int] = 42) -> np.ndarray:
    """Build a reasonable demo board: a moving glider plus some random noise.

    The glider guarantees the glider detector has something to find, while the
    random noise produces a lively simulation that eventually settles.
    """
    board = random_board(rows, cols, density=0.12, seed=seed)
    # Stamp a clean glider in the top-left region so it can be detected.
    board[0:5, 0:5] = 0
    board[0:3, 0:3] = make_glider(0, 0, 3, 3)[0:3, 0:3]
    return board


@typechecked
def save_board(board: np.ndarray, path: str) -> None:
    """Pickle ``board`` to ``path`` (creating parent directories as needed)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(board, handle)
    log.info("Saved initial state (%dx%d) to %s", *board.shape, path)


@typechecked
def load_initial_state(path: str,
                       shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load and validate the board stored in the pickle at ``path``.

    Raises :class:`FileNotFoundError` when the file is missing so the caller can
    decide whether to generate a fresh state.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Initial state file not found: {path}")

    with open(path, "rb") as handle:
        board = pickle.load(handle)

    if not isinstance(board, np.ndarray):
        raise TypeError(
            f"initial.pkl must contain a numpy array, got {type(board)!r}")

    # Normalise to a boolean-like uint8 board of 0/1 values.
    board = (board != 0).astype(np.uint8)
    if shape is not None and board.shape != shape:
        log.warning(
            "initial.pkl shape %s differs from configured shape %s; "
            "using the file's shape.", board.shape, shape)
    log.info("Loaded initial state (%dx%d) from %s", *board.shape, path)
    return board
