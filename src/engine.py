"""The parallel Game-of-Life simulation engine.

The engine advances the board one generation at a time.  To satisfy
requirement *c* the expensive neighbour-counting step is split into horizontal
row-blocks and computed by a :class:`multiprocessing.Pool` of worker
processes.  Each block is handed a one-row "halo" above and below so that the
transition at block boundaries is computed correctly.
"""
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from multiprocessing import Pool
from typeguard import typechecked

log = logging.getLogger(__name__)


@typechecked
def _next_generation_block(padded_block: np.ndarray) -> np.ndarray:
    """Compute the next state for the interior rows of ``padded_block``.

    ``padded_block`` is a slice of the zero-padded board containing the block's
    own rows plus one halo row above and below and one halo column on each
    side.  The Game-of-Life rules are applied in a fully vectorised manner.

    This is a module-level function (not a closure/lambda) so it can be pickled
    and dispatched to worker processes.
    """
    # Sum of the eight neighbours for every interior cell.
    neighbours = (padded_block[:-2, :-2] + padded_block[:-2, 1:-1] +
                  padded_block[:-2, 2:] + padded_block[1:-1, :-2] +
                  padded_block[1:-1, 2:] + padded_block[2:, :-2] +
                  padded_block[2:, 1:-1] + padded_block[2:, 2:])

    current = padded_block[1:-1, 1:-1]
    # Live cell survives with 2 or 3 neighbours; dead cell is born with exactly 3.
    survives = (current == 1) & ((neighbours == 2) | (neighbours == 3))
    born = (current == 0) & (neighbours == 3)
    return (survives | born).astype(np.uint8)


@typechecked
@dataclass
class Stats:
    """Per-iteration statistics produced by :meth:`GameEngine.step`."""
    iteration: int
    live_cells: int
    dead_cells: int
    born: int
    died: int
    exec_time_ms: float


@typechecked
class GameEngine:
    """Owns the board and advances the simulation using a process pool."""

    def __init__(self, board: np.ndarray, workers: int = 4) -> None:
        self.board: np.ndarray = (board != 0).astype(np.uint8)
        self.rows, self.cols = self.board.shape
        self.iteration: int = 0
        # Never spawn more workers than there are rows to process.
        self.workers: int = max(1, min(workers, self.rows))
        self._pool: Optional[Pool] = None
        if self.workers > 1:
            self._pool = Pool(processes=self.workers)
            log.info("Engine started with %d worker processes", self.workers)
        else:
            log.info("Engine started in single-process mode")

    def _row_bounds(self) -> List[tuple]:
        """Split the rows into roughly equal ``(start, stop)`` blocks."""
        edges = np.linspace(0, self.rows, self.workers + 1).astype(int)
        return [(int(edges[i]), int(edges[i + 1])) for i in range(self.workers)
                if int(edges[i]) < int(edges[i + 1])]

    def _compute_next(self) -> np.ndarray:
        """Return the next generation, parallelising over row-blocks."""
        # Pad with a zero border so neighbour counting needs no edge cases.
        padded = np.pad(self.board, 1, mode="constant", constant_values=0)

        if self._pool is None:
            return _next_generation_block(padded)

        # Each block gets its rows plus the surrounding halo rows/columns.
        blocks = [
            padded[start:stop + 2, :] for start, stop in self._row_bounds()
        ]
        results = self._pool.map(_next_generation_block, blocks)
        return np.vstack(results)

    def step(self) -> Stats:
        """Advance one generation and return the resulting statistics."""
        start = time.perf_counter()
        new_board = self._compute_next()
        exec_time_ms = (time.perf_counter() - start) * 1000.0

        born = int(np.sum((self.board == 0) & (new_board == 1)))
        died = int(np.sum((self.board == 1) & (new_board == 0)))

        self.board = new_board
        self.iteration += 1

        live = int(np.sum(new_board))
        stats = Stats(
            iteration=self.iteration,
            live_cells=live,
            dead_cells=self.board.size - live,
            born=born,
            died=died,
            exec_time_ms=exec_time_ms,
        )
        return stats

    def close(self) -> None:
        """Tear down the worker pool.  Safe to call multiple times."""
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "GameEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
