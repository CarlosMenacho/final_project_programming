"""Real-time DearPyGui interface (requirements *d*, *e*).

The :class:`SimulationController` glues the engine, database and detectors
together, while :class:`GameOfLifeApp` renders the board as a live texture and
exposes pause/resume plus basic statistics.
"""
import logging
import time
from typing import Optional

import dearpygui.dearpygui as dpg
import numpy as np
from typeguard import typechecked

from src.database import Database
from src.detection import GliderDetector, StabilityDetector
from src.engine import GameEngine, Stats

log = logging.getLogger(__name__)

# RGBA colours (0..1) for dead and live cells plus the grid lines.
_DEAD_COLOR = np.array([0.09, 0.09, 0.12, 1.0], dtype=np.float32)
_LIVE_COLOR = np.array([0.20, 0.85, 0.45, 1.0], dtype=np.float32)
_GRID_COLOR = np.array([0.18, 0.18, 0.22, 1.0], dtype=np.float32)


@typechecked
class SimulationController:
    """Advance the simulation while persisting stats and running detectors."""

    def __init__(self,
                 engine: GameEngine,
                 database: Database,
                 stability: StabilityDetector,
                 gliders: GliderDetector,
                 max_iterations: int = 1000,
                 detect_gliders: bool = True) -> None:
        self.engine = engine
        self.database = database
        self.stability = stability
        self.gliders = gliders
        self.max_iterations = max_iterations
        self.detect_gliders = detect_gliders

        self.finished: bool = False
        self.stop_reason: str = ""
        self.last_stats: Optional[Stats] = None

        # Seed the stability detector with the initial board.
        self.stability.update(self.engine.board)

    def step(self) -> Stats:
        """Advance one generation and run persistence + detection."""
        stats = self.engine.step()
        self.last_stats = stats

        self.database.record(stats.iteration, stats.live_cells,
                             stats.dead_cells, stats.exec_time_ms)

        if self.detect_gliders:
            self.gliders.log_detections(stats.iteration, self.engine.board)

        is_stable, period = self.stability.update(self.engine.board)
        if is_stable:
            kind = "still life" if period in (
                0, 1) else f"oscillator (period {period})"
            self.stop_reason = f"Stable state reached: {kind}"
            self.finished = True
            log.info("%s at iteration %d - stopping simulation.",
                     self.stop_reason, stats.iteration)
        elif stats.live_cells == 0:
            self.stop_reason = "Board is empty"
            self.finished = True
            log.info("%s at iteration %d - stopping simulation.",
                     self.stop_reason, stats.iteration)
        elif stats.iteration >= self.max_iterations:
            self.stop_reason = "Maximum iterations reached"
            self.finished = True
            log.info("%s (%d) - stopping simulation.", self.stop_reason,
                     self.max_iterations)

        return stats


@typechecked
class GameOfLifeApp:
    """DearPyGui front-end driving a :class:`SimulationController`."""

    def __init__(self,
                 controller: SimulationController,
                 fps: float = 10.0,
                 cell_size: int = 12) -> None:
        self.controller = controller
        self.fps = max(0.5, fps)
        self.cell_size = max(1, cell_size)
        self.paused: bool = False

        board = controller.engine.board
        self.rows, self.cols = board.shape
        # Full-resolution texture dimensions (one square block per cell). The
        # texture is built at display size so DearPyGui renders it 1:1 and never
        # bilinearly upscales a tiny buffer, which is what made the grid blurry.
        self.tex_w = self.cols * self.cell_size
        self.tex_h = self.rows * self.cell_size
        # Draw grid lines only when cells are large enough for them to help.
        self._draw_grid = self.cell_size >= 4
        self._last_step_time: float = 0.0

    def _board_texture(self) -> np.ndarray:
        """Return the current board as a flat, full-resolution RGBA buffer.

        Each cell is expanded into a ``cell_size`` × ``cell_size`` block via
        nearest-neighbour upscaling (``np.repeat``) so cells stay crisp instead
        of being interpolated by the GPU.
        """
        board = self.controller.engine.board
        # Broadcast the two colours according to the live/dead mask.
        mask = board.astype(bool)[..., None]
        rgba = np.where(mask, _LIVE_COLOR, _DEAD_COLOR).astype(np.float32)
        # Nearest-neighbour upscale to display resolution.
        rgba = np.repeat(np.repeat(rgba, self.cell_size, axis=0),
                         self.cell_size, axis=1)
        if self._draw_grid:
            # Paint a 1px grid line along the top/left edge of every cell.
            rgba[::self.cell_size, :, :] = _GRID_COLOR
            rgba[:, ::self.cell_size, :] = _GRID_COLOR
        return np.ascontiguousarray(rgba).ravel()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        dpg.set_item_label("pause_button",
                           "Resume" if self.paused else "Pause")
        log.info("Simulation %s", "paused" if self.paused else "resumed")

    def _update_stats_text(self) -> None:
        stats = self.controller.last_stats
        if stats is None:
            return
        dpg.set_value("stat_iteration", f"Iteration: {stats.iteration}")
        dpg.set_value("stat_live", f"Live cells: {stats.live_cells}")
        dpg.set_value("stat_dead", f"Dead cells: {stats.dead_cells}")
        dpg.set_value("stat_delta", f"Born: {stats.born}   Died: {stats.died}")
        dpg.set_value("stat_time", f"Step time: {stats.exec_time_ms:.2f} ms")
        status = self.controller.stop_reason if self.controller.finished else (
            "Paused" if self.paused else "Running")
        dpg.set_value("stat_status", f"Status: {status}")

    def _build_ui(self) -> None:
        with dpg.texture_registry():
            dpg.add_dynamic_texture(width=self.tex_w,
                                    height=self.tex_h,
                                    default_value=self._board_texture(),
                                    tag="board_texture")

        with dpg.window(tag="main_window"):
            dpg.add_text("Conway's Game of Life", color=(120, 220, 160))
            with dpg.group(horizontal=True):
                dpg.add_button(label="Pause",
                               tag="pause_button",
                               callback=self._toggle_pause)
                dpg.add_button(label="Step",
                               callback=lambda: self._single_step())
            dpg.add_separator()
            dpg.add_text("Iteration: 0", tag="stat_iteration")
            dpg.add_text("Live cells: 0", tag="stat_live")
            dpg.add_text("Dead cells: 0", tag="stat_dead")
            dpg.add_text("Born: 0   Died: 0", tag="stat_delta")
            dpg.add_text("Step time: 0.00 ms", tag="stat_time")
            dpg.add_text("Status: Running", tag="stat_status")
            dpg.add_separator()
            # Display 1:1 with the texture so no scaling/filtering occurs.
            dpg.add_image("board_texture",
                          width=self.tex_w,
                          height=self.tex_h)

    def _single_step(self) -> None:
        """Advance exactly one generation (used by the Step button)."""
        if self.controller.finished:
            return
        self.controller.step()
        dpg.set_value("board_texture", self._board_texture())
        self._update_stats_text()

    def _tick(self) -> None:
        """Advance the simulation if enough time has elapsed and not paused."""
        if self.paused or self.controller.finished:
            return
        now = time.perf_counter()
        if now - self._last_step_time < 1.0 / self.fps:
            return
        self._last_step_time = now
        self.controller.step()
        dpg.set_value("board_texture", self._board_texture())
        self._update_stats_text()

    def run(self) -> None:
        """Create the context, build the UI and enter the render loop."""
        dpg.create_context()
        self._build_ui()
        width = max(360, self.cols * self.cell_size + 40)
        height = self.rows * self.cell_size + 220
        dpg.create_viewport(title="Game of Life", width=width, height=height)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            self._tick()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
