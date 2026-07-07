"""Application entry point wired through Hydra (requirements i, o).

Run with::

    python run_all.py

or override any config value from the command line, e.g.::

    python run_all.py simulation.fps=30 grid.rows=80 grid.cols=80
"""
import logging

import hydra
from omegaconf import DictConfig
from typeguard import typechecked

from src.board import default_initial_state, load_initial_state, save_board
from src.database import Database
from src.detection import GliderDetector, StabilityDetector
from src.engine import GameEngine
from src.gui import GameOfLifeApp, SimulationController

log = logging.getLogger(__name__)


@typechecked
def _obtain_initial_state(cfg: DictConfig):
    """Load ``initial.pkl`` or generate it when allowed by the config."""
    path = cfg.initial_state.path
    try:
        return load_initial_state(path, shape=(cfg.grid.rows, cfg.grid.cols))
    except FileNotFoundError:
        if not cfg.initial_state.auto_generate:
            raise
        log.warning("%s missing - generating a fresh initial state.", path)
        board = default_initial_state(cfg.grid.rows,
                                      cfg.grid.cols,
                                      seed=cfg.initial_state.seed)
        save_board(board, path)
        return board


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Build every component and run the simulation."""
    board = _obtain_initial_state(cfg)

    database = Database(url=cfg.database.url)
    stability = StabilityDetector(history=cfg.simulation.stability_history)
    gliders = GliderDetector()

    with GameEngine(board, workers=cfg.multiprocessing.workers) as engine:
        controller = SimulationController(
            engine=engine,
            database=database,
            stability=stability,
            gliders=gliders,
            max_iterations=cfg.simulation.max_iterations,
            detect_gliders=cfg.simulation.detect_gliders,
        )

        if cfg.gui.enabled:
            app = GameOfLifeApp(controller,
                                fps=cfg.simulation.fps,
                                cell_size=cfg.gui.cell_size)
            app.run()
        else:
            # Headless mode: run until a stopping condition is reached.
            log.info("Running headless (GUI disabled).")
            while not controller.finished:
                controller.step()

    log.info("Simulation finished: %s", controller.stop_reason or "GUI closed")


if __name__ == "__main__":
    main()
