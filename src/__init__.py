"""Conway's Game of Life - a modular, parallelized implementation.

The package is organized in focused modules:

* :mod:`src.board`      -- initial-state loading/generation helpers.
* :mod:`src.engine`     -- the multiprocessing simulation engine.
* :mod:`src.detection`  -- stable-state and glider detectors.
* :mod:`src.database`   -- SQLAlchemy persistence layer.
* :mod:`src.gui`        -- the DearPyGui real-time interface.
"""
