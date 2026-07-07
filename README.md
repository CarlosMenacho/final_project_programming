# Conway's Game of Life — Parallel Edition

A modular implementation of Conway's Game of Life featuring a NumPy engine that
distributes generation computation across multiple processes, a real-time
DearPyGui interface, SQLite persistence via SQLAlchemy, and colourised logging
through Hydra.

## Rules

| Condition | Outcome |
|-----------|---------|
| Live cell with **< 2** live neighbours | dies (underpopulation) |
| Live cell with **2 or 3** live neighbours | survives |
| Live cell with **> 3** live neighbours | dies (overpopulation) |
| Dead cell with **exactly 3** live neighbours | becomes alive (reproduction) |

## Features

- **NumPy** board representation and fully vectorised transition rules.
- **Multiprocessing**: the board is split into horizontal row-blocks (each with
  a one-row halo) and the next generation is computed in parallel.
- **DearPyGui** GUI showing the board in real time with live statistics
  (iteration, live/dead counts, born/died, per-step time) and **pause/resume**.
- Initial state loaded from **`initial.pkl`** (auto-generated on first run).
- **SQLite + SQLAlchemy** persistence: every iteration stores the iteration
  number, live/dead cell counts and the step execution time in milliseconds.
- **Stable-state detection**: the run stops on still lifes, empty boards and
  short-period oscillators.
- **Glider detection** logged with a colourised Hydra logger.
- Type-checked at runtime with `@typechecked`, documented per Python
  conventions and formatted with **yapf**.

## Project layout

```
.
├── config/
│   └── default.yaml        # Hydra configuration
├── src/
│   ├── board.py            # initial-state load / generate
│   ├── engine.py           # parallel simulation engine
│   ├── detection.py        # stability + glider detectors
│   ├── database.py         # SQLAlchemy persistence
│   └── gui.py              # DearPyGui interface + controller
├── generate_initial.py     # standalone initial.pkl generator
├── run_all.py              # Hydra entry point
├── environment.yml         # conda environment definition
└── README.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate gol-fast
```

## Usage

Generate an initial state (optional — created automatically if missing):

```bash
python generate_initial.py --rows 40 --cols 40
```

Run the simulation with the GUI:

```bash
python run_all.py
```

Any configuration value can be overridden on the command line:

```bash
# Larger board, faster playback, more workers
python run_all.py grid.rows=80 grid.cols=80 simulation.fps=30 multiprocessing.workers=8

# Headless run (no window), useful for benchmarking / CI
python run_all.py gui.enabled=false
```

### Controls

- **Pause / Resume** — toggle the simulation.
- **Step** — advance a single generation while paused.

## Data

Iteration records are written to `outputs/game_of_life.db`. Inspect them with:

```bash
sqlite3 outputs/game_of_life.db "SELECT * FROM iterations LIMIT 10;"
```

## Configuration reference (`config/default.yaml`)

| Key | Meaning |
|-----|---------|
| `grid.rows`, `grid.cols` | board size for generated states |
| `initial_state.path` | pickle file to load the seed from |
| `simulation.max_iterations` | hard stop for the run |
| `simulation.fps` | target generations per second in the GUI |
| `simulation.stability_history` | oscillator period detected up to this value |
| `multiprocessing.workers` | number of worker processes |
| `database.url` | SQLAlchemy database URL |
| `gui.enabled`, `gui.cell_size` | GUI toggle and cell pixel size |

## Development

Format the code with yapf:

```bash
yapf -ir src run_all.py generate_initial.py
```
