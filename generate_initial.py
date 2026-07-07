"""Standalone helper to (re)create the ``initial.pkl`` seed file.

Usage::

    python generate_initial.py                 # 40x40 demo board
    python generate_initial.py --rows 80 --cols 80 --density 0.2
"""
import argparse

from typeguard import typechecked

from src.board import default_initial_state, random_board, save_board


@typechecked
def build(rows: int, cols: int, density: float, seed: int, demo: bool) -> None:
    """Create a board and pickle it to ``initial.pkl``."""
    if demo:
        board = default_initial_state(rows, cols, seed=seed)
    else:
        board = random_board(rows, cols, density=density, seed=seed)
    save_board(board, "initial.pkl")
    print(f"Wrote initial.pkl with {int(board.sum())} live cells "
          f"on a {rows}x{cols} board.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate initial.pkl")
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=40)
    parser.add_argument("--density", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random",
                        action="store_true",
                        help="pure random board instead of the demo board")
    args = parser.parse_args()
    build(args.rows, args.cols, args.density, args.seed, demo=not args.random)


if __name__ == "__main__":
    main()
