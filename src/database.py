"""SQLAlchemy persistence layer (requirement *g*).

Every iteration is stored as one :class:`IterationRecord` row holding the
iteration number, the live/dead cell counts and the wall-clock time the
iteration took, in milliseconds.
"""
import logging
import os
from typing import Optional

from sqlalchemy import Column, Float, Integer, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from typeguard import typechecked

log = logging.getLogger(__name__)

Base = declarative_base()


class IterationRecord(Base):
    """One persisted generation of the simulation."""

    __tablename__ = "iterations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    iteration = Column(Integer, nullable=False, index=True)
    live_cells = Column(Integer, nullable=False)
    dead_cells = Column(Integer, nullable=False)
    exec_time_ms = Column(Float, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (f"<IterationRecord iter={self.iteration} "
                f"live={self.live_cells} dead={self.dead_cells} "
                f"time={self.exec_time_ms:.3f}ms>")


@typechecked
class Database:
    """Thin wrapper around a SQLAlchemy engine/session for the simulation."""

    def __init__(self, url: str = "sqlite:///outputs/game_of_life.db") -> None:
        # Make sure the parent directory of a file-based SQLite DB exists.
        if url.startswith("sqlite:///"):
            db_path = url.replace("sqlite:///", "", 1)
            directory = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(directory, exist_ok=True)

        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)
        self._Session = sessionmaker(bind=self.engine, future=True)
        log.info("Database ready at %s", url)

    def record(self, iteration: int, live_cells: int, dead_cells: int,
               exec_time_ms: float) -> None:
        """Insert a single iteration record."""
        with self._Session() as session:
            session.add(
                IterationRecord(
                    iteration=iteration,
                    live_cells=live_cells,
                    dead_cells=dead_cells,
                    exec_time_ms=exec_time_ms,
                ))
            session.commit()

    def count(self) -> int:
        """Return how many iteration records are currently stored."""
        with self._Session() as session:
            return session.query(IterationRecord).count()

    def last(self) -> Optional[IterationRecord]:
        """Return the most recently stored record, if any."""
        with self._Session() as session:
            return (session.query(IterationRecord).order_by(
                IterationRecord.iteration.desc()).first())
