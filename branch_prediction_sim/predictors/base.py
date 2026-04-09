"""
Abstract base class for all branch predictors.

Every predictor must implement:
  - predict(pc)      -> 0 (not-taken) or 1 (taken)
  - update(pc, actual) -> update internal state after resolution
  - read_energy()    -> energy cost of one prediction lookup
  - write_energy()   -> energy cost of one state update
"""

from abc import ABC, abstractmethod


class BasePredictor(ABC):
    """Abstract interface for a branch predictor."""

    @abstractmethod
    def predict(self, pc: int) -> int:
        """Predict branch direction. Returns 0 (NT) or 1 (T)."""
        ...

    @abstractmethod
    def update(self, pc: int, actual: int) -> None:
        """Update predictor state after branch resolution."""
        ...

    @abstractmethod
    def read_energy(self) -> float:
        """Energy cost of one prediction lookup (read)."""
        ...

    @abstractmethod
    def write_energy(self) -> float:
        """Energy cost of one state update (write)."""
        ...
