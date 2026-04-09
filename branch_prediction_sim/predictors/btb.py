"""
Branch Target Buffer (BTB) Predictor.

A cache-like structure that stores both the predicted direction AND
the target address of branches. Indexed by branch PC, with tag matching
to handle aliasing.

BTB1: 1-bit direction (last outcome)
BTB2: 2-bit saturating counter for direction

The paper explores both BTB1 and BTB2 configurations across varying
sizes and associativities. BTB is generally larger than BHT/PHT due
to storing target addresses, but underperforms GAs in accuracy.

References:
  - Perleberg & Smith, "Branch target buffer design and optimization,"
    IEEE Transactions, 1993.  [Paper ref 9]
  - Fagan & Russell, "Partial Resolution in Branch Target Buffers,"
    MICRO, 1995.  [Paper ref 12]
"""

from predictors.base import BasePredictor
from core.energy import ENERGY


class BTBEntry:
    """Single BTB entry: tag, target, and direction counter."""
    __slots__ = ['valid', 'tag', 'target', 'counter']

    def __init__(self):
        self.valid = False
        self.tag = 0
        self.target = 0
        self.counter = 1  # Weakly Not-Taken


class BTBPredictor(BasePredictor):
    """
    Branch Target Buffer with configurable direction bits.

    Args:
        entries:   number of BTB entries
        counter_bits: 1 for BTB1, 2 for BTB2
    """

    def __init__(self, entries=64, counter_bits=2):
        self.entries = entries
        self.counter_bits = counter_bits
        self.max_counter = (1 << counter_bits) - 1
        self.threshold = self.max_counter // 2 + 1  # >= threshold => taken
        self.table = [BTBEntry() for _ in range(entries)]

    def _decompose(self, pc):
        """Split PC into index and tag."""
        idx = pc % self.entries
        tag = pc // self.entries
        return idx, tag

    def predict(self, pc):
        idx, tag = self._decompose(pc)
        entry = self.table[idx]
        if entry.valid and entry.tag == tag:
            return 1 if entry.counter >= self.threshold else 0
        return 0  # miss => predict not-taken

    def update(self, pc, actual):
        idx, tag = self._decompose(pc)
        entry = self.table[idx]

        if not entry.valid or entry.tag != tag:
            # Allocate new entry
            entry.valid = True
            entry.tag = tag
            entry.target = pc + 1  # simplified target
            entry.counter = self.threshold if actual == 1 else self.threshold - 1
        else:
            # Update existing entry
            if actual == 1:
                entry.counter = min(self.max_counter, entry.counter + 1)
            else:
                entry.counter = max(0, entry.counter - 1)

    def read_energy(self):
        return ENERGY['btb_read']

    def write_energy(self):
        return ENERGY['btb_write']
