"""
Branch History Table (BHT) Predictor — 2-bit saturating counter.

One of the simplest dynamic branch predictors. Uses a table of 2-bit
saturating counters indexed by the lower bits of the branch PC.

States:
  0 = Strongly Not-Taken
  1 = Weakly Not-Taken
  2 = Weakly Taken
  3 = Strongly Taken

Prediction: counter >= 2 => Taken, else Not-Taken.

The 2-bit counter provides hysteresis: it takes two consecutive
mispredictions to change the prediction direction. This is effective
for loops (only 1 misprediction at loop exit instead of 2).

References:
  - Smith, "A Study of Branch Prediction Strategies," ISCA 1981.
    [Paper ref 4]
"""

from branch_prediction_sim.predictors.base import BasePredictor
from branch_prediction_sim.core.energy import ENERGY


class BHTPredictor(BasePredictor):
    """Branch History Table with 2-bit saturating counters."""

    def __init__(self, entries=256):
        """
        Args:
            entries: number of table entries (indexed by PC % entries)
        """
        self.entries = entries
        self.table = [1] * entries  # init to Weakly Not-Taken

    def predict(self, pc):
        return 1 if self.table[pc % self.entries] >= 2 else 0

    def update(self, pc, actual):
        idx = pc % self.entries
        if actual == 1:
            self.table[idx] = min(3, self.table[idx] + 1)
        else:
            self.table[idx] = max(0, self.table[idx] - 1)

    def read_energy(self):
        return ENERGY['bht_read']

    def write_energy(self):
        return ENERGY['bht_write']
