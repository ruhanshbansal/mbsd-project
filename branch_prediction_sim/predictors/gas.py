"""
GAs (Global Adaptive Set) Predictor — two-level adaptive.

A correlating branch predictor that uses both the branch PC and the
global branch history to make predictions:

  1. Global History Register (GHR): shift register recording the last
     N branch outcomes (taken=1, not-taken=0).

  2. Pattern History Table (PHT): indexed by hash(PC, GHR).
     Each PHT entry is a 2-bit saturating counter.

The GAs predictor captures correlation between different branches.
The paper finds GAs consistently outperforms BHT for the Lx architecture.

References:
  - Yeh & Patt, "Two-level adaptive training branch prediction,"
    MICRO-24, 1991.  [Paper ref 7]
  - Chang & Choi, "Branch prediction using global and local branch
    history information," IEEE CDT, 2002.  [Paper ref 6]
"""

from branch_prediction_sim.predictors.base import BasePredictor
from branch_prediction_sim.core.energy import ENERGY


class GAsPredictor(BasePredictor):
    """GAs (Global Adaptive Set) two-level adaptive predictor."""

    def __init__(self, ghr_bits=12, pht_sets=64, pht_entries_per_set=16):
        """
        Args:
            ghr_bits:           width of the Global History Register
            pht_sets:           number of PHT sets (indexed by PC)
            pht_entries_per_set: entries per set (indexed by GHR)

        Default: 12-bit GHR, 64 sets × 16 entries = 1024 total PHT entries.
        Within the paper's explored design space (up to 1024 entries).
        """
        self.ghr_bits = ghr_bits
        self.ghr = 0
        self.pht_sets = pht_sets
        self.pht_entries = pht_entries_per_set
        # Two-level: set selected by PC, entry within set selected by GHR
        self.pht = [[1] * pht_entries_per_set for _ in range(pht_sets)]

    def _index(self, pc):
        """Compute (set_index, entry_index) from PC and GHR."""
        # Spread PC bits better to reduce aliasing for close PCs
        pc_hash = ((pc >> 2) ^ (pc >> 7) ^ pc) % self.pht_sets
        ghr_idx = self.ghr % self.pht_entries
        return pc_hash, ghr_idx

    def predict(self, pc):
        s, e = self._index(pc)
        return 1 if self.pht[s][e] >= 2 else 0

    def update(self, pc, actual):
        s, e = self._index(pc)
        if actual == 1:
            self.pht[s][e] = min(3, self.pht[s][e] + 1)
        else:
            self.pht[s][e] = max(0, self.pht[s][e] - 1)
        # Shift actual outcome into GHR
        self.ghr = ((self.ghr << 1) | actual) & ((1 << self.ghr_bits) - 1)

    def read_energy(self):
        return ENERGY['pht_read'] + ENERGY['ghr_update']

    def write_energy(self):
        return ENERGY['pht_write'] + ENERGY['ghr_update']
