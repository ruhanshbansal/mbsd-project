"""
Branch Detector (BD) + Decompression Buffer — Paper Section 4.

This is the CORE CONTRIBUTION of the paper.

In a standard VLIW processor, the branch predictor is accessed every
cycle regardless of whether the current instruction is a branch. This
wastes significant energy on non-branch instructions (~75% of all
instructions in typical workloads).

The Branch Detector exploits the pre-fetch decompression buffer:

  1. Instructions are fetched from cache into the decompression buffer
  2. The BD performs a PARTIAL PRE-DECODE of each bundle in the buffer
  3. If a branch is detected => activate the branch predictor
  4. If no branch => predictor remains IDLE (saves energy)

Key properties:
  - The BD is DETERMINISTIC (not probabilistic, not random)
  - It adds only 2 logic levels to the critical path
  - The decompression buffer depth controls the energy-performance
    tradeoff (deeper = more lookahead = lower effective mispred penalty)

Buffer-empty scenario (paper Section 4):
  "When the decompression unit is empty, branches are directly sent to
   the decode unit and the branch detector cannot activate the branch
   predictor block."
  After a misprediction flush, the buffer must be refilled. During
  refilling, the BD is inactive and cannot filter predictor accesses.

References:
  - Palermo et al., "Branch Prediction Techniques for Low-Power VLIW
    Processors," GLSVLSI 2003, Section 4.
"""

from core.energy import ENERGY
from core.pipeline import ISSUE_WIDTH, MISPRED_PENALTY


class BranchDetector:
    """
    Branch Detector with decompression buffer.

    The BD scans each bundle in the decompression buffer to detect
    whether it contains a branch operation. This is a partial pre-decode
    that checks instruction opcodes without full decoding.

    Args:
        depth:      decompression buffer depth (1, 2, 4, or 8 entries)
        bd_enabled: if False, predictor is accessed every cycle (no filtering)
    """

    def __init__(self, depth=1, bd_enabled=False):
        self.depth = depth
        self.bd_enabled = bd_enabled

        # Deeper buffers provide more lookahead, allowing earlier
        # branch detection and reduced effective misprediction penalty.
        # Each extra entry of depth reduces penalty by 1 cycle.
        self.lookahead_reduction = min(depth - 1, MISPRED_PENALTY - 1)
        self.effective_penalty = max(1, MISPRED_PENALTY - self.lookahead_reduction)

    def should_access_predictor(self, bundle):
        """
        Determine if the branch predictor should be activated for this bundle.

        Without BD: always returns True (predictor accessed every cycle).
        With BD:    returns True only if bundle contains a branch.

        This is the key filtering mechanism that saves energy.
        """
        if not self.bd_enabled:
            return True  # No filtering — predictor accessed unconditionally
        return bundle['has_branch']

    def bd_energy_per_bundle(self):
        """Energy for BD to pre-decode one bundle."""
        return ENERGY['bd_predecode'] if self.bd_enabled else 0.0

    def buffer_sram_energy(self):
        """
        Static energy for the decompression buffer SRAM (per cycle).

        Deeper buffers consume more energy because they have more SRAM
        entries that are continuously powered. This creates the energy
        differentiation between buffer depths visible in paper Figure 4.
        """
        return self.depth * ENERGY['buffer_sram_per_entry']

    @property
    def refill_cycles(self):
        """
        Number of cycles to refill the decompression buffer after a
        misprediction flush. During refilling, the BD is inactive.
        """
        return self.depth

    @staticmethod
    def decompression_energy(bundle):
        """Energy to decompress a VLIW bundle (proportional to active ops)."""
        active_ops = sum(1 for op in bundle['ops'] if op != 'NOP')
        return active_ops * ENERGY['decompressor_per_op']

    def filter_rate(self, bundles):
        """
        Compute the fraction of bundles where the BD successfully filters
        out a predictor access (i.e., non-branch bundles).

        Higher filter rate = more energy saved.
        """
        if not self.bd_enabled or not bundles:
            return 0.0
        non_branch = sum(1 for b in bundles if not b['has_branch'])
        return non_branch / len(bundles)
