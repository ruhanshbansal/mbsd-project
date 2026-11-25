"""
Instruction Set Simulator (ISS) — cycle-approximate VLIW pipeline simulator.

Models the Lx 4-issue, 6-stage VLIW pipeline with:
  - Bundle-level execution (VLIW semantics)
  - Configurable branch predictor (pluggable via BasePredictor)
  - Optional Branch Detector (BD) filtering
  - Decompression buffer with variable depth and lookahead
  - Buffer-empty state after misprediction flushes
  - Pipeline stalls from data hazards and cache misses
  - Component-level energy accounting
  - Per-pattern accuracy tracking

The paper uses an augmented ISS of the Lx processor with plugged-in
power models for the core, decompressor, and branch predictor.

References:
  - Paper Section 5: "The experimental results have been obtained by
    using an augmented Instruction Set Simulator of the Lx architecture."
  - Paper ref [14]: Bosca et al., "Energy Estimation of Embedded VLIW
    Processors Based on Instruction Clustering," DATE 2002.
"""

from collections import defaultdict
from branch_prediction_sim.core.pipeline import PIPELINE_STAGES
from branch_prediction_sim.core.energy import ENERGY, compute_flush_energy
from branch_prediction_sim.bd.detector import BranchDetector


class ISS:
    """
    Cycle-approximate Instruction Set Simulator for Lx VLIW.

    Processes a trace of VLIW bundles and produces cycle counts,
    prediction statistics, and component-level energy breakdown.
    """

    def __init__(self, predictor, buffer_depth=1, bd_enabled=False):
        """
        Args:
            predictor:    branch predictor instance (BasePredictor)
            buffer_depth: decompression buffer depth (1, 2, 4, 8)
            bd_enabled:   enable Branch Detector filtering
        """
        self.predictor = predictor
        self.bd = BranchDetector(depth=buffer_depth, bd_enabled=bd_enabled)

        # Simulation counters
        self.cycles = 0
        self.branches = 0
        self.correct = 0
        self.mispredictions = 0
        self.pred_accesses = 0
        self.pred_wasted = 0  # accesses on non-branch bundles
        self.stall_cycles_total = 0

        # Buffer-empty state tracking
        # After a misprediction flush, the decompression buffer is empty
        # and needs 'depth' cycles to refill. During refilling, the BD
        # cannot function (paper Section 4).
        self.buffer_fill_remaining = 0
        self.buffer_refills = 0
        self.bd_inactive_branches = 0  # branches hit during buffer refill

        # Per-pattern accuracy tracking
        self.pattern_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        # Energy accumulators
        self.e_core = 0.0
        self.e_predictor = 0.0
        self.e_decomp = 0.0
        self.e_bd = 0.0
        self.e_flush = 0.0
        self.e_buffer = 0.0

    def simulate(self, bundles):
        """
        Run simulation over a trace of VLIW bundles.

        Args:
            bundles: list of bundle dicts from TraceGenerator
        """
        for bundle in bundles:
            self.cycles += 1

            # ── Pipeline stalls (affect all configurations equally) ──
            stall = bundle.get('stall_cycles', 0)
            self.cycles += stall
            self.stall_cycles_total += stall

            # ── Core execution energy (incurred every cycle, inc. stalls) ──
            self.e_core += ENERGY['core_per_cycle'] * (1 + stall)

            # ── Decompression energy (per bundle, not per stall) ──
            self.e_decomp += self.bd.decompression_energy(bundle)

            # ── Buffer SRAM static energy (always on, inc. stalls) ──
            self.e_buffer += self.bd.buffer_sram_energy() * (1 + stall)

            # ── Buffer-filling state check ───────────────────────────
            buffer_active = self.buffer_fill_remaining <= 0
            if self.buffer_fill_remaining > 0:
                self.buffer_fill_remaining -= 1

            # ── BD filtering decision ────────────────────────────────
            if self.bd.bd_enabled and buffer_active:
                # BD is active: filter predictor accesses (saves energy)
                self.e_bd += self.bd.bd_energy_per_bundle()
                access = self.bd.should_access_predictor(bundle)
            elif not self.bd.bd_enabled:
                # No BD: always access predictor
                access = True
            else:
                # BD enabled but buffer is refilling (empty after flush)
                # Paper: "the branch detector cannot activate the branch
                # predictor block" → BD filtering is disabled.
                # The predictor IS still accessed (unconditionally, like
                # the no-BD case), losing the BD's energy-saving filtering
                # benefit but maintaining prediction accuracy.
                access = True
                if bundle['has_branch']:
                    self.bd_inactive_branches += 1

            # ── Branch processing ────────────────────────────────────
            if bundle['has_branch']:
                self.branches += 1
                pc = bundle['branch_pc']
                actual = bundle['branch_outcome']
                pattern = bundle.get('pattern_type', 'unknown')

                if access:
                    self.pred_accesses += 1
                    prediction = self.predictor.predict(pc)
                    self.e_predictor += self.predictor.read_energy()
                else:
                    prediction = 0  # default: not-taken (no predictor access)

                # Evaluate prediction
                if prediction == actual:
                    self.correct += 1
                    self.pattern_stats[pattern]['correct'] += 1
                else:
                    self.mispredictions += 1
                    penalty = self.bd.effective_penalty
                    self.cycles += penalty
                    self.e_flush += compute_flush_energy(penalty, PIPELINE_STAGES)

                    # After flush: buffer is empty, needs refilling
                    if self.bd.bd_enabled:
                        self.buffer_fill_remaining = self.bd.refill_cycles
                        self.buffer_refills += 1

                self.pattern_stats[pattern]['total'] += 1

                # Update predictor state (always, even if we didn't predict)
                self.predictor.update(pc, actual)
                self.e_predictor += self.predictor.write_energy()

            else:
                # Non-branch bundle: without BD, predictor is still accessed
                if access:
                    self.pred_accesses += 1
                    self.pred_wasted += 1
                    self.e_predictor += self.predictor.read_energy()

    def results(self):
        """Return a dict of all simulation metrics."""
        total_energy = (self.e_core + self.e_predictor + self.e_decomp +
                        self.e_bd + self.e_flush + self.e_buffer)
        accuracy = self.correct / self.branches if self.branches > 0 else None
        misses_per_kilo_branches = (self.mispredictions / self.branches * 1000
                if self.branches > 0 else None)

        return {
            'cycles':             self.cycles,
            'branches':           self.branches,
            'correct':            self.correct,
            'mispredictions':     self.mispredictions,
            'accuracy':           accuracy,
            'mispredictions_per_kilo_branches': misses_per_kilo_branches,
            'pred_accesses':      self.pred_accesses,
            'pred_wasted':        self.pred_wasted,
            'stall_cycles':       self.stall_cycles_total,
            'eff_penalty':        self.bd.effective_penalty,
            'buffer_refills':     self.buffer_refills,
            'bd_inactive_branches': self.bd_inactive_branches,
            'energy_total':       total_energy,
            'energy_core':        self.e_core,
            'energy_predictor':   self.e_predictor,
            'energy_decomp':      self.e_decomp,
            'energy_bd':          self.e_bd,
            'energy_flush':       self.e_flush,
            'energy_buffer':      self.e_buffer,
            'pattern_stats':      dict(self.pattern_stats),
        }
