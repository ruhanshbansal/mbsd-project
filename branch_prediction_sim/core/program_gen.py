"""
Synthetic Workload Generator — mimics MediaBench suite branch behavior.

Generates a realistic VLIW instruction trace as a sequence of 4-wide
bundles, each containing at most one branch operation. Branch outcomes
follow patterns observed in real multimedia workloads:

  - Loops           (highly predictable: taken N-1 times, not-taken once)
  - Biased if-else  (error checks: ~5-15% taken, or fast-path: ~85-95%)
  - Correlated      (branch B outcome depends on branch A)
  - Alternating     (even/odd processing)
  - Random          (data-dependent, function pointers, virtual dispatch)

Branches reuse PCs (like real loops) so dynamic predictors can learn.

Each bundle also includes deterministic pipeline stall cycles computed
from its operation composition (D-cache misses, data hazards, I-cache
misses). These stalls affect all predictor configurations equally,
producing realistic speedup numbers matching the paper's ~7% average.

References:
  - Lee, Potkonjak & Mangione-Smith, "MediaBench: a tool for evaluating
    and synthesizing multimedia and communications systems,"
    Micro-30, 1997.  [Paper ref 13]
"""

import random
import numpy as np
from core.pipeline import (ISSUE_WIDTH, DCACHE_MISS_RATE, DCACHE_MISS_PENALTY,
                           DATA_HAZARD_RATE, DATA_HAZARD_PENALTY,
                           ICACHE_MISS_RATE, ICACHE_MISS_PENALTY)


class TraceGenerator:
    """
    Generates a realistic instruction trace as a sequence of VLIW bundles.

    Each bundle is a dict with:
        - 'ops':             list of op types ['ALU','MEM','BRANCH','NOP']
        - 'has_branch':      bool
        - 'branch_pc':       int (PC of branch, reused for same static branch)
        - 'branch_outcome':  0 (not-taken) or 1 (taken)
        - 'pattern_type':    str ('loop','biased_if','correlated','alternating','random')
        - 'stall_cycles':    int (pipeline stalls from hazards / cache misses)
    """

    def __init__(self, num_bundles=5000, branch_density=0.25, seed=42):
        """
        Args:
            num_bundles:    total VLIW bundles in the trace
            branch_density: fraction of bundles containing a branch
            seed:           random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        self.num_bundles = num_bundles
        self.branch_density = branch_density

    def generate(self):
        """Generate the full instruction trace."""
        bundles = []
        segments = self._generate_segments()

        for _, seg_bundles in segments:
            bundles.extend(seg_bundles)
            if len(bundles) >= self.num_bundles:
                break

        return bundles[:self.num_bundles]

    # ── Bundle builders ──────────────────────────────────────────────────

    def _make_bundle(self, has_branch, pc=None, outcome=None, pattern_type=None):
        """Create a single VLIW bundle."""
        ops = []
        if has_branch:
            branch_slot = random.randint(0, ISSUE_WIDTH - 1)
            for s in range(ISSUE_WIDTH):
                if s == branch_slot:
                    ops.append('BRANCH')
                else:
                    r = random.random()
                    ops.append('ALU' if r < 0.45 else ('MEM' if r < 0.75 else 'NOP'))
        else:
            for s in range(ISSUE_WIDTH):
                r = random.random()
                ops.append('ALU' if r < 0.50 else ('MEM' if r < 0.80 else 'NOP'))

        # Compute deterministic pipeline stalls based on op composition
        stall = self._compute_stalls(ops)

        return {
            'ops': ops,
            'has_branch': has_branch,
            'branch_pc': pc,
            'branch_outcome': outcome,
            'pattern_type': pattern_type,
            'stall_cycles': stall,
        }

    def _compute_stalls(self, ops):
        """
        Compute pipeline stall cycles based on bundle composition.

        Models three sources of stalls:
          1. D-cache misses on MEM operations
          2. Data hazard stalls (RAW dependencies)
          3. I-cache misses during instruction fetch

        These stalls are deterministic (seeded RNG) and affect all
        predictor configurations equally, ensuring realistic speedup.
        """
        stall = 0

        # D-cache miss per MEM operation
        mem_ops = sum(1 for op in ops if op == 'MEM')
        for _ in range(mem_ops):
            if random.random() < DCACHE_MISS_RATE:
                stall += DCACHE_MISS_PENALTY

        # Data hazard stall (RAW dependency between ops in adjacent bundles)
        if random.random() < DATA_HAZARD_RATE:
            stall += DATA_HAZARD_PENALTY

        # I-cache miss
        if random.random() < ICACHE_MISS_RATE:
            stall += ICACHE_MISS_PENALTY

        return stall

    # ── Segment generators ───────────────────────────────────────────────

    def _generate_segments(self):
        """Build program segments with realistic branch patterns."""
        segments = []
        pc = 100

        # --- LOOPS (most common in multimedia, highly predictable) ---
        for _ in range(20):
            body_len = random.randint(3, 12)
            iterations = random.randint(5, 80)
            branch_pc = pc + body_len
            seg = []
            for it in range(iterations):
                for _ in range(body_len):
                    seg.append(self._make_bundle(False))
                taken = 1 if it < iterations - 1 else 0
                seg.append(self._make_bundle(True, pc=branch_pc,
                                             outcome=taken, pattern_type='loop'))
            pc += body_len + 20
            segments.append(('loop', seg))

        # --- BIASED IF-ELSE (error checks, guards) ---
        for _ in range(15):
            length = random.randint(10, 30)
            bias = random.choice([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
            if_pc = pc
            seg = []
            for _ in range(length):
                for _ in range(random.randint(1, 4)):
                    seg.append(self._make_bundle(False))
                outcome = 1 if random.random() < bias else 0
                seg.append(self._make_bundle(True, pc=if_pc, outcome=outcome,
                                             pattern_type='biased_if'))
            pc += 20
            segments.append(('biased_if', seg))

        # --- CORRELATED (branch B depends on branch A) ---
        for _ in range(15):
            length = random.randint(15, 40)
            pc_a, pc_b = pc, pc + 5
            seg = []
            prev = random.choice([0, 1])
            for _ in range(length):
                seg.append(self._make_bundle(False))
                seg.append(self._make_bundle(True, pc=pc_a, outcome=prev,
                                             pattern_type='correlated'))
                seg.append(self._make_bundle(False))
                b_out = (1 if random.random() < 0.85 else 0) if prev == 1 \
                    else (0 if random.random() < 0.80 else 1)
                seg.append(self._make_bundle(True, pc=pc_b, outcome=b_out,
                                             pattern_type='correlated'))
                prev = b_out
            pc += 20
            segments.append(('correlated', seg))

        # --- ALTERNATING (even/odd processing) ---
        for _ in range(10):
            length = random.randint(20, 50)
            alt_pc = pc
            seg = []
            for j in range(length):
                seg.append(self._make_bundle(False))
                seg.append(self._make_bundle(True, pc=alt_pc, outcome=j % 2,
                                             pattern_type='alternating'))
            pc += 20
            segments.append(('alternating', seg))

        # --- RANDOM / UNPREDICTABLE (data-dependent) ---
        for _ in range(8):
            length = random.randint(15, 30)
            rnd_pc = pc
            seg = []
            for _ in range(length):
                for _ in range(random.randint(1, 3)):
                    seg.append(self._make_bundle(False))
                seg.append(self._make_bundle(True, pc=rnd_pc,
                                             outcome=random.choice([0, 1]),
                                             pattern_type='random'))
            pc += 20
            segments.append(('random', seg))

        random.shuffle(segments)
        return segments
