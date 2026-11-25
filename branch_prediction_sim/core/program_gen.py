"""Generate seeded synthetic VLIW bundles with five branch-pattern families.

These traces are an educational workload, not captured MediaBench traces.
Stall probabilities and penalties are fixed model assumptions. Branch density
emerges from the segment mix; it is not an independently controlled input.
"""

import random
from branch_prediction_sim.core.pipeline import (ISSUE_WIDTH, DCACHE_MISS_RATE, DCACHE_MISS_PENALTY,
                           DATA_HAZARD_RATE, DATA_HAZARD_PENALTY,
                           ICACHE_MISS_RATE, ICACHE_MISS_PENALTY)


class TraceGenerator:
    """
    Generates a synthetic instruction trace as a sequence of VLIW bundles.

    Each bundle is a dict with:
        - 'ops':             list of op types ['ALU','MEM','BRANCH','NOP']
        - 'has_branch':      bool
        - 'branch_pc':       int (PC of branch, reused for same static branch)
        - 'branch_outcome':  0 (not-taken) or 1 (taken)
        - 'pattern_type':    str ('loop','biased_if','correlated','alternating','random')
        - 'stall_cycles':    int (pipeline stalls from hazards / cache misses)
    """

    def __init__(self, num_bundles=5000, seed=42):
        """
        Args:
            num_bundles:    total VLIW bundles in the trace
            seed:           random seed for reproducibility
        """
        if not isinstance(num_bundles, int) or isinstance(num_bundles, bool) or num_bundles < 1:
            raise ValueError("num_bundles must be a positive integer")
        self.seed = seed
        self.rng = random.Random(seed)
        self.num_bundles = num_bundles

    def generate(self):
        """Generate the full instruction trace."""
        self.rng.seed(self.seed)
        bundles = []
        while len(bundles) < self.num_bundles:
            for _, segment in self._generate_segments():
                remaining = self.num_bundles - len(bundles)
                bundles.extend(segment[:remaining])
                if len(bundles) == self.num_bundles:
                    break
        return bundles

    # ── Bundle builders ──────────────────────────────────────────────────

    def _make_bundle(self, has_branch, pc=None, outcome=None, pattern_type=None):
        """Create a single VLIW bundle."""
        ops = []
        if has_branch:
            branch_slot = self.rng.randint(0, ISSUE_WIDTH - 1)
            for s in range(ISSUE_WIDTH):
                if s == branch_slot:
                    ops.append('BRANCH')
                else:
                    r = self.rng.random()
                    ops.append('ALU' if r < 0.45 else ('MEM' if r < 0.75 else 'NOP'))
        else:
            for s in range(ISSUE_WIDTH):
                r = self.rng.random()
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
        predictor configurations equally, keeping comparisons on the same trace.
        """
        stall = 0

        # D-cache miss per MEM operation
        mem_ops = sum(1 for op in ops if op == 'MEM')
        for _ in range(mem_ops):
            if self.rng.random() < DCACHE_MISS_RATE:
                stall += DCACHE_MISS_PENALTY

        # Data hazard stall (RAW dependency between ops in adjacent bundles)
        if self.rng.random() < DATA_HAZARD_RATE:
            stall += DATA_HAZARD_PENALTY

        # I-cache miss
        if self.rng.random() < ICACHE_MISS_RATE:
            stall += ICACHE_MISS_PENALTY

        return stall

    # ── Segment generators ───────────────────────────────────────────────

    def _generate_segments(self):
        """Build synthetic program segments with recurring branch patterns."""
        segments = []
        pc = 100

        # --- LOOPS (most common in multimedia, highly predictable) ---
        for _ in range(20):
            body_len = self.rng.randint(3, 12)
            iterations = self.rng.randint(5, 80)
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
            length = self.rng.randint(10, 30)
            bias = self.rng.choice([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])
            if_pc = pc
            seg = []
            for _ in range(length):
                for _ in range(self.rng.randint(1, 4)):
                    seg.append(self._make_bundle(False))
                outcome = 1 if self.rng.random() < bias else 0
                seg.append(self._make_bundle(True, pc=if_pc, outcome=outcome,
                                             pattern_type='biased_if'))
            pc += 20
            segments.append(('biased_if', seg))

        # --- CORRELATED (branch B depends on branch A) ---
        for _ in range(15):
            length = self.rng.randint(15, 40)
            pc_a, pc_b = pc, pc + 5
            seg = []
            prev = self.rng.choice([0, 1])
            for _ in range(length):
                seg.append(self._make_bundle(False))
                seg.append(self._make_bundle(True, pc=pc_a, outcome=prev,
                                             pattern_type='correlated'))
                seg.append(self._make_bundle(False))
                b_out = (1 if self.rng.random() < 0.85 else 0) if prev == 1 \
                    else (0 if self.rng.random() < 0.80 else 1)
                seg.append(self._make_bundle(True, pc=pc_b, outcome=b_out,
                                             pattern_type='correlated'))
                prev = b_out
            pc += 20
            segments.append(('correlated', seg))

        # --- ALTERNATING (even/odd processing) ---
        for _ in range(10):
            length = self.rng.randint(20, 50)
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
            length = self.rng.randint(15, 30)
            rnd_pc = pc
            seg = []
            for _ in range(length):
                for _ in range(self.rng.randint(1, 3)):
                    seg.append(self._make_bundle(False))
                seg.append(self._make_bundle(True, pc=rnd_pc,
                                             outcome=self.rng.choice([0, 1]),
                                             pattern_type='random'))
            pc += 20
            segments.append(('random', seg))

        self.rng.shuffle(segments)
        return segments
