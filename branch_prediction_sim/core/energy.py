"""
Energy Model — CACTI-inspired, component-level energy accounting.

All values are in relative (normalized) units, where 1.0 = baseline core
energy per cycle. This matches the paper's methodology where energy is
reported as a ratio relative to the reference Lx configuration.

References:
  - Wilton & Jouppi, "CACTI: Enhanced Cache Access and Cycle Time Model,"
    IEEE JSSC, 31(5):677-688, 1996.  [Paper ref 15]
  - Bosca et al., "Energy Estimation of Embedded VLIW Processors Based
    on Instruction Clustering," DATE 2002.  [Paper ref 14]
"""


# =============================================================================
# Component-level energy costs (relative units)
# =============================================================================
ENERGY = {
    # Core
    'core_per_cycle':       1.0,

    # BHT predictor (small SRAM array)
    'bht_read':             0.06,
    'bht_write':            0.08,

    # GAs predictor (PHT + GHR)
    'pht_read':             0.09,
    'pht_write':            0.11,
    'ghr_update':           0.01,

    # BTB (tag + data SRAM, larger than BHT)
    'btb_read':             0.12,
    'btb_write':            0.15,

    # Branch Detector pre-decode logic
    'bd_predecode':         0.02,

    # Decompressor
    'decompressor_per_op':  0.008,

    # Decompression buffer SRAM (static power, scales with depth)
    # Deeper buffers use more SRAM entries → more energy per cycle.
    # This creates the differentiation between depth=4 and depth=8
    # visible in paper Figure 4.
    'buffer_sram_per_entry': 0.015,

    # Pipeline flush on misprediction
    'flush_per_stage':      0.20,

    # Perceptron predictor (dot product + weight update)
    'perceptron_inference': 0.10,
    'perceptron_update':    0.12,

    # I-cache miss energy overhead
    'icache_miss':          0.10,
}


def compute_flush_energy(penalty_cycles, pipeline_stages):
    """Energy cost of flushing the pipeline on a misprediction."""
    return penalty_cycles * ENERGY['flush_per_stage'] * pipeline_stages
