"""
Pipeline Model — Lx 4-issue VLIW, 6-stage pipeline.

The Lx processor is a scalable, customizable VLIW architecture designed
by STMicroelectronics and HP Labs. It was used as the experimental
platform in the paper.

References:
  - Faraboschi et al., "Lx: a technology platform for customizable
    VLIW embedded processing," ISCA 2000.  [Paper ref 11]

Pipeline stages (6):
  1. Fetch        — instruction cache access
  2. Decode       — bundle decompression + branch detection
  3. Issue        — operand read
  4. Execute 1    — ALU / branch resolution
  5. Execute 2    — memory access
  6. Writeback    — register file update

Misprediction penalty = 3 cycles (stages between fetch and branch
resolution at Execute 1).
"""


# =============================================================================
# Lx Architecture Constants
# =============================================================================
ISSUE_WIDTH     = 4     # operations per VLIW bundle
PIPELINE_STAGES = 6     # total pipeline stages
MISPRED_PENALTY = 3     # cycles flushed on misprediction


# =============================================================================
# Pipeline Stall Model
# =============================================================================
# Fixed synthetic stall assumptions shared by every configuration.
# They are not calibrated against real workloads or a cycle-accurate model.

DCACHE_MISS_RATE    = 0.08    # probability per MEM operation
DCACHE_MISS_PENALTY = 12      # cycles per D-cache miss

DATA_HAZARD_RATE    = 0.25    # probability per bundle (RAW hazard)
DATA_HAZARD_PENALTY = 2       # cycles per data hazard stall

ICACHE_MISS_RATE    = 0.020   # probability per bundle fetch
ICACHE_MISS_PENALTY = 10      # cycles per I-cache miss
