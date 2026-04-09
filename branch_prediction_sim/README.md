# Branch Prediction Techniques for Low-Power VLIW Processors

A cycle-approximate simulator replicating the experimental methodology of:

> G. Palermo, M. Sami, C. Silvano, V. Zaccaria, R. Zafalon,  
> "Branch Prediction Techniques for Low-Power VLIW Processors,"  
> *GLSVLSI 2003*, pp. 225-228.

Extended with a **Perceptron-based ML predictor** (Jimenez & Lin, 2001) as a novel contribution beyond the original paper.

## Architecture Modeled

| Parameter | Value |
|-----------|-------|
| Architecture | Lx (4-issue VLIW) |
| Pipeline | 6-stage |
| Misprediction Penalty | 3 cycles (reduced by buffer depth) |
| Decompression Buffer | Variable (1, 2, 4, 8 entries) |
| Pipeline Stalls | Data hazards, D-cache misses, I-cache misses |

## Predictors Implemented

| Predictor | Type | Reference |
|-----------|------|-----------|
| Static | Always not-taken (Lx default) | Baseline |
| BHT | 2-bit saturating counter | Smith, ISCA 1981 |
| GAs | Two-level adaptive (GHR + PHT) | Yeh & Patt, MICRO-24 1991 |
| BTB1/BTB2 | Branch Target Buffer (1/2-bit) | Perleberg & Smith, 1993 |
| **Perceptron** | **Neural predictor (ML)** | **Jimenez & Lin, HPCA 2001** |

## Simulation Features

- **Branch Detector (BD)** — paper's core contribution: filters predictor accesses
- **Buffer-empty state** — models BD deactivation after misprediction flushes
- **Buffer SRAM energy** — deeper buffers cost more energy (differentiates d=4 vs d=8)
- **Pipeline stall model** — data hazards, D-cache/I-cache misses (realistic speedup)
- **Per-pattern accuracy** — tracks accuracy across loop, correlated, alternating patterns
- **Component-level energy** — CACTI-inspired breakdown (core, predictor, BD, flush, buffer)

## Project Structure

```
branch_prediction_sim/
├── core/
│   ├── program_gen.py     # Synthetic workload (MediaBench-like)
│   ├── pipeline.py        # Lx pipeline constants + stall model
│   └── energy.py          # CACTI-inspired energy model
├── predictors/
│   ├── base.py            # Abstract predictor interface
│   ├── bht.py             # Branch History Table
│   ├── gas.py             # GAs two-level adaptive
│   ├── btb.py             # Branch Target Buffer
│   └── perceptron.py      # Perceptron (ML extension)
├── bd/
│   └── detector.py        # Branch Detector + decompression buffer
├── simulation/
│   ├── iss.py             # Instruction Set Simulator
│   └── dse.py             # Design Space Exploration
├── analysis/
│   ├── metrics.py         # Metric computation
│   └── plots.py           # Publication-quality figures
├── main.py                # Entry point
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Output

- Console: trace stats, per-config results, normalized metrics, buffer sweep
- `results.png`: 6-panel figure (energy-delay scatter, accuracy, buffer depth, filtering)
- `results_patterns.png`: per-pattern accuracy + energy breakdown

## Key Results

| Configuration | Accuracy | Speedup | Energy Save |
|--------------|----------|---------|-------------|
| BHT (no BD) | 77.3% | +9.9% | +9.0% |
| GAs (no BD) | 81.7% | +10.9% | +8.7% |
| GAs + BD (d=1) | 81.7% | +10.9% | +10.2% |
| GAs + BD (d=4) | 81.7% | +13.8% | +9.7% |
| **Perceptron + BD (d=4)** | **83.1%** | **+13.9%** | **+9.9%** |

**Paper comparison:** Paper reports ~7% avg speedup, ~15% avg energy saving across MediaBench benchmarks. Our simulation shows ~11-14% speedup and ~10% energy saving — same order of magnitude, with differences attributable to synthetic vs real benchmark traces.

The Perceptron predictor achieves the highest accuracy (83.1%) by capturing complex inter-branch correlations, particularly excelling on correlated patterns (96%) where traditional BHT achieves only 80%.
