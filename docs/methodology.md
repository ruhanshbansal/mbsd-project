# Model and experiment notes

## Scope

The simulator processes a pre-generated trace of four-slot bundles. It counts base cycles, synthetic stalls, and branch-misprediction penalties. It does not execute real binaries, simulate each pipeline stage, implement RTL, or model a complete processor ISA.

All nine default configurations receive the same trace and begin with fresh predictor state. Prediction is evaluated before each online update. There is no separate training/test split or warm-up interval; accuracy includes cold-start behavior.

## Workload generation

`TraceGenerator` combines loops, biased conditions, correlated pairs, alternating branches, and random branches. A local seeded random-number generator makes repeated calls reproducible without changing Python's global RNG state. Longer requested traces generate additional segments until the exact requested length is reached.

Branch density is an observed property of the generated segment mix. The old `branch_density` constructor argument was unused and has been removed. These are not MediaBench traces; no claim of matching a real application's distribution is made. Small traces can omit some pattern families, which plots label `n/a`.

## Pipeline and detector assumptions

- Four issue slots per bundle; six stages are represented through constants and flush-energy accounting, not a stage-by-stage pipeline.
- Three-cycle base misprediction penalty. The buffer lookahead heuristic reduces this to a minimum of one cycle as depth increases.
- Fixed stochastic data-hazard and cache-miss penalties are generated into the trace and shared across configurations.
- The detector suppresses predictor lookups on non-branch bundles while its buffer is active. It temporarily stops filtering after a flush.
- The refill counter advances once per bundle; it is a simplified proxy and does not model a physical refill pipeline.
- BTB entries use tags and direction counters. Target addresses are placeholders and do not drive control flow.
- The GAs implementation retains a 12-bit history register, but its default 16 entries per set select only the low four history bits.
- Logical lookup counts include the static predictor's opportunities, although its lookup energy cost is zero.

## Energy interpretation

`core/energy.py` defines hand-selected relative costs for the core, predictor accesses/updates, decompression, detection, buffer storage, and flushes. They are not extracted from CACTI, a technology library, synthesis, or hardware measurements.

The model charges core and buffer costs for base/stall cycles and a separate heuristic cost for misprediction flushes. It does not fully model voltage, frequency, process technology, leakage, predictor area, access timing, or physical implementation. Predictor costs are fixed constants rather than functions of table size. Comparisons therefore describe behavior **within these assumptions**.

## Metric definitions

Let `C` and `E` be the configuration's simulated cycles and total relative energy; let `C_ref` and `E_ref` be the static reference values.

| Metric | Definition |
| --- | --- |
| Prediction accuracy | Correct predictions / branch bundles |
| Normalized cycles (`norm_delay`) | `C / C_ref` |
| Cycle reduction (%) | `100 * (1 - C / C_ref)` |
| Speedup factor | `C_ref / C` |
| Speedup (%) | `100 * (C_ref / C - 1)` |
| Modeled energy saving (%) | `100 * (1 - E / E_ref)` |
| Mispredictions per kilo-branches | `1000 * mispredictions / branch bundles` |

For an equal-frequency comparison, a 20% cycle reduction is a 1.25x speedup (25% higher throughput). These must not be labeled interchangeably. Accuracy and the branch-normalized miss rate are `null`/`n/a` when a trace has no branches. The miss rate is not MPKI, because its denominator is branches rather than instructions.

## Interpreting the reference run

The default perceptron-plus-detector configuration uses 64 perceptrons, a 16-outcome history, and buffer depth 4. Its gains relative to the static reference include both predictor and buffer/detector changes. GAs plus the same depth is a closer comparator for the predictor choice.

Seed 42 is a reproducible example, not evidence of general superiority. Stronger conclusions would require multiple seeds, real traces, workload-level variation, matched area/latency budgets, and calibrated energy costs. The perceptron does not win every metric: the depth-1 GAs configuration has lower modeled energy in the reference run.

## Changes from the original experiment

- The default seed-42 cycle counts, accuracies, and energy totals are preserved.
- `cycle_reduction_pct` names the earlier `(1 - normalized_delay)` calculation correctly; `speedup_pct` now uses the reciprocal-cycle formula.
- `mpki` is replaced by `mispredictions_per_kilo_branches`.
- Traces longer than the original generated segment pool now honor the requested length.
- Generator state is local and reset for each `generate()` call; the unused density argument is removed.
- Headless and interactive plotting share one implementation.

Raw counters, observed pattern counts, energy assumptions, pipeline constants, seed, and runtime version are included in `results.json` for inspection.
