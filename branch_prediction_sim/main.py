"""
main.py — Orchestrates all simulation layers.

Runs the full pipeline:
  1. Generate synthetic workload (MediaBench-like trace)
  2. Run Design Space Exploration across all configurations
  3. Sweep decompression buffer depths
  4. Compute and print metrics
  5. Generate publication-quality figures

Usage:
    python main.py
"""

from core.program_gen import TraceGenerator
from simulation.dse import run_dse, run_buffer_sweep, normalize
from analysis.metrics import (compute_normalized_metrics, compute_trace_stats,
                              print_trace_stats, print_results_table,
                              print_normalized_table, print_depth_sweep)
from analysis.plots import generate_all_plots
from bd.detector import BranchDetector


def main():
    # ── 1. Generate workload ─────────────────────────────────────────────
    print("=" * 70)
    print("Branch Prediction for Low-Power VLIW Processors")
    print("Replication of Palermo et al. (GLSVLSI 2003) + ML Extension")
    print("=" * 70)
    print()

    gen = TraceGenerator(num_bundles=5000, branch_density=0.25, seed=42)
    bundles = gen.generate()

    stats = compute_trace_stats(bundles)
    print_trace_stats(stats)

    # Compute BD filter rate
    bd = BranchDetector(depth=1, bd_enabled=True)
    filt_rate = bd.filter_rate(bundles)
    print(f"  BD filter rate:      {filt_rate*100:.1f}% of accesses avoided\n")

    # ── 2. Run Design Space Exploration ──────────────────────────────────
    print("--- Design Space Exploration ---")
    all_results, ref = run_dse(bundles)
    print_results_table(all_results)

    # ── 3. Normalized metrics ────────────────────────────────────────────
    norm_metrics = compute_normalized_metrics(all_results, ref)
    print_normalized_table(norm_metrics)

    # ── 4. Buffer depth sweep ────────────────────────────────────────────
    depth_results = run_buffer_sweep(bundles)
    print_depth_sweep(depth_results, ref)

    # ── 5. Generate plots ────────────────────────────────────────────────
    norm = normalize(all_results, ref)
    generate_all_plots(all_results, norm, depth_results, ref,
                       output_path='results.png')

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find best paper method and best ML method
    paper_best = None
    ml_best = None
    for m in norm_metrics:
        if 'GAs + BD' in m['name']:
            if paper_best is None or m['energy_save_pct'] > paper_best['energy_save_pct']:
                paper_best = m
        if 'Perceptron' in m['name']:
            ml_best = m

    if paper_best:
        print(f"\nBest paper method:  {paper_best['name']}")
        print(f"  Speedup:     {paper_best['speedup_pct']:+.1f}%")
        print(f"  Energy save: {paper_best['energy_save_pct']:+.1f}%")

    if ml_best:
        print(f"\nML extension:       {ml_best['name']}")
        print(f"  Speedup:     {ml_best['speedup_pct']:+.1f}%")
        print(f"  Energy save: {ml_best['energy_save_pct']:+.1f}%")

    if paper_best and ml_best:
        delta_s = ml_best['speedup_pct'] - paper_best['speedup_pct']
        delta_e = ml_best['energy_save_pct'] - paper_best['energy_save_pct']
        print(f"\nML improvement over paper:")
        print(f"  Extra speedup:     {delta_s:+.2f}% points")
        print(f"  Extra energy save: {delta_e:+.2f}% points")

    # Paper comparison
    print(f"\n--- Paper Target Comparison ---")
    print(f"  Paper reports:   ~7% avg speedup, ~15% energy save")
    if paper_best:
        print(f"  Simulation:      {paper_best['speedup_pct']:+.1f}% speedup, "
              f"{paper_best['energy_save_pct']:+.1f}% energy save")

    print()


if __name__ == '__main__':
    main()
