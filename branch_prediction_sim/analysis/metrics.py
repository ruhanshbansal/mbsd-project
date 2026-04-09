"""
Metrics Computation — derived metrics for analysis and reporting.

Computes normalized energy, delay, speedup, energy savings, and
produces summary tables for console output.
"""

from collections import defaultdict


def compute_normalized_metrics(all_results, ref):
    """
    Compute normalized metrics relative to the reference configuration.

    Returns:
        list of dicts with keys:
            name, norm_energy, norm_delay, speedup_pct, energy_save_pct
    """
    metrics = []
    for name, r in all_results.items():
        ne = r['energy_total'] / ref['energy_total']
        nd = r['cycles'] / ref['cycles']
        metrics.append({
            'name':            name,
            'norm_energy':     ne,
            'norm_delay':      nd,
            'speedup_pct':     (1 - nd) * 100,
            'energy_save_pct': (1 - ne) * 100,
        })
    return metrics


def compute_trace_stats(bundles):
    """
    Compute statistics about the instruction trace.

    Returns:
        dict with trace-level statistics
    """
    total = len(bundles)
    branches = sum(1 for b in bundles if b['has_branch'])

    patterns = defaultdict(int)
    for b in bundles:
        if b['pattern_type']:
            patterns[b['pattern_type']] += 1

    total_stalls = sum(b.get('stall_cycles', 0) for b in bundles)
    avg_stalls = total_stalls / total if total > 0 else 0

    return {
        'total_bundles': total,
        'branch_bundles': branches,
        'non_branch_bundles': total - branches,
        'branch_density': branches / total if total > 0 else 0,
        'pattern_counts': dict(patterns),
        'total_stall_cycles': total_stalls,
        'avg_stalls_per_bundle': avg_stalls,
    }


def print_trace_stats(stats):
    """Print trace statistics to console."""
    print("Trace Statistics:")
    print(f"  Total bundles:       {stats['total_bundles']}")
    print(f"  Branch bundles:      {stats['branch_bundles']} "
          f"({stats['branch_density']*100:.1f}%)")
    print(f"  Non-branch bundles:  {stats['non_branch_bundles']} "
          f"({(1-stats['branch_density'])*100:.1f}%)")
    print(f"  Branch patterns:     {stats['pattern_counts']}")
    print(f"  Pipeline stalls:     {stats['total_stall_cycles']} total "
          f"({stats['avg_stalls_per_bundle']:.2f} avg/bundle)")
    print()


def print_results_table(all_results):
    """Print detailed results table."""
    header = (f"{'Config':<30} {'Cycles':>7} {'Acc%':>7} {'Mispred':>8} "
              f"{'PredAcc':>8} {'Wasted':>7} {'E_total':>9} "
              f"{'E_pred':>8} {'E_flush':>8}")
    print(header)
    print("-" * len(header))

    for name, r in all_results.items():
        print(f"{name:<30} {r['cycles']:>7d} {r['accuracy']*100:>6.1f}% "
              f"{r['mispredictions']:>8d} {r['pred_accesses']:>8d} "
              f"{r['pred_wasted']:>7d} {r['energy_total']:>9.1f} "
              f"{r['energy_predictor']:>8.2f} {r['energy_flush']:>8.2f}")


def print_normalized_table(metrics):
    """Print normalized metrics table."""
    print(f"\n--- Normalized to Static (Lx Reference) ---")
    header = (f"{'Config':<30} {'NormEnergy':>11} {'NormDelay':>10} "
              f"{'Speedup':>9} {'EnergySave':>11}")
    print(header)
    print("-" * len(header))

    for m in metrics:
        print(f"{m['name']:<30} {m['norm_energy']:>11.4f} {m['norm_delay']:>10.4f} "
              f"{m['speedup_pct']:>+8.1f}% {m['energy_save_pct']:>+10.1f}%")


def print_depth_sweep(depth_results, ref):
    """Print buffer depth sweep results."""
    print(f"\n--- Buffer Depth Sweep (GAs + BD) ---")
    print(f"{'Depth':>6} {'Cycles':>7} {'Acc%':>7} {'Penalty':>8} "
          f"{'PredAcc':>8} {'BufRefill':>10} {'NormE':>7} {'NormD':>7}")
    print("-" * 70)

    for d in sorted(depth_results.keys()):
        r = depth_results[d]
        ne = r['energy_total'] / ref['energy_total']
        nd = r['cycles'] / ref['cycles']
        print(f"{d:>6} {r['cycles']:>7d} {r['accuracy']*100:>6.1f}% "
              f"{r['eff_penalty']:>8d} {r['pred_accesses']:>8d} "
              f"{r['buffer_refills']:>10d} {ne:>7.4f} {nd:>7.4f}")
