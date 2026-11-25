"""Derived metrics with explicit denominators and units."""

from collections import Counter


def compute_normalized_metrics(all_results, ref):
    """Normalize energy/cycles and distinguish cycle reduction from speedup."""
    if ref['cycles'] <= 0 or ref['energy_total'] <= 0:
        raise ValueError('Reference cycles and energy must be positive')
    metrics = []
    for name, result in all_results.items():
        if result['cycles'] <= 0:
            raise ValueError('Configuration cycles must be positive')
        energy = result['energy_total'] / ref['energy_total']
        delay = result['cycles'] / ref['cycles']
        metrics.append({
            'name': name,
            'norm_energy': energy,
            'norm_delay': delay,
            'cycle_reduction_pct': (1 - delay) * 100,
            'speedup_factor': 1 / delay,
            'speedup_pct': (1 / delay - 1) * 100,
            'energy_save_pct': (1 - energy) * 100,
        })
    return metrics


def compute_trace_stats(bundles):
    """Return actual trace composition, rather than an intended density."""
    total = len(bundles)
    branches = sum(bundle['has_branch'] for bundle in bundles)
    stalls = sum(bundle.get('stall_cycles', 0) for bundle in bundles)
    return {
        'total_bundles': total,
        'branch_bundles': branches,
        'non_branch_bundles': total - branches,
        'branch_density': branches / total if total else 0,
        'pattern_counts': dict(Counter(
            bundle['pattern_type'] for bundle in bundles if bundle['pattern_type']
        )),
        'total_stall_cycles': stalls,
        'avg_stalls_per_bundle': stalls / total if total else 0,
    }


def format_accuracy(value):
    return 'n/a' if value is None else f'{value * 100:.1f}%'


def print_trace_stats(stats):
    print(f"Trace: {stats['total_bundles']:,} synthetic bundles | "
          f"{stats['branch_bundles']:,} branches | "
          f"{stats['branch_density']:.1%} observed branch density")
    print(f"Patterns: {stats['pattern_counts']}")


def print_results_table(all_results, metrics):
    print('\nAll comparisons use the static predictor as the reference.')
    print(f"{'Configuration':<31} {'Accuracy':>9} {'Cycles saved':>13} "
          f"{'Speedup':>9} {'Energy saved':>13}")
    print('-' * 79)
    for metric in metrics:
        result = all_results[metric['name']]
        print(f"{metric['name']:<31} {format_accuracy(result['accuracy']):>9} "
              f"{metric['cycle_reduction_pct']:>12.1f}% "
              f"{metric['speedup_factor']:>8.3f}x "
              f"{metric['energy_save_pct']:>12.1f}%")
