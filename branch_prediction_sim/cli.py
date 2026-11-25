"""Reproducible experiments, console summaries, and portable result exports."""

import argparse
import csv
import json
import platform
from pathlib import Path

from branch_prediction_sim import __version__
from branch_prediction_sim.analysis.metrics import (
    compute_normalized_metrics, compute_trace_stats,
    print_results_table, print_trace_stats,
)
from branch_prediction_sim.core import pipeline
from branch_prediction_sim.core.energy import ENERGY
from branch_prediction_sim.core.program_gen import TraceGenerator
from branch_prediction_sim.simulation.dse import run_buffer_sweep, run_dse


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def run_experiment(num_bundles=5000, seed=42):
    """Use one seeded workload for every predictor and buffer configuration."""
    bundles = TraceGenerator(num_bundles=num_bundles, seed=seed).generate()
    results, reference = run_dse(bundles)
    return {
        'schema_version': 1,
        'metadata': {
            'simulator_version': __version__,
            'python_version': platform.python_version(),
            'seed': seed,
            'requested_bundles': num_bundles,
            'workload': 'synthetic patterned VLIW bundles',
            'baseline': '1. Static (Lx Ref)',
            'energy_units': 'relative model units; not measured joules',
            'pipeline_assumptions': {
                key: value for key, value in vars(pipeline).items() if key.isupper()
            },
            'energy_costs': dict(ENERGY),
        },
        'trace': compute_trace_stats(bundles),
        'results': results,
        'metrics': compute_normalized_metrics(results, reference),
        'buffer_sweep': run_buffer_sweep(bundles),
    }


def write_results(report, output_dir):
    """Save full experiment metadata plus a flat table for comparisons."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'results.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    rows = []
    for metric in report['metrics']:
        result = report['results'][metric['name']]
        rows.append({
            **metric,
            'accuracy_pct': None if result['accuracy'] is None else result['accuracy'] * 100,
            'cycles': result['cycles'],
            'energy_relative_units': result['energy_total'],
            'branches': result['branches'],
            'mispredictions': result['mispredictions'],
        })
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare branch predictors on a synthetic VLIW workload.'
    )
    parser.add_argument('--bundles', type=positive_int, default=5000,
                        help='number of synthetic bundles (default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='workload seed (default: 42)')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'),
                        help='results directory (default: outputs)')
    display = parser.add_mutually_exclusive_group()
    display.add_argument('--no-plots', action='store_true', help='export only JSON and CSV')
    display.add_argument('--show', action='store_true', help='also open interactive figures')
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)

    report = run_experiment(args.bundles, args.seed)
    print('VLIW Branch Prediction | Synthetic workload experiment')
    print_trace_stats(report['trace'])
    print_results_table(report['results'], report['metrics'])
    write_results(report, args.output_dir)
    if not args.no_plots:
        from branch_prediction_sim.analysis.plots import generate_all_plots

        generate_all_plots(report, args.output_dir, show=args.show)
    print(f'\nResults saved to {args.output_dir.resolve()}')
    print('Energy is modeled in relative units; results are not silicon measurements.')
    return 0
