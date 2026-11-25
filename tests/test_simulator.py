"""Behavioral checks for the simulator and its public experiment interface."""

import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from branch_prediction_sim.analysis.metrics import compute_normalized_metrics
from branch_prediction_sim.cli import run_experiment, write_results
from branch_prediction_sim.core.program_gen import TraceGenerator
from branch_prediction_sim.predictors import BHTPredictor, BTBPredictor, PerceptronPredictor, StaticPredictor
from branch_prediction_sim.simulation.iss import ISS

ROOT = Path(__file__).resolve().parents[1]


class TraceTests(unittest.TestCase):
    def test_seed_is_repeatable_and_isolated(self):
        random.seed(123)
        state = random.getstate()
        generator = TraceGenerator(500, seed=42)
        first = generator.generate()
        TraceGenerator(500, seed=7).generate()
        self.assertEqual(first, generator.generate())
        self.assertEqual(first, TraceGenerator(500, seed=42).generate())
        self.assertEqual(state, random.getstate())

    def test_different_seeds_produce_different_traces(self):
        self.assertNotEqual(TraceGenerator(100, seed=42).generate(),
                            TraceGenerator(100, seed=7).generate())

    def test_large_trace_has_exact_requested_length(self):
        bundles = TraceGenerator(25000).generate()
        self.assertEqual(len(bundles), 25000)
        self.assertTrue(all(len(b['ops']) == 4 for b in bundles))
        self.assertTrue(all(b['ops'].count('BRANCH') == int(b['has_branch']) for b in bundles))

    def test_invalid_trace_size(self):
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                TraceGenerator(size)


class PredictorTests(unittest.TestCase):
    def test_bht_hysteresis_and_saturation(self):
        predictor = BHTPredictor(4)
        self.assertEqual(predictor.predict(0), 0)
        for _ in range(10):
            predictor.update(0, 1)
        self.assertEqual(predictor.table[0], 3)
        predictor.update(0, 0)
        self.assertEqual(predictor.predict(0), 1)
        predictor.update(0, 0)
        self.assertEqual(predictor.predict(0), 0)
        for _ in range(10):
            predictor.update(0, 0)
        self.assertEqual(predictor.table[0], 0)

    def test_btb_tag_collision(self):
        predictor = BTBPredictor(entries=2)
        predictor.update(0, 1)
        self.assertEqual(predictor.predict(0), 1)
        self.assertEqual(predictor.predict(2), 0)
        predictor.update(2, 1)
        self.assertEqual(predictor.predict(0), 0)
        self.assertEqual(predictor.predict(2), 1)

    def test_perceptron_learns_repeated_outcome(self):
        for outcome in (0, 1):
            predictor = PerceptronPredictor()
            for _ in range(100):
                predictor.update(7, outcome)
            self.assertEqual(predictor.predict(7), outcome)


class ModelTests(unittest.TestCase):
    def test_cycle_count_on_known_trace(self):
        sim = ISS(StaticPredictor())
        sim.simulate([
            {'ops': ['BRANCH', 'NOP', 'NOP', 'NOP'], 'has_branch': True,
             'branch_pc': 1, 'branch_outcome': 1, 'pattern_type': 'known', 'stall_cycles': 2},
            {'ops': ['ALU', 'NOP', 'NOP', 'NOP'], 'has_branch': False, 'stall_cycles': 0},
        ])
        result = sim.results()
        self.assertEqual(result['cycles'], 7)  # two bundles + two stalls + three-cycle flush
        self.assertEqual(result['mispredictions'], 1)
        self.assertEqual(result['mispredictions_per_kilo_branches'], 1000)

    def test_empty_trace_has_undefined_accuracy(self):
        result = ISS(StaticPredictor()).results()
        self.assertIsNone(result['accuracy'])
        self.assertIsNone(result['mispredictions_per_kilo_branches'])

    def test_detector_filters_non_branches(self):
        bundles = [{'ops': ['ALU'] * 4, 'has_branch': False} for _ in range(10)]
        disabled = ISS(BHTPredictor(), bd_enabled=False)
        enabled = ISS(BHTPredictor(), bd_enabled=True)
        disabled.simulate(bundles)
        enabled.simulate(bundles)
        self.assertEqual(disabled.results()['pred_accesses'], 10)
        self.assertEqual(enabled.results()['pred_accesses'], 0)

    def test_energy_components_sum_to_total(self):
        report = run_experiment(300)
        for result in report['results'].values():
            components = ['energy_core', 'energy_predictor', 'energy_decomp',
                          'energy_bd', 'energy_flush', 'energy_buffer']
            self.assertAlmostEqual(result['energy_total'], sum(result[k] for k in components))
            self.assertEqual(result['branches'], result['correct'] + result['mispredictions'])


class MetricTests(unittest.TestCase):
    def test_cycle_reduction_is_not_speedup(self):
        metric = compute_normalized_metrics(
            {'example': {'cycles': 80, 'energy_total': 90}},
            {'cycles': 100, 'energy_total': 100},
        )[0]
        self.assertAlmostEqual(metric['cycle_reduction_pct'], 20)
        self.assertAlmostEqual(metric['speedup_pct'], 25)
        self.assertAlmostEqual(metric['speedup_factor'], 1.25)
        self.assertAlmostEqual(metric['energy_save_pct'], 10)

    def test_zero_reference_rejected(self):
        with self.assertRaises(ValueError):
            compute_normalized_metrics({}, {'cycles': 0, 'energy_total': 0})

    def test_original_default_results_preserved(self):
        expected = json.loads((ROOT / 'tests/fixtures/default_results.json').read_text())
        actual = run_experiment()['results']
        for name, fields in expected.items():
            for key, value in fields.items():
                with self.subTest(configuration=name, metric=key):
                    self.assertAlmostEqual(actual[name][key], value)


class InterfaceTests(unittest.TestCase):
    def test_json_and_csv_exports(self):
        report = run_experiment(100)
        with tempfile.TemporaryDirectory() as folder:
            write_results(report, folder)
            stored = json.loads((Path(folder) / 'results.json').read_text())
            self.assertEqual(stored['trace']['total_bundles'], 100)
            self.assertEqual(stored['metadata']['seed'], 42)
            with (Path(folder) / 'summary.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertIn('cycle_reduction_pct', rows[0])
            self.assertIn('speedup_factor', rows[0])

    def test_module_and_legacy_script(self):
        for entry in (['-m', 'branch_prediction_sim'], ['branch_prediction_sim/main.py']):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as folder:
                result = subprocess.run(
                    [sys.executable, *entry, '--bundles', '100', '--no-plots', '--output-dir', folder],
                    cwd=ROOT, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((Path(folder) / 'results.json').exists())

    def test_cli_rejects_invalid_size(self):
        result = subprocess.run([sys.executable, '-m', 'branch_prediction_sim', '--bundles', '0'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('positive integer', result.stderr)

    def test_headless_plots_with_missing_patterns(self):
        from branch_prediction_sim.analysis.plots import generate_all_plots
        report = run_experiment(1)
        with tempfile.TemporaryDirectory() as folder:
            generate_all_plots(report, folder)
            for filename in ('overview.png', 'patterns_energy.png'):
                self.assertEqual((Path(folder) / filename).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')


if __name__ == '__main__':
    unittest.main()
