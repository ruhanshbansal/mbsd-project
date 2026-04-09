"""
Design Space Exploration (DSE) — sweeps all predictor + BD configurations.

Replicates the paper's methodology of exploring the design space by
varying:
  - Predictor type: Static, BHT, GAs, BTB1, BTB2, Perceptron
  - Branch Detector: on/off
  - Decompression buffer depth: 1, 2, 4, 8

Results are normalized to the reference Lx configuration (static
predictor, no BD) to produce the energy-delay scatter plots from
the paper (Figures 2, 3, 4).
"""

from simulation.iss import ISS
from predictors import (StaticPredictor, BHTPredictor, GAsPredictor,
                        BTBPredictor, PerceptronPredictor)


# Buffer depths to sweep (paper Figure 4)
BUFFER_DEPTHS = [1, 2, 4, 8]


def get_default_configs():
    """
    Return the standard set of configurations to evaluate.

    Includes:
      - Paper configurations: Static, BHT, GAs, BTB1, BTB2
      - Paper + BD configurations: GAs+BD at various depths
      - ML extension: Perceptron+BD

    Returns:
        dict of {name: {'pred': predictor, 'depth': int, 'bd': bool}}
    """
    return {
        # --- Reference ---
        '1. Static (Lx Ref)':       {'pred': StaticPredictor(),                  'depth': 1, 'bd': False},

        # --- Paper predictors without BD (Figure 2) ---
        '2. BHT (no BD)':           {'pred': BHTPredictor(128),                  'depth': 1, 'bd': False},
        '3. GAs (no BD)':           {'pred': GAsPredictor(),                     'depth': 1, 'bd': False},
        '4. BTB1 (no BD)':          {'pred': BTBPredictor(64, counter_bits=1),   'depth': 1, 'bd': False},
        '5. BTB2 (no BD)':          {'pred': BTBPredictor(128, counter_bits=2),  'depth': 1, 'bd': False},

        # --- Paper predictors with BD (Figure 3) ---
        '6. GAs + BD (d=1)':        {'pred': GAsPredictor(),                     'depth': 1, 'bd': True},
        '7. GAs + BD (d=4)':        {'pred': GAsPredictor(),                     'depth': 4, 'bd': True},
        '8. GAs + BD (d=8)':        {'pred': GAsPredictor(),                     'depth': 8, 'bd': True},

        # --- ML Extension (beyond paper) ---
        '9. Perceptron + BD (d=4)': {'pred': PerceptronPredictor(64, 16),        'depth': 4, 'bd': True},
    }


def run_dse(bundles, configs=None):
    """
    Run the full design space exploration.

    Args:
        bundles: instruction trace from TraceGenerator
        configs: dict of configurations (uses defaults if None)

    Returns:
        all_results: dict of {name: results_dict}
        ref_results: results for the reference configuration
    """
    if configs is None:
        configs = get_default_configs()

    all_results = {}

    for name, cfg in configs.items():
        sim = ISS(cfg['pred'], cfg['depth'], cfg['bd'])
        sim.simulate(bundles)
        all_results[name] = sim.results()

    ref = all_results['1. Static (Lx Ref)']
    return all_results, ref


def run_buffer_sweep(bundles):
    """
    Sweep decompression buffer depth for GAs + BD.
    Replicates paper Figure 4.

    Returns:
        depth_results: dict of {depth: results_dict}
    """
    depth_results = {}
    for d in BUFFER_DEPTHS:
        sim = ISS(GAsPredictor(), buffer_depth=d, bd_enabled=True)
        sim.simulate(bundles)
        depth_results[d] = sim.results()
    return depth_results


def normalize(results, ref):
    """
    Normalize results to the reference configuration.

    Returns:
        dict of {name: (norm_energy, norm_delay)}
    """
    return {
        name: (r['energy_total'] / ref['energy_total'],
               r['cycles'] / ref['cycles'])
        for name, r in results.items()
    }
