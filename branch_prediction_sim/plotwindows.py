"""
plotwindows.py — Interactive, zoomable plot viewer.

Opens matplotlib GUI windows with all simulation plots. Each plot is
in its own window so you can zoom, pan, and inspect data points
independently.

Usage:
    python plotwindows.py

Keyboard shortcuts (in any plot window):
    - Scroll wheel: zoom in/out
    - Click + drag:  pan
    - 'h' or Home:   reset to original view
    - 's':           save current view as PNG
    - 'q':           close window
"""

import matplotlib
matplotlib.use('TkAgg')  # Interactive GUI backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from core.program_gen import TraceGenerator
from simulation.dse import run_dse, run_buffer_sweep, normalize
from analysis.metrics import compute_normalized_metrics, compute_trace_stats
from bd.detector import BranchDetector


# ─── Style ───────────────────────────────────────────────────────────────

PALETTE = {
    'Static (Lx Ref)':       '#c0392b',
    'BHT (no BD)':           '#2980b9',
    'GAs (no BD)':           '#27ae60',
    'BTB1 (no BD)':          '#f39c12',
    'BTB2 (no BD)':          '#e67e22',
    'GAs + BD (d=1)':        '#8e44ad',
    'GAs + BD (d=4)':        '#16a085',
    'GAs + BD (d=8)':        '#2c3e50',
    'Perceptron + BD (d=4)': '#e91e63',
}

MARKERS = {
    'Static (Lx Ref)':       's',
    'BHT (no BD)':           '^',
    'GAs (no BD)':           'D',
    'BTB1 (no BD)':          '<',
    'BTB2 (no BD)':          'v',
    'GAs + BD (d=1)':        'o',
    'GAs + BD (d=4)':        'p',
    'GAs + BD (d=8)':        'h',
    'Perceptron + BD (d=4)': '*',
}

PATTERN_ORDER = ['loop', 'biased_if', 'correlated', 'alternating', 'random']
PATTERN_LABELS = {
    'loop': 'Loop', 'biased_if': 'Biased If', 'correlated': 'Correlated',
    'alternating': 'Alternating', 'random': 'Random',
}


def _apply_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _short(name):
    return name.split('. ', 1)[1] if '. ' in name else name


def _is_bd(name):
    s = _short(name)
    return 'BD' in s and 'no BD' not in s


def _is_no_bd(name):
    s = _short(name)
    return 'BD' not in s or 'no BD' in s


def _label(name):
    return _short(name).replace(' + BD', '\n+BD').replace(' (', '\n(')


# ─── Run simulation ─────────────────────────────────────────────────────

def run_simulation():
    """Run the full simulation and return all results."""
    print("Running simulation...")
    gen = TraceGenerator(num_bundles=5000, branch_density=0.25, seed=42)
    bundles = gen.generate()
    stats = compute_trace_stats(bundles)

    all_results, ref = run_dse(bundles)
    norm = normalize(all_results, ref)
    norm_metrics = compute_normalized_metrics(all_results, ref)
    depth_results = run_buffer_sweep(bundles)

    print("Simulation complete. Opening interactive plots...\n")
    print("  Tip: Use scroll wheel to zoom, click+drag to pan")
    print("  Press 'h' to reset view, 's' to save, 'q' to close\n")
    return all_results, ref, norm, norm_metrics, depth_results


# ─── Individual interactive plot windows ─────────────────────────────────

def plot_energy_delay_no_bd(norm):
    """Window 1: Energy-Delay scatter WITHOUT BD (Paper Figure 2)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.canvas.manager.set_window_title('Figure 2 — Energy-Delay (No BD)')

    ne_vals, nd_vals = [], []
    for name, (ne, nd) in norm.items():
        if not _is_no_bd(name):
            continue
        s = _short(name)
        ax.scatter(ne, nd, marker=MARKERS.get(s, 'o'), color=PALETTE.get(s, 'gray'),
                   s=200, label=s, zorder=5, edgecolors='#333', linewidth=0.8)
        ax.annotate(s, (ne, nd), textcoords='offset points', xytext=(10, 8),
                    fontsize=8, alpha=0.8)
        ne_vals.append(ne)
        nd_vals.append(nd)

    ax.axhline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)
    ax.axvline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)

    if ne_vals:
        margin = 0.015
        ax.set_xlim(min(ne_vals) - margin, max(ne_vals) + margin)
        ax.set_ylim(min(nd_vals) - margin, max(nd_vals) + margin)

    ax.set_xlabel('Normalized Energy')
    ax.set_ylabel('Normalized Delay')
    ax.set_title('Energy-Delay Scatter: Without BD\n(Paper Figure 2)')
    ax.legend(loc='upper left')
    fig.tight_layout()


def plot_energy_delay_with_bd(norm):
    """Window 2: Energy-Delay scatter WITH BD (Paper Figure 3)."""
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.canvas.manager.set_window_title('Figure 3 — Energy-Delay (With BD)')

    ne_vals, nd_vals = [], []

    if '1. Static (Lx Ref)' in norm:
        ne, nd = norm['1. Static (Lx Ref)']
        ax.scatter(ne, nd, marker='s', color=PALETTE['Static (Lx Ref)'],
                   s=200, label='Static (Lx Ref)', zorder=5,
                   edgecolors='#333', linewidth=0.8)
        ax.annotate('Lx Ref', (ne, nd), textcoords='offset points',
                    xytext=(10, 8), fontsize=8, alpha=0.8)
        ne_vals.append(ne)
        nd_vals.append(nd)

    for name, (ne, nd) in norm.items():
        if not _is_bd(name):
            continue
        s = _short(name)
        ax.scatter(ne, nd, marker=MARKERS.get(s, 'o'), color=PALETTE.get(s, 'gray'),
                   s=200, label=s, zorder=5, edgecolors='#333', linewidth=0.8)
        ax.annotate(s, (ne, nd), textcoords='offset points', xytext=(10, 8),
                    fontsize=8, alpha=0.8)
        ne_vals.append(ne)
        nd_vals.append(nd)

    ax.axhline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)
    ax.axvline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)

    if ne_vals:
        margin = 0.015
        ax.set_xlim(min(ne_vals) - margin, max(ne_vals) + margin)
        ax.set_ylim(min(nd_vals) - margin, max(nd_vals) + margin)

    ax.set_xlabel('Normalized Energy')
    ax.set_ylabel('Normalized Delay')
    ax.set_title('Energy-Delay Scatter: With BD\n(Paper Figure 3)')
    ax.legend(loc='upper left')
    fig.tight_layout()


def plot_accuracy(all_results):
    """Window 3: Prediction accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title('Prediction Accuracy')

    names = list(all_results.keys())
    accs = [all_results[n]['accuracy'] * 100 for n in names]
    colors = [PALETTE.get(_short(n), '#888') for n in names]
    labels = [_label(n) for n in names]

    bars = ax.bar(range(len(names)), accs, color=colors,
                  edgecolor='#333', linewidth=0.6, width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9, ha='center')
    ax.set_ylabel('Prediction Accuracy (%)')
    ax.set_title('Branch Prediction Accuracy')
    ax.set_ylim(max(0, min(accs) - 15), 102)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
    fig.tight_layout()


def plot_buffer_depth(depth_results, ref):
    """Window 4: Buffer depth sweep — energy and delay (Paper Figure 4)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.canvas.manager.set_window_title('Figure 4 — Buffer Depth Sweep')

    depths = sorted(depth_results.keys())
    norm_e = [depth_results[d]['energy_total'] / ref['energy_total'] for d in depths]
    norm_d = [depth_results[d]['cycles'] / ref['cycles'] for d in depths]
    dc = ['#27ae60', '#2980b9', '#8e44ad', '#e67e22']

    # Energy
    bars1 = ax1.bar([str(d) for d in depths], norm_e,
                    color=dc[:len(depths)], edgecolor='#333', linewidth=0.6, width=0.6)
    ax1.axhline(1.0, color='#c0392b', ls='--', lw=1.5, label='Lx Reference', alpha=0.7)
    ax1.set_xlabel('Decompression Buffer Depth')
    ax1.set_ylabel('Normalized Energy')
    ax1.set_title('Energy vs Buffer Depth (GAs+BD)')
    ax1.set_ylim(min(norm_e) - 0.03, 1.05)
    ax1.legend()
    for bar, val in zip(bars1, norm_e):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Delay
    bars2 = ax2.bar([str(d) for d in depths], norm_d,
                    color=dc[:len(depths)], edgecolor='#333', linewidth=0.6, width=0.6)
    ax2.axhline(1.0, color='#c0392b', ls='--', lw=1.5, label='Lx Reference', alpha=0.7)
    ax2.set_xlabel('Decompression Buffer Depth')
    ax2.set_ylabel('Normalized Delay')
    ax2.set_title('Delay vs Buffer Depth (GAs+BD)')
    ax2.set_ylim(min(norm_d) - 0.03, 1.05)
    ax2.legend()
    for bar, val in zip(bars2, norm_d):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Paper Figure 4 — Buffer Depth Sweep', fontsize=14, fontweight='bold')
    fig.tight_layout()


def plot_predictor_accesses(all_results):
    """Window 5: Predictor access filtering (BD energy saving)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title('Predictor Access Filtering')

    names = list(all_results.keys())
    labels = [_label(n) for n in names]
    total = [all_results[n]['pred_accesses'] for n in names]
    wasted = [all_results[n]['pred_wasted'] for n in names]
    useful = [t - w for t, w in zip(total, wasted)]

    ax.bar(range(len(names)), useful, label='Useful (branch)',
           color='#2980b9', edgecolor='#333', lw=0.5, width=0.7)
    ax.bar(range(len(names)), wasted, bottom=useful,
           label='Wasted (non-branch)', color='#e74c3c',
           edgecolor='#333', lw=0.5, width=0.7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9, ha='center')
    ax.set_ylabel('Predictor Accesses')
    ax.set_title('Predictor Access Filtering\n(BD Energy Saving Mechanism)')
    ax.legend()

    for i, t in enumerate(total):
        ax.text(i, t + 30, str(t), ha='center', fontsize=9)
    fig.tight_layout()


def plot_per_pattern(all_results):
    """Window 6: Per-pattern prediction accuracy (ML advantage)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.canvas.manager.set_window_title('Per-Pattern Accuracy (ML Advantage)')

    names = list(all_results.keys())
    configs = []
    for n in names:
        ps = all_results[n].get('pattern_stats', {})
        if ps and any(ps.get(p, {}).get('total', 0) > 0 for p in PATTERN_ORDER):
            configs.append(n)

    if not configs:
        ax.text(0.5, 0.5, 'No pattern data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        return

    n_configs = len(configs)
    n_patterns = len(PATTERN_ORDER)
    bar_height = 0.8 / n_configs
    y_positions = np.arange(n_patterns)

    for i, cfg in enumerate(configs):
        ps = all_results[cfg].get('pattern_stats', {})
        accs = []
        for pat in PATTERN_ORDER:
            if pat in ps and ps[pat]['total'] > 0:
                accs.append(ps[pat]['correct'] / ps[pat]['total'] * 100)
            else:
                accs.append(0)

        offsets = y_positions + (i - n_configs / 2 + 0.5) * bar_height
        s = _short(cfg)
        bars = ax.barh(offsets, accs, height=bar_height * 0.9,
                       label=s, color=PALETTE.get(s, '#888'),
                       edgecolor='#333', linewidth=0.4)
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{acc:.0f}%', va='center', fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([PATTERN_LABELS[p] for p in PATTERN_ORDER], fontsize=11)
    ax.set_xlabel('Prediction Accuracy (%)')
    ax.set_title('Per-Pattern Prediction Accuracy\n(ML Advantage on Complex Patterns)')
    ax.set_xlim(0, 110)
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.invert_yaxis()
    fig.tight_layout()


def plot_energy_breakdown(all_results, ref):
    """Window 7: Energy breakdown by component."""
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.canvas.manager.set_window_title('Energy Breakdown')

    names = list(all_results.keys())
    labels = [_label(n) for n in names]
    ref_total = ref['energy_total']

    e_core   = [all_results[n]['energy_core'] / ref_total for n in names]
    e_pred   = [all_results[n]['energy_predictor'] / ref_total for n in names]
    e_decomp = [all_results[n]['energy_decomp'] / ref_total for n in names]
    e_bd     = [all_results[n]['energy_bd'] / ref_total for n in names]
    e_flush  = [all_results[n]['energy_flush'] / ref_total for n in names]
    e_buffer = [all_results[n].get('energy_buffer', 0) / ref_total for n in names]

    x = range(len(names))
    w = 0.7
    cc = {'Core': '#34495e', 'Predictor': '#2980b9', 'Decompress': '#27ae60',
          'BD': '#f39c12', 'Buffer SRAM': '#8e44ad', 'Flush': '#c0392b'}

    ax.bar(x, e_core, width=w, label='Core', color=cc['Core'], edgecolor='white', lw=0.3)
    bot = list(e_core)
    ax.bar(x, e_pred, bottom=bot, width=w, label='Predictor', color=cc['Predictor'],
           edgecolor='white', lw=0.3)
    bot = [a + b for a, b in zip(bot, e_pred)]
    ax.bar(x, e_decomp, bottom=bot, width=w, label='Decompress', color=cc['Decompress'],
           edgecolor='white', lw=0.3)
    bot = [a + b for a, b in zip(bot, e_decomp)]
    ax.bar(x, e_bd, bottom=bot, width=w, label='BD', color=cc['BD'],
           edgecolor='white', lw=0.3)
    bot = [a + b for a, b in zip(bot, e_bd)]
    ax.bar(x, e_buffer, bottom=bot, width=w, label='Buffer SRAM', color=cc['Buffer SRAM'],
           edgecolor='white', lw=0.3)
    bot = [a + b for a, b in zip(bot, e_buffer)]
    ax.bar(x, e_flush, bottom=bot, width=w, label='Flush', color=cc['Flush'],
           edgecolor='white', lw=0.3)

    ax.axhline(1.0, color='#c0392b', ls='--', lw=1.2, alpha=0.7, label='Reference')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, ha='center')
    ax.set_ylabel('Normalized Energy')
    ax.set_title('Energy Breakdown by Component\n(Normalized to Lx Reference)')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    fig.tight_layout()


def plot_combined_scatter(norm):
    """Window 8: Combined energy-delay scatter (all configs)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title('Combined Energy-Delay Scatter')

    for name, (ne, nd) in norm.items():
        s = _short(name)
        is_bd = _is_bd(name)
        ax.scatter(ne, nd, marker=MARKERS.get(s, 'o'), color=PALETTE.get(s, 'gray'),
                   s=220 if is_bd else 160, label=s, zorder=5,
                   edgecolors='#333', linewidth=0.8,
                   alpha=1.0 if is_bd else 0.7)
        ax.annotate(s, (ne, nd), textcoords='offset points',
                    xytext=(12, 8), fontsize=7.5, alpha=0.85)

    ax.axhline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)
    ax.axvline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)

    # Pareto frontier shading
    ax.fill_between([0.87, 1.0], 0.84, 1.0, alpha=0.06, color='green',
                    label='Energy-Delay improvement zone')

    ax.set_xlabel('Normalized Energy')
    ax.set_ylabel('Normalized Delay')
    ax.set_title('Energy-Delay Design Space\n(All Configurations)')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    _apply_style()
    all_results, ref, norm, norm_metrics, depth_results = run_simulation()

    print("Opening 8 interactive plot windows...")
    print("  Close all windows or press Ctrl+C to exit.\n")

    plot_energy_delay_no_bd(norm)
    plot_energy_delay_with_bd(norm)
    plot_accuracy(all_results)
    plot_buffer_depth(depth_results, ref)
    plot_predictor_accesses(all_results)
    plot_per_pattern(all_results)
    plot_energy_breakdown(all_results, ref)
    plot_combined_scatter(norm)

    plt.show()  # Blocks here until all windows are closed


if __name__ == '__main__':
    main()
