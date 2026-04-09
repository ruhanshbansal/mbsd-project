"""
Visualization — publication-quality figures replicating and extending the paper.

Generates 8 plots across 2 figures:

Figure 1 (results.png) — 2×3 grid:
  Panel 1: Energy-Delay scatter without BD  (paper Figure 2)
  Panel 2: Energy-Delay scatter with BD     (paper Figure 3)
  Panel 3: Prediction accuracy              (all configs)
  Panel 4: Buffer depth vs Energy           (paper Figure 4)
  Panel 5: Buffer depth vs Delay            (paper Figure 4)
  Panel 6: Predictor access filtering       (BD effect)

Figure 2 (results_patterns.png) — 1×2:
  Panel 1: Per-pattern accuracy heatmap     (NEW — ML contribution)
  Panel 2: Energy breakdown by component    (stacked bar)

All figures use normalized metrics matching the paper's presentation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ─── Professional color palette ──────────────────────────────────────────
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

# Pattern display names and order
PATTERN_ORDER = ['loop', 'biased_if', 'correlated', 'alternating', 'random']
PATTERN_LABELS = {
    'loop':        'Loop',
    'biased_if':   'Biased If',
    'correlated':  'Correlated',
    'alternating': 'Alternating',
    'random':      'Random',
}


def _apply_style():
    """Apply consistent publication-quality styling."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _short_name(full_name):
    """Extract short name from numbered config name."""
    return full_name.split('. ', 1)[1] if '. ' in full_name else full_name


def _get_color(name):
    return PALETTE.get(_short_name(name), '#888888')


def _get_marker(name):
    return MARKERS.get(_short_name(name), 'o')


def _format_label(name):
    """Format name for x-axis labels (with line breaks)."""
    short = _short_name(name)
    return short.replace(' + BD', '\n+BD').replace(' (', '\n(')


def _is_bd_config(name):
    """Check if a config uses the Branch Detector."""
    short = _short_name(name)
    return 'BD' in short and 'no BD' not in short


def _is_no_bd_config(name):
    """Check if a config does NOT use the Branch Detector (or is reference)."""
    short = _short_name(name)
    return 'BD' not in short or 'no BD' in short


# ═════════════════════════════════════════════════════════════════════════
# Main figure generation
# ═════════════════════════════════════════════════════════════════════════

def generate_all_plots(all_results, norm, depth_results, ref,
                       output_path='results.png'):
    """
    Generate the complete set of publication-quality figures.

    Args:
        all_results:   dict of {name: results_dict}
        norm:          dict of {name: (norm_energy, norm_delay)}
        depth_results: dict of {depth: results_dict}
        ref:           reference configuration results
        output_path:   path to save the main figure
    """
    _apply_style()

    # ── Figure 1: Main 6-panel (2×3) ──
    _generate_main_figure(all_results, norm, depth_results, ref, output_path)

    # ── Figure 2: Per-pattern + energy breakdown ──
    pattern_path = output_path.replace('.png', '_patterns.png')
    _generate_pattern_figure(all_results, ref, pattern_path)


def _generate_main_figure(all_results, norm, depth_results, ref, output_path):
    """Generate the main 2×3 panel figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        'Branch Prediction for Low-Power VLIW Processors\n'
        'Replication of Palermo et al. (GLSVLSI 2003) + ML Extension',
        fontsize=14, fontweight='bold', y=0.99
    )

    names = list(all_results.keys())

    # Row 1
    _plot_energy_delay_no_bd(axes[0, 0], norm)
    _plot_energy_delay_with_bd(axes[0, 1], norm)
    _plot_accuracy(axes[0, 2], all_results, names)

    # Row 2
    _plot_depth_energy(axes[1, 0], depth_results, ref)
    _plot_depth_delay(axes[1, 1], depth_results, ref)
    _plot_predictor_accesses(axes[1, 2], all_results, names)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n[OK] Main plots saved to {output_path}")
    plt.close(fig)


def _generate_pattern_figure(all_results, ref, output_path):
    """Generate the per-pattern accuracy + energy breakdown figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Per-Pattern Analysis & Energy Breakdown',
        fontsize=13, fontweight='bold', y=1.02
    )

    names = list(all_results.keys())

    _plot_pattern_accuracy(axes[0], all_results, names)
    _plot_energy_breakdown(axes[1], all_results, names, ref)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Pattern plots saved to {output_path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Individual plot functions
# ═════════════════════════════════════════════════════════════════════════

def _plot_energy_delay_no_bd(ax, norm):
    """Energy-Delay scatter for configs WITHOUT BD (paper Figure 2)."""
    ne_vals, nd_vals = [], []
    for name, (ne, nd) in norm.items():
        if not _is_no_bd_config(name):
            continue
        short = _short_name(name)
        ax.scatter(ne, nd,
                   marker=MARKERS.get(short, 'o'),
                   color=PALETTE.get(short, 'gray'),
                   s=180, label=short,
                   zorder=5, edgecolors='#333333', linewidth=0.8)
        ne_vals.append(ne)
        nd_vals.append(nd)

    ax.axhline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)
    ax.axvline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)

    # Auto-zoom with margin
    if ne_vals and nd_vals:
        margin = 0.02
        ax.set_xlim(min(ne_vals) - margin, max(ne_vals) + margin)
        ax.set_ylim(min(nd_vals) - margin, max(nd_vals) + margin)

    ax.set_xlabel('Normalized Energy')
    ax.set_ylabel('Normalized Delay')
    ax.set_title('Energy-Delay: Without BD\n(Paper Figure 2)')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9)


def _plot_energy_delay_with_bd(ax, norm):
    """Energy-Delay scatter for configs WITH BD (paper Figure 3)."""
    ne_vals, nd_vals = [], []

    # Plot reference point for context
    if '1. Static (Lx Ref)' in norm:
        ne, nd = norm['1. Static (Lx Ref)']
        ax.scatter(ne, nd, marker='s', color=PALETTE['Static (Lx Ref)'],
                   s=180, label='Static (Lx Ref)',
                   zorder=5, edgecolors='#333333', linewidth=0.8)
        ne_vals.append(ne)
        nd_vals.append(nd)

    for name, (ne, nd) in norm.items():
        if not _is_bd_config(name):
            continue
        short = _short_name(name)
        ax.scatter(ne, nd,
                   marker=MARKERS.get(short, 'o'),
                   color=PALETTE.get(short, 'gray'),
                   s=180, label=short,
                   zorder=5, edgecolors='#333333', linewidth=0.8)
        ne_vals.append(ne)
        nd_vals.append(nd)

    ax.axhline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)
    ax.axvline(1.0, color='#c0392b', ls='--', alpha=0.5, lw=1.2)

    # Auto-zoom with margin
    if ne_vals and nd_vals:
        margin = 0.02
        ax.set_xlim(min(ne_vals) - margin, max(ne_vals) + margin)
        ax.set_ylim(min(nd_vals) - margin, max(nd_vals) + margin)

    ax.set_xlabel('Normalized Energy')
    ax.set_ylabel('Normalized Delay')
    ax.set_title('Energy-Delay: With BD\n(Paper Figure 3)')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9)


def _plot_accuracy(ax, all_results, names):
    """Bar chart of prediction accuracy across all configs."""
    accs = [all_results[n]['accuracy'] * 100 for n in names]
    colors = [_get_color(n) for n in names]
    labels = [_format_label(n) for n in names]

    bars = ax.bar(range(len(names)), accs, color=colors,
                  edgecolor='#333333', linewidth=0.6, width=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=6.5, ha='center')
    ax.set_ylabel('Prediction Accuracy (%)')
    ax.set_title('Branch Prediction Accuracy')
    ax.set_ylim(max(0, min(accs) - 15), 102)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', fontsize=6.5, fontweight='bold')


def _plot_depth_energy(ax, depth_results, ref):
    """Buffer depth vs normalized energy (paper Figure 4 top)."""
    depths = sorted(depth_results.keys())
    norm_e = [depth_results[d]['energy_total'] / ref['energy_total'] for d in depths]
    dc = ['#27ae60', '#2980b9', '#8e44ad', '#e67e22']

    bars = ax.bar([str(d) for d in depths], norm_e,
                  color=dc[:len(depths)], edgecolor='#333333', linewidth=0.6,
                  width=0.6)
    ax.axhline(1.0, color='#c0392b', ls='--', lw=1.5,
               label='Lx Reference', alpha=0.7)
    ax.set_xlabel('Decompression Buffer Depth')
    ax.set_ylabel('Normalized Energy')
    ax.set_title('Energy vs Buffer Depth (GAs+BD)\n(Paper Figure 4)')
    ax.set_ylim(min(norm_e) - 0.03, 1.05)
    ax.legend(fontsize=8)

    for bar, val in zip(bars, norm_e):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')


def _plot_depth_delay(ax, depth_results, ref):
    """Buffer depth vs normalized delay (paper Figure 4 bottom)."""
    depths = sorted(depth_results.keys())
    norm_d = [depth_results[d]['cycles'] / ref['cycles'] for d in depths]
    dc = ['#27ae60', '#2980b9', '#8e44ad', '#e67e22']

    bars = ax.bar([str(d) for d in depths], norm_d,
                  color=dc[:len(depths)], edgecolor='#333333', linewidth=0.6,
                  width=0.6)
    ax.axhline(1.0, color='#c0392b', ls='--', lw=1.5,
               label='Lx Reference', alpha=0.7)
    ax.set_xlabel('Decompression Buffer Depth')
    ax.set_ylabel('Normalized Delay')
    ax.set_title('Delay vs Buffer Depth (GAs+BD)\n(Paper Figure 4)')
    ax.set_ylim(min(norm_d) - 0.03, 1.05)
    ax.legend(fontsize=8)

    for bar, val in zip(bars, norm_d):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')


def _plot_predictor_accesses(ax, all_results, names):
    """Predictor access count showing BD filtering effect."""
    labels = [_format_label(n) for n in names]

    total  = [all_results[n]['pred_accesses'] for n in names]
    wasted = [all_results[n]['pred_wasted'] for n in names]
    useful = [t - w for t, w in zip(total, wasted)]

    ax.bar(range(len(names)), useful, label='Useful (branch)',
           color='#2980b9', edgecolor='#333333', lw=0.5, width=0.7)
    ax.bar(range(len(names)), wasted, bottom=useful,
           label='Wasted (non-branch)', color='#e74c3c',
           edgecolor='#333333', lw=0.5, width=0.7)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=6.5, ha='center')
    ax.set_ylabel('Predictor Accesses')
    ax.set_title('Predictor Access Filtering\n(BD Energy Saving Mechanism)')
    ax.legend(fontsize=7)

    for i, t in enumerate(total):
        ax.text(i, t + 20, str(t), ha='center', fontsize=6.5)


def _plot_pattern_accuracy(ax, all_results, names):
    """
    Per-pattern prediction accuracy as a grouped horizontal bar chart.

    This is the key ML contribution plot: shows that the Perceptron
    outperforms traditional predictors on complex patterns (correlated,
    alternating) while maintaining performance on simple patterns (loops).
    """
    # Collect configs that have pattern stats
    configs_to_show = []
    for n in names:
        ps = all_results[n].get('pattern_stats', {})
        if ps and any(ps[p]['total'] > 0 for p in PATTERN_ORDER if p in ps):
            configs_to_show.append(n)

    if not configs_to_show:
        ax.text(0.5, 0.5, 'No pattern data', ha='center', va='center',
                transform=ax.transAxes)
        return

    n_configs = len(configs_to_show)
    n_patterns = len(PATTERN_ORDER)
    bar_height = 0.8 / n_configs

    y_positions = np.arange(n_patterns)

    for i, cfg_name in enumerate(configs_to_show):
        ps = all_results[cfg_name].get('pattern_stats', {})
        accs = []
        for pat in PATTERN_ORDER:
            if pat in ps and ps[pat]['total'] > 0:
                accs.append(ps[pat]['correct'] / ps[pat]['total'] * 100)
            else:
                accs.append(0)

        offsets = y_positions + (i - n_configs / 2 + 0.5) * bar_height
        short = _short_name(cfg_name)
        color = PALETTE.get(short, '#888888')

        bars = ax.barh(offsets, accs, height=bar_height * 0.9,
                       label=short, color=color,
                       edgecolor='#333333', linewidth=0.4)

        # Add value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{acc:.0f}%', va='center', fontsize=5.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([PATTERN_LABELS.get(p, p) for p in PATTERN_ORDER])
    ax.set_xlabel('Prediction Accuracy (%)')
    ax.set_title('Per-Pattern Prediction Accuracy\n(ML Advantage on Complex Patterns)')
    ax.set_xlim(0, 110)
    ax.legend(fontsize=6, loc='lower right', ncol=2, framealpha=0.9)
    ax.invert_yaxis()


def _plot_energy_breakdown(ax, all_results, names, ref):
    """Stacked bar chart of energy breakdown by component (normalized)."""
    labels = [_format_label(n) for n in names]

    # Normalize all energy components to reference total
    ref_total = ref['energy_total']
    e_core   = [all_results[n]['energy_core'] / ref_total for n in names]
    e_pred   = [all_results[n]['energy_predictor'] / ref_total for n in names]
    e_decomp = [all_results[n]['energy_decomp'] / ref_total for n in names]
    e_bd     = [all_results[n]['energy_bd'] / ref_total for n in names]
    e_flush  = [all_results[n]['energy_flush'] / ref_total for n in names]
    e_buffer = [all_results[n].get('energy_buffer', 0) / ref_total for n in names]

    x = range(len(names))
    w = 0.7

    component_colors = {
        'Core':       '#34495e',
        'Predictor':  '#2980b9',
        'Decompress': '#27ae60',
        'BD':         '#f39c12',
        'Flush':      '#c0392b',
        'Buffer':     '#8e44ad',
    }

    bars_core = ax.bar(x, e_core, width=w, label='Core',
                       color=component_colors['Core'], edgecolor='white', lw=0.3)
    bot1 = e_core

    bars_pred = ax.bar(x, e_pred, bottom=bot1, width=w, label='Predictor',
                       color=component_colors['Predictor'], edgecolor='white', lw=0.3)
    bot2 = [a + b for a, b in zip(bot1, e_pred)]

    ax.bar(x, e_decomp, bottom=bot2, width=w, label='Decompress',
           color=component_colors['Decompress'], edgecolor='white', lw=0.3)
    bot3 = [a + b for a, b in zip(bot2, e_decomp)]

    ax.bar(x, e_bd, bottom=bot3, width=w, label='BD',
           color=component_colors['BD'], edgecolor='white', lw=0.3)
    bot4 = [a + b for a, b in zip(bot3, e_bd)]

    ax.bar(x, e_buffer, bottom=bot4, width=w, label='Buffer SRAM',
           color=component_colors['Buffer'], edgecolor='white', lw=0.3)
    bot5 = [a + b for a, b in zip(bot4, e_buffer)]

    ax.bar(x, e_flush, bottom=bot5, width=w, label='Flush',
           color=component_colors['Flush'], edgecolor='white', lw=0.3)

    # Reference line
    ax.axhline(1.0, color='#c0392b', ls='--', lw=1.2, alpha=0.7,
               label='Reference')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, ha='center')
    ax.set_ylabel('Normalized Energy')
    ax.set_title('Energy Breakdown by Component\n(Normalized to Lx Reference)')
    ax.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.9)
