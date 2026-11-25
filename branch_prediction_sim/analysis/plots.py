"""Shared headless and interactive figures for an experiment report."""

from pathlib import Path

import numpy as np

INK = '#172D3B'
TEAL = '#167D8D'
BLUE = '#487CA8'
ORANGE = '#C17335'
GRAY = '#A3ADB5'
PATTERNS = ['loop', 'biased_if', 'correlated', 'alternating', 'random']


def short_name(name):
    return name.split('. ', 1)[-1].replace(' (Lx Ref)', ' (reference)')


def _style_axis(ax, title, xlabel):
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=14, color=INK)
    ax.set_xlabel(xlabel, labelpad=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#D1D9DE')
    ax.tick_params(length=0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.18)


def _header(fig, title, report):
    metadata = report['metadata']
    count = report['trace']['total_bundles']
    fig.suptitle(title, x=.04, y=.99, ha='left', fontsize=20, fontweight='bold', color=INK)
    fig.text(.04, .925,
             f"{count:,} synthetic bundles  /  seed {metadata['seed']}  /  static reference",
             fontsize=10, color='#596C78')
    fig.text(.04, .015,
             'Cycle-approximate model. Energy uses relative component costs, not hardware measurements.',
             fontsize=9, color='#596C78')


def _overview(plt, report):
    results = report['results']
    names = list(results)
    labels = [short_name(name) for name in names]
    metrics = report['metrics']
    y = np.arange(len(names))
    colors = [TEAL if 'Perceptron' in name else BLUE for name in names]
    colors[0] = GRAY
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    _header(fig, 'Branch prediction, measured within the model', report)

    ax = axes[0, 0]
    accuracy = [results[name]['accuracy'] for name in names]
    values = [value * 100 if value is not None else 0 for value in accuracy]
    ax.barh(y, values, color=colors, height=.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    for i, (value, raw) in enumerate(zip(values, accuracy)):
        ax.text(value + 1, i, 'n/a' if raw is None else f'{value:.1f}%', va='center', fontsize=9)
    _style_axis(ax, '01  Prediction accuracy', 'Correct branch predictions (%)')

    ax = axes[0, 1]
    ax.barh(y - .17, [m['cycle_reduction_pct'] for m in metrics],
            height=.3, color=TEAL, label='Cycles saved')
    ax.barh(y + .17, [m['energy_save_pct'] for m in metrics],
            height=.3, color=ORANGE, label='Modeled energy saved')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color=GRAY, linewidth=.8)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    _style_axis(ax, '02  Change from the static reference', 'Reduction (%) - larger is better')

    ax = axes[1, 0]
    sweep = report['buffer_sweep']
    depths = sorted(sweep, key=int)
    ref = results[report['metadata']['baseline']]
    ax.plot([str(d) for d in depths], [sweep[d]['energy_total'] / ref['energy_total'] for d in depths],
            marker='o', color=ORANGE, linewidth=2, label='Modeled energy')
    ax.plot([str(d) for d in depths], [sweep[d]['cycles'] / ref['cycles'] for d in depths],
            marker='o', color=TEAL, linewidth=2, label='Cycles')
    ax.axhline(1, color=GRAY, linestyle='--', linewidth=1, label='Static reference')
    ax.set_ylabel('Ratio to static reference - smaller is better')
    ax.legend(frameon=False, fontsize=9)
    _style_axis(ax, '03  Buffer-depth tradeoff / GAs + BD', 'Decompression-buffer entries')

    ax = axes[1, 1]
    branch = [results[name]['pred_accesses'] - results[name]['pred_wasted'] for name in names]
    other = [results[name]['pred_wasted'] for name in names]
    ax.barh(y, branch, color=TEAL, height=.58, label='Branch bundles')
    ax.barh(y, other, left=branch, color='#CED8DF', height=.58, label='Non-branch bundles')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    _style_axis(ax, '04  Predictor access filtering', 'Logical lookup opportunities (count)')
    fig.tight_layout(rect=[.02, .04, .99, .91], w_pad=3, h_pad=3)
    return fig


def _detail(plt, report):
    results = report['results']
    names = list(results)
    labels = [short_name(name) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    _header(fig, 'Where the differences come from', report)
    ax = axes[0]
    accuracy = np.full((len(names), len(PATTERNS)), np.nan)
    for row, name in enumerate(names):
        for col, pattern in enumerate(PATTERNS):
            counts = results[name]['pattern_stats'].get(pattern)
            if counts and counts['total']:
                accuracy[row, col] = 100 * counts['correct'] / counts['total']
    cmap = plt.get_cmap('Blues').with_extremes(bad='#EDF1F4')
    ax.imshow(np.ma.masked_invalid(accuracy), cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_yticks(range(len(names)), labels)
    ax.set_xticks(range(len(PATTERNS)), ['Loop', 'Biased if', 'Correlated', 'Alternating', 'Random'])
    for row in range(len(names)):
        for col in range(len(PATTERNS)):
            value = accuracy[row, col]
            ax.text(col, row, 'n/a' if np.isnan(value) else f'{value:.0f}%',
                    ha='center', va='center', fontsize=9,
                    color='white' if value > 65 else INK)
    _style_axis(ax, '05  Accuracy by branch pattern', 'Absent patterns are shown as n/a')
    ax.grid(False)

    ax = axes[1]
    reference = results[report['metadata']['baseline']]['energy_total']
    components = [
        ('energy_core', 'Core', INK), ('energy_predictor', 'Predictor', BLUE),
        ('energy_decomp', 'Decompression', '#91B7C3'), ('energy_bd', 'Detector', TEAL),
        ('energy_buffer', 'Buffer', '#D4BCA0'), ('energy_flush', 'Flush', ORANGE),
    ]
    left = np.zeros(len(names))
    for key, label, color in components:
        values = np.array([results[name][key] / reference for name in names])
        ax.barh(range(len(names)), values, left=left, color=color, height=.58, label=label)
        left += values
    ax.set_yticks(range(len(names)), labels)
    ax.invert_yaxis()
    ax.axvline(1, color=GRAY, linestyle='--', linewidth=1)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.15), fontsize=8)
    _style_axis(ax, '06  Relative energy composition', 'Component energy / static total energy')
    fig.tight_layout(rect=[.02, .07, .99, .89], w_pad=3)
    return fig


def generate_all_plots(report, output_dir, show=False):
    """Save shared figures; optionally open them using a GUI backend."""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK}):
        figures = [(_overview(plt, report), 'overview.png'),
                   (_detail(plt, report), 'patterns_energy.png')]
        try:
            for figure, filename in figures:
                figure.savefig(output_dir / filename, dpi=160, facecolor='white')
            if show:
                plt.show()
        finally:
            for figure, _ in figures:
                plt.close(figure)
