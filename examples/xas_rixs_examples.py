#!/usr/bin/env python
"""
Worked examples for :mod:`xas_rixs`.

    python examples/xas_rixs_examples.py            # list the examples
    python examples/xas_rixs_examples.py 2          # run one
    python examples/xas_rixs_examples.py all        # run all of them

Each example writes a PNG and the corresponding data files into ``output/``.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import xas_rixs as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'

# Categorical series colours, always assigned in this fixed order (never cycled),
# and a single-hue ink for stacked spectra that are identified by direct labels.
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
INK, MUTED, GRID = '#1a1a19', '#5c5b55', '#e3e2dc'
SURFACE = '#fcfcfb'
#: sequential, one hue light -> dark, for intensity maps
SEQ = LinearSegmentedColormap.from_list('seq', [SURFACE, '#9cc3ee', SERIES[0], '#0d2f5c'])

mpl.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'legend.frameon': False, 'lines.linewidth': 1.6,
    'figure.dpi': 130, 'savefig.dpi': 150,
})


def _save(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / f'{name}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {path}')


def _label(ax, x, y, text):
    ax.annotate(text, (x, y), ha='left', va='bottom', fontsize=9, color=MUTED)


# ----------------------------------------------------------------------------

def example_1():
    """Ni2+ L2,3 XAS: the effect of the cubic crystal field."""
    fig, ax = plt.subplots(figsize=(7, 4))
    omega = np.linspace(-6, 22, 2000)
    for c, ten_dq in zip(SERIES, (0.0, 1.0, 2.0)):
        ion = xr.solve_ion(xr.IonModel(atom='Ni', valence=2, ten_dq=ten_dq,
                                       temperature=10), verbose=False)
        res = ion.xas(omega, gamma_c=0.3, gauss_fwhm=0.2)
        res.save(outdir=OUT, prefix=f'ex1_Ni2_10Dq{ten_dq:g}')
        ax.plot(res.omega, res.intensity, color=c, label=f'10Dq = {ten_dq:g} eV')
    ax.set_xlabel('photon energy relative to the L$_3$ peak (eV)')
    ax.set_ylabel('isotropic XAS (arb. units)')
    ax.set_title('Ni$^{2+}$ L$_{2,3}$: crystal-field splitting of the L$_3$ and L$_2$ lines',
                 fontsize=10, loc='left')
    ax.legend()
    _save(fig, 'example_1_ni_crystal_field')


def example_2():
    """L2,3 XAS across the 3d series (octahedral, 10Dq = 1.5 eV)."""
    ions = [('Ti', 4), ('V', 3), ('Cr', 3), ('Mn', 2), ('Fe', 3), ('Fe', 2),
            ('Co', 2), ('Ni', 2), ('Cu', 2)]
    fig, ax = plt.subplots(figsize=(6.5, 8))
    omega = np.linspace(-8, 25, 2000)
    for n, (atom, val) in enumerate(ions):
        ion = xr.solve_ion(xr.IonModel(atom=atom, valence=val, ten_dq=1.5,
                                       temperature=10), verbose=False)
        res = ion.xas(omega, gamma_c=0.3, gauss_fwhm=0.2, normalize=True)
        y = res.intensity + 1.1 * n
        ax.plot(res.omega, y, color=INK, lw=1.1)
        ax.fill_between(res.omega, 1.1 * n, y, color=SERIES[0], alpha=0.12)
        _label(ax, 16, 1.1 * n + 0.35, f"{atom}$^{{{val}+}}$  d$^{{{ion.info['nd']}}}$")
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('photon energy relative to the strongest peak (eV)')
    ax.set_title('L$_{2,3}$ XAS of the 3d series, O$_h$, 10Dq = 1.5 eV',
                 fontsize=10, loc='left')
    _save(fig, 'example_2_3d_series')


def example_3():
    """Co2+ XMCD in an exchange field, and the orbital sum rule."""
    ion = xr.solve_ion(xr.IonModel(atom='Co', valence=2, ten_dq=0.9,
                                   ext_B=(0, 0, 0.02), temperature=5), verbose=False)
    omega = np.linspace(-6, 22, 2000)
    # normal incidence along -z: the photon helicity is along the field
    plus = ion.xas(omega, pol='left', gamma_c=0.3)
    minus = ion.xas(omega, pol='right', gamma_c=0.3)
    zpol = ion.xas(omega, pol=[0, 0, 1], gamma_c=0.3)
    plus.save(outdir=OUT, prefix='ex3_Co2_left')
    minus.save(outdir=OUT, prefix='ex3_Co2_right')
    norm = max(plus.intensity.max(), minus.intensity.max())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=(3, 2)))
    a1.plot(omega, plus.intensity / norm, color=SERIES[0], label='left circular')
    a1.plot(omega, minus.intensity / norm, color=SERIES[1], label='right circular')
    a1.set_ylabel('XAS (arb. units)')
    a1.legend()
    a1.set_title('Co$^{2+}$ L$_{2,3}$ XMCD, O$_h$, exchange field along the beam',
                 fontsize=10, loc='left')
    diff = (plus.intensity - minus.intensity) / norm
    a2.plot(omega, diff, color=INK, lw=1.2)
    a2.axhline(0, color=MUTED, lw=0.6)
    a2.set_ylabel('XMCD (left - right)')
    a2.set_xlabel('photon energy relative to the L$_3$ peak (eV)')
    w = {k: r.poles[:, 1].sum() for k, r in (('+', plus), ('-', minus), ('0', zpol))}
    ratio = (w['+'] - w['-']) / (w['+'] + w['-'] + w['0'])
    lz = 2 * ion.info['n_holes'] * ratio
    _label(a2, 9, 0.6 * diff.min(), f"orbital sum rule: |<L$_z$>| = {abs(lz):.3f}")
    _save(fig, 'example_3_co_xmcd')


def example_4():
    """Ni2+ L3 RIXS map: d-d excitations as a function of incident energy."""
    ion = xr.solve_ion(xr.IonModel(atom='Ni', valence=2, ten_dq=1.1,
                                   temperature=10), verbose=False)
    xs = ion.xas(np.linspace(-3, 5, 800), gamma_c=0.3)
    ominc = np.linspace(-3, 5, 161)
    eloss = np.linspace(-0.5, 4.5, 500)
    res = ion.rixs(ominc, eloss, pol_in='LV', pol_out='unpolarized',
                   gamma_c=0.3, gamma_f=0.06)
    res.save(outdir=OUT, prefix='ex4_Ni2_L3_map')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True,
                                 gridspec_kw=dict(width_ratios=(1, 3.2)))
    a1.plot(xs.intensity / xs.intensity.max(), xs.omega, color=INK, lw=1.2)
    a1.set_xlabel('XAS')
    a1.set_ylabel('incident energy relative to the L$_3$ peak (eV)')
    a1.set_xticks([])
    mesh = a2.pcolormesh(res.eloss, res.ominc, res.intensity / res.intensity.max(),
                         cmap=SEQ, shading='auto', vmax=0.6, rasterized=True)
    fig.colorbar(mesh, ax=a2, label='RIXS intensity (rel.)', extend='max')
    a2.set_xlabel('energy loss (eV)')
    a2.set_title('Ni$^{2+}$ O$_h$, 10Dq = 1.1 eV, $\\sigma$ incident, no analyser',
                 fontsize=10, loc='left')
    _save(fig, 'example_4_ni_rixs_map')


def example_5():
    """Cu2+ in a square-planar field: the d-d excitations depend on polarisation."""
    ion = xr.solve_ion(xr.IonModel(atom='Cu', valence=2, cf='square_planar',
                                   ten_dq=1.6, cf_ds=0.25, edge='L3',
                                   temperature=10), verbose=False)
    eloss = np.linspace(-0.3, 3.0, 800)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, (pol, name) in zip(SERIES, [('LH', '$\\pi$ incident'), ('LV', '$\\sigma$ incident')]):
        res = ion.rixs([0.0], eloss, pol_in=pol, pol_out='unpolarized',
                       thin=np.radians(20), thout=np.radians(130),
                       gamma_c=0.3, gamma_f=0.04, gauss_fwhm=0.05)
        res.save(outdir=OUT, prefix=f'ex5_Cu2_{pol}')
        ax.plot(res.eloss, res.intensity[0], color=c, label=name)
    ax.set_xlabel('energy loss (eV)')
    ax.set_ylabel('RIXS intensity (arb. units)')
    ax.set_title('Cu$^{2+}$ D$_{4h}$ L$_3$ RIXS at the peak, 2$\\theta$ = 150$^\\circ$, '
                 'grazing incidence', fontsize=10, loc='left')
    ax.legend()
    _save(fig, 'example_5_cu_rixs_polarization')


def example_6():
    """Lanthanide M4,5 XAS: Ce3+, Nd3+, Gd3+ (Lanczos), Er3+, Yb3+."""
    ions = [('Ce', 3), ('Nd', 3), ('Gd', 3), ('Er', 3), ('Yb', 3)]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    omega = np.linspace(-14, 45, 3000)
    for n, (atom, val) in enumerate(ions):
        ion = xr.solve_ion(xr.IonModel(atom=atom, valence=val, edge='M45',
                                       temperature=10), verbose=False)
        res = ion.xas(omega, gamma_c=0.25, gauss_fwhm=0.2, normalize=True)
        y = res.intensity + 1.1 * n
        ax.plot(res.omega, y, color=INK, lw=1.1)
        ax.fill_between(res.omega, 1.1 * n, y, color=SERIES[0], alpha=0.12)
        _label(ax, -13.5, 1.1 * n + 0.45, f"{atom}$^{{{val}+}}$  4f$^{{{ion.info['nd']}}}$"
               f"\n{ion.method}")
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('photon energy relative to the M$_5$ peak (eV)')
    ax.set_title('M$_{4,5}$ (3d $\\to$ 4f) XAS of trivalent lanthanides',
                 fontsize=10, loc='left')
    _save(fig, 'example_6_lanthanide_m45')


EXAMPLES = {n: f for n, f in enumerate(
    [example_1, example_2, example_3, example_4, example_5, example_6], start=1)}


def main(argv):
    if not argv:
        for n, f in EXAMPLES.items():
            print(f"  {n}: {f.__doc__.strip().splitlines()[0]}")
        return 0
    todo = EXAMPLES if argv[0] == 'all' else {int(a): EXAMPLES[int(a)] for a in argv}
    for n, f in todo.items():
        print(f"example {n}: {f.__doc__.strip().splitlines()[0]}")
        f()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
