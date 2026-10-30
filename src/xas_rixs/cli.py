"""
Command-line interface::

    xas-rixs xas  --atom Ni --valence 2 --edge L23 --ten_dq 1.1 --plot
    xas-rixs xas  --atom Co --valence 2 --ext_B "[0,0,0.01]" --pol left --plot
    xas-rixs rixs --atom Ni --valence 2 --ominc "[0, 1.6]" --pol_in LV --plot
    xas-rixs rixs --config my_ion.json --ominc "[0]" --thin 20 --thout 130

Every :class:`~xas_rixs.IonModel` field is an option of the same name (lists
and dicts as JSON) and ``--config`` reads the same names from a JSON file.
Angles on the command line are in degrees.
"""

from __future__ import annotations

__all__ = ['main', 'build_parser']

import argparse
import json
from dataclasses import fields

import numpy as np

from .model import IonModel
from .solver import solve_ion
from .spectra import plot_rixs_map, plot_xas, rixs, save_result, xas


def _coerce(value: str, ftype):
    txt = str(ftype)
    if value in ('none', 'None', 'null'):
        return None
    if ftype is bool or ('bool' in txt and 'Optional' not in txt):
        return str(value).lower() in ('1', 'true', 'yes', 'on')
    if 'dict' in txt or 'Sequence' in txt:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value if 'str' in txt else [float(v) for v in value.replace(',', ' ').split()]
    if 'int' in txt and 'float' not in txt:
        return int(float(value))
    if 'float' in txt:
        return float(value)
    return value


def _pol(value: str):
    """A polarisation spec from the command line: a name, an angle in degrees,
    or a JSON vector."""
    try:
        v = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(v, (int, float)):
        return ('linear', np.radians(v))
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='xas-rixs',
        description='XAS and RIXS of a single ion in the crystal-field multiplet '
                    'model, built on edrixs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('kind', choices=('xas', 'rixs'))
    p.add_argument('--config', help='JSON file with IonModel fields')
    p.add_argument('--prefix', help='output file prefix (default: the label)')
    p.add_argument('--outdir', default='.', help='directory for the output files')
    p.add_argument('--no-save', action='store_true', help='do not write files')
    p.add_argument('--plot', action='store_true', help='show a plot')
    p.add_argument('--quiet', action='store_true')

    s = p.add_argument_group('spectrum')
    s.add_argument('--pol', default='isotropic',
                   help='XAS polarisation: isotropic, LH, LV, left, right, an '
                        'angle in degrees, or a JSON vector')
    s.add_argument('--pol_in', default='LH', help='RIXS incident polarisation')
    s.add_argument('--pol_out', default='unpolarized', help='RIXS scattered polarisation')
    s.add_argument('--ominc', default='[0.0]', help='RIXS incident energies, JSON list')
    s.add_argument('--eloss', help="RIXS loss grid, e.g. --eloss=-0.5,4,300")
    s.add_argument('--thin', type=float, help='incidence angle (deg); default '
                   '90 for XAS, 75 for RIXS')
    s.add_argument('--thout', type=float, default=75.0, help='exit angle (deg)')
    s.add_argument('--phi', type=float, default=0.0, help='azimuth (deg)')
    s.add_argument('--gamma_f', type=float, default=0.05, help='final-state HWHM (eV)')
    s.add_argument('--gauss_fwhm', type=float, default=0.0, help='resolution FWHM (eV)')

    grp = p.add_argument_group('ion (see IonModel for the documentation)')
    for f in fields(IonModel):
        grp.add_argument(f'--{f.name}', default=None, metavar='V',
                         help=f'{f.type} (default: {getattr(IonModel(), f.name)!r})')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    kw = {}
    if args.config:
        with open(args.config) as fh:
            kw.update(json.load(fh))
    for f in fields(IonModel):
        v = getattr(args, f.name)
        if v is not None:
            kw[f.name] = _coerce(v, f.type)
    ion = solve_ion(IonModel(**kw), verbose=not args.quiet)

    phi = np.radians(args.phi)
    if args.kind == 'xas':
        thin = np.radians(90.0 if args.thin is None else args.thin)
        res = xas(ion, pol=_pol(args.pol), thin=thin, phi=phi,
                  gauss_fwhm=args.gauss_fwhm)
    else:
        thin = np.radians(75.0 if args.thin is None else args.thin)
        eloss = None
        if args.eloss:
            a, b, n = args.eloss.split(',')
            eloss = np.linspace(float(a), float(b), int(n))
        res = rixs(ion, json.loads(args.ominc), eloss, pol_in=_pol(args.pol_in),
                   pol_out=_pol(args.pol_out), thin=thin,
                   thout=np.radians(args.thout), phi=phi, gamma_f=args.gamma_f,
                   gauss_fwhm=args.gauss_fwhm)

    if not args.no_save:
        written = save_result(res, prefix=args.prefix, outdir=args.outdir)
        if not args.quiet:
            print('  wrote ' + ', '.join(written.values()))
    if args.plot:
        import matplotlib.pyplot as plt
        if args.kind == 'xas':
            plot_xas(res)
        elif len(res.ominc) > 5:
            plot_rixs_map(res)
        else:
            _, ax = plt.subplots(figsize=(7, 4.5))
            for k, om in enumerate(res.ominc):
                ax.plot(res.eloss, res.intensity[k], label=f'{om:g} eV')
            ax.set_xlabel('energy loss (eV)')
            ax.set_ylabel('intensity (arb. units)')
            ax.legend(title='incident energy', fontsize=8)
        plt.show()
    return 0
