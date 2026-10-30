#!/usr/bin/env python
"""
Build ``atom_data/`` -- the Hartree-Fock atomic parameter tables that
:mod:`xas_rixs` uses when edrixs has nothing tabulated for an element or a
core level.

Three upstream tables are merged, none of which is redistributed here in its
original form:

``Slater integrals and spin-orbit couplings``
    Crispy's ``src/crispy/quanty/parameters/<El>.h5``
    (https://github.com/mretegan/crispy, MIT).  These are Cowan RCN/RCG
    Hartree-Fock values -- the same code and the same numbers edrixs
    tabulates, at four decimals and for 50 elements instead of 14, covering
    3d/4d/5d/4f/5f valence shells and 1s...5p core holes.  Crispy stores one
    group per *configuration*, e.g. ``2p5,3d8``, covering both the XPS final
    state (a core hole, valence occupancy unchanged) and the XAS/RIXS
    intermediate state (a core hole, one more valence electron).

``Core-level binding energies``
    Elam, Ravel and Sieber, Radiat. Phys. Chem. 63, 121 (2002), as shipped in
    XrayDB's ``data_sources/elam.dat`` (https://github.com/xraypy/XrayDB).
    These agree exactly with the ``ene`` entries of the edrixs atom data.

``Core-hole widths``
    Keski-Rahkonen and Krause, At. Data Nucl. Data Tables 14, 139 (1974), as
    shipped in XrayDB's ``data_sources/keskirahkonen_krause.dat``; FWHM in eV.
    NOTE these are *not* the numbers edrixs tabulates -- edrixs uses an
    unattributed and generally broader set -- so :mod:`xas_rixs` prefers the
    edrixs widths wherever it has them and only falls back to these.

Output: one JSON file per element, keyed like the edrixs atom data
(``'3d8'``, ``'3d8_2p5'``, ``'L23'``, ...) but with the Slater integrals as a
name -> value mapping rather than a bare list, so the file does not depend on
edrixs's internal ordering convention.

Usage::

    python3 tools/make_atom_data.py           # download sources, write the tables
    python3 make_atom_data.py --cache DIR     # keep/reuse the downloads in DIR
    python3 make_atom_data.py --offline       # fail instead of downloading
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

#: repository root, so that --out and the download cache do not depend on cwd
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'src' / 'xas_rixs' / 'atom_data'

CRISPY = ('https://raw.githubusercontent.com/mretegan/crispy/main/'
          'src/crispy/quanty/parameters/{}.h5')
XRAYDB = 'https://raw.githubusercontent.com/xraypy/XrayDB/master/data_sources/{}'

ELEMENTS = """Ac Ag Am Au Bk Ce Cf Cm Co Cr Cu Dy Er Eu Fe Gd Hf Ho Ir La Mn Mo
Nb Nd Ni Np Os Pa Pd Pm Pr Pt Pu Re Rh Ru Sc Sm Ta Tb Tc Th Ti Tm U V W Y Yb
Zr""".split()

#: atomic number of every element up to Cf
Z_OF = {s: i + 1 for i, s in enumerate("""H He Li Be B C N O F Ne Na Mg Al Si P
S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc
Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb
Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk
Cf""".split())}

#: Z of the last element before each valence block starts filling, so that
#: ``group = Z - BLOCK_OFFSET[block]`` and ``n_valence = group - charge``.
#: Checked against every configuration Crispy tabulates (see ``--check``).
BLOCK_OFFSET = {'3d': 18, '4d': 36, '4f': 54, '5d': 68, '5f': 86}

#: the sub-shells that make up each edrixs-style edge label, highest binding
#: energy first
EDGE_LEVELS = {
    'K': ['K'],
    'L1': ['L1'], 'L2': ['L2'], 'L3': ['L3'], 'L23': ['L2', 'L3'],
    'M1': ['M1'], 'M2': ['M2'], 'M3': ['M3'], 'M23': ['M2', 'M3'],
    'M4': ['M4'], 'M5': ['M5'], 'M45': ['M4', 'M5'],
    'N1': ['N1'], 'N2': ['N2'], 'N3': ['N3'], 'N23': ['N2', 'N3'],
    'N4': ['N4'], 'N5': ['N5'], 'N45': ['N4', 'N5'],
    'N6': ['N6'], 'N7': ['N7'], 'N67': ['N6', 'N7'],
    'O1': ['O1'], 'O2': ['O2'], 'O3': ['O3'], 'O23': ['O2', 'O3'],
    'O4': ['O4'], 'O5': ['O5'], 'O45': ['O4', 'O5'],
    'P1': ['P1'], 'P2': ['P2'], 'P3': ['P3'], 'P23': ['P2', 'P3'],
}

_PARAM = re.compile(r'^([FG])(\d+)\((\d[spdf]),(\d[spdf])\)$')
_ZETA = re.compile(r'^ζ\((\d[spdf])\)$')


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------

def fetch(url: str, path: str, offline: bool) -> str:
    if os.path.exists(path):
        return path
    if offline:
        raise SystemExit(f"missing {path} and --offline was given ({url})")
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    print(f"  downloading {os.path.basename(path)} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(path, 'wb') as fh:
        fh.write(r.read())
    return path


# ---------------------------------------------------------------------------
# the three tables
# ---------------------------------------------------------------------------

def read_edge_energies(path: str) -> dict:
    """``{element: {'L3': energy_eV, ...}}`` from Elam's tabulation."""
    out, el = {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith('Element'):
                el = line.split()[1]
                out[el] = {}
            elif line.startswith('Edge '):
                p = line.split()
                out[el][p[1]] = float(p[2])
    return out


def read_core_widths(path: str) -> dict:
    """``{element: {'L3': FWHM_eV, ...}}`` from Keski-Rahkonen and Krause."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) == 4:
                out.setdefault(p[1], {})[p[2]] = float(p[3])
    return out


def read_crispy(path: str, valence_shell: str) -> dict:
    """Slater integrals and SOC per configuration from one Crispy HDF5 file.

    Returns ``{'3d8_2p5': {'slater': {'F2_dd': ..}, 'soc_v': .., 'soc_c': ..}}``
    with the Crispy configuration names translated to the edrixs convention.
    """
    import h5py
    out = {}
    with h5py.File(path, 'r') as h5:
        for conf in h5:
            parts = conf.split(',')
            valence = parts[-1]                       # e.g. '3d8'
            core = parts[0] if len(parts) > 1 else None   # e.g. '2p5'
            if not valence.startswith(valence_shell):
                raise ValueError(f"{path}: {conf!r} is not a {valence_shell} "
                                 f"configuration")
            v_shell, c_shell = valence_shell, core[:2] if core else None
            key = valence if core is None else f'{valence}_{core}'

            slater, soc_v, soc_c = {'F0_dd': 0.0}, 0.0, 0.0
            for name, node in h5[conf]['Atomic'].items():
                val = round(float(node[()]), 6)
                if name.startswith('U('):
                    # Crispy stores U as a placeholder 0; xas_rixs builds the
                    # monopoles from the user's U_dd / U_dc instead
                    continue
                m = _PARAM.match(name)
                if m:
                    kind, k, a, b = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                    shells = {a, b}
                    if shells == {v_shell}:
                        slater[f'{kind}{k}_dd'] = val
                    elif shells == {v_shell, c_shell}:
                        slater[f'{kind}{k}_dc'] = val
                    else:
                        raise ValueError(f"{path}/{conf}: unexpected {name!r}")
                    continue
                m = _ZETA.match(name)
                if m:
                    if m.group(1) == v_shell:
                        soc_v = val
                    elif m.group(1) == c_shell:
                        soc_c = val
                    else:
                        raise ValueError(f"{path}/{conf}: unexpected {name!r}")
                    continue
                raise ValueError(f"{path}/{conf}: unparsed parameter {name!r}")
            if core is not None:
                slater.setdefault('F0_dc', 0.0)
            out[key] = dict(slater=slater, soc_v=soc_v, soc_c=soc_c)
    return out


# ---------------------------------------------------------------------------
# assembling
# ---------------------------------------------------------------------------

def valence_shell_of(configs) -> str:
    shells = {c.split(',')[-1][:2] for c in configs}
    if len(shells) != 1:
        raise ValueError(f"more than one valence shell: {shells}")
    return shells.pop()


def build(element: str, cache: str, energies: dict, widths: dict,
          offline: bool) -> dict:
    import h5py
    path = fetch(CRISPY.format(element), os.path.join(cache, f'{element}.h5'),
                 offline)
    with h5py.File(path, 'r') as h5:
        shell = valence_shell_of(h5.keys())
    out = read_crispy(path, shell)

    group = Z_OF[element] - BLOCK_OFFSET[shell]
    out['_meta'] = dict(
        symbol=element, Z=Z_OF[element], group=group, valence_shell=shell,
        slater_source='crispy', energy_source='elam', width_source='keski_krause',
        generated=str(date.today()))

    for label, levels in EDGE_LEVELS.items():
        ene = [energies.get(element, {}).get(s) for s in levels]
        if any(e is None for e in ene):
            continue
        gam = [widths.get(element, {}).get(s) for s in levels]
        if any(g is None for g in gam):
            continue
        # Elam has M2 < M3 for Co; edrixs sorts these, and so do we
        pair = sorted(zip(ene, gam), key=lambda t: -t[0])
        out[label] = dict(ene=[p[0] for p in pair], gamma=[p[1] for p in pair])
    return out


def check(out: dict, element: str) -> None:
    """Sanity checks that would have caught a mis-parse."""
    meta = out['_meta']
    shell, group = meta['valence_shell'], meta['group']
    occ = sorted({int(k.split('_')[0][2:]) for k in out if k[0].isdigit()})
    # Crispy generates the ground-state configurations of every tabulated
    # charge plus one extra electron (the XAS final state)
    expect = list(range(min(occ), max(occ) + 1))
    assert occ == expect, f"{element}: gaps in the occupancies {occ}"
    assert 0 <= max(occ) <= 2 * (2 * 'spdf'.index(shell[1]) + 1), element
    assert group - min(occ) > 0, f"{element}: implausible group number {group}"
    norb = 2 * (2 * 'spdf'.index(shell[1]) + 1)
    for k, v in out.items():
        if not k[0].isdigit():
            continue
        n = int(k.split('_')[0][2:])
        if 2 <= n <= norb - 2:      # an open shell must have multipoles and SOC
            assert v['slater'].get('F2_dd'), f"{element}/{k}: no F2_dd"
            assert v['soc_v'] > 0, f"{element}/{k}: no valence SOC"
        if '_' in k:
            core = k.split('_')[1][:2]
            assert (v['soc_c'] > 0) == (core[1] != 's'), f"{element}/{k}"
            assert v['slater'].get('G%d_dc' % abs(
                'spdf'.index(core[1]) - 'spdf'.index(shell[1])) or 0) is not None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cache', default=str(ROOT / '.atom_data_cache'),
                    help='where the upstream files are downloaded to')
    ap.add_argument('--out', default=str(DEFAULT_OUT),
                    help='output directory (default: the packaged tables)')
    ap.add_argument('--offline', action='store_true',
                    help='use only what is already in --cache')
    ap.add_argument('elements', nargs='*', default=None,
                    help='elements to build (default: all 50)')
    args = ap.parse_args(argv)

    try:
        import h5py            # noqa: F401
    except ImportError:
        raise SystemExit("this script needs h5py to read the Crispy tables")

    print(f"sources -> {args.cache}/")
    energies = read_edge_energies(
        fetch(XRAYDB.format('elam.dat'),
              os.path.join(args.cache, 'elam.dat'), args.offline))
    widths = read_core_widths(
        fetch(XRAYDB.format('keskirahkonen_krause.dat'),
              os.path.join(args.cache, 'keskirahkonen_krause.dat'), args.offline))

    os.makedirs(args.out, exist_ok=True)
    todo = args.elements or ELEMENTS
    total = 0
    for el in todo:
        out = build(el, args.cache, energies, widths, args.offline)
        check(out, el)
        n_conf = sum(1 for k in out if k[0].isdigit())
        n_edge = sum(1 for k in out if k in EDGE_LEVELS)
        total += n_conf
        with open(os.path.join(args.out, f'{el}.json'), 'w') as fh:
            json.dump(out, fh, indent=0, sort_keys=True, ensure_ascii=True)
            fh.write('\n')
        print(f"  {el:<3s} {out['_meta']['valence_shell']}  "
              f"{n_conf:3d} configurations, {n_edge:2d} edges")

    # a small index so xas_rixs can resolve valences without opening 50 files
    index = {}
    for name in sorted(os.listdir(args.out)):
        if name.endswith('.json') and not name.startswith('_'):
            with open(os.path.join(args.out, name)) as fh:
                meta = json.load(fh)['_meta']
            index[meta['symbol']] = {k: meta[k] for k in
                                     ('Z', 'group', 'valence_shell')}
    with open(os.path.join(args.out, '_index.json'), 'w') as fh:
        json.dump(index, fh, indent=0, sort_keys=True)
        fh.write('\n')

    print(f"{len(todo)} elements, {total} configurations -> {args.out}/ "
          f"({len(index)} in _index.json)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
