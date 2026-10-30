"""
Atomic parameters: Hartree-Fock Slater integrals, spin-orbit couplings, edge
energies and core-hole widths.

Two tabulations are read, both keyed by configuration in the edrixs
convention (``'3d8'``, ``'3d9_2p5'``) and by edge label (``'L23'``, ``'L3'``):

``'crispy'``
    ``atom_data/<El>.json``, built by ``tools/make_atom_data.py`` from the
    Crispy parameter files -- 50 elements, 3d/4d/5d/4f/5f valence shells,
    1s to 5p core holes.
``'edrixs'``
    the atom data shipped with edrixs -- 14 elements.

For XAS and RIXS the configuration that matters besides the ground state is
the *intermediate* state :math:`d^{n+1}\\underline{c}`: one core electron has
been promoted into the valence shell.  (That is what
:func:`edrixs.get_atom_data` returns too; the XPS final state
:math:`d^{n}\\underline{c}` is a different row of the same tables.)
"""

from __future__ import annotations

__all__ = ['atomic_parameters', 'available_atoms', 'available_edges',
           'nd_from_valence', 'core_shell', 'TM_GROUP', 'DEFAULT_SHELL',
           'ELEMENT_INDEX', 'SOURCES', 'EDGE_SOURCES', 'ATOM_DATA_DIR']

import json
import os
import warnings
from typing import Optional

import edrixs

#: directory of the Crispy-derived tables written by ``make_atom_data.py``
ATOM_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'atom_data')

#: order in which ``source='auto'`` tries the Slater/SOC tables.  Both are
#: Cowan RCN/RCG Hartree-Fock values; Crispy comes first because a handful of
#: edrixs rows near a filled shell repeat a neighbouring occupancy.
SOURCES = ('crispy', 'edrixs')

#: edge energies and core-hole widths are merged separately, edrixs first --
#: its widths differ from the Keski-Rahkonen and Krause values that fill the
#: gaps, and this keeps the elements edrixs covers on its own numbers.
EDGE_SOURCES = ('edrixs', 'crispy')

#: group number (valence-shell + outer-s electrons of the neutral atom), so
#: that ``nd = TM_GROUP[atom] - valence``.  Extended from ``_index.json``.
TM_GROUP = {'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11}

#: shell that a formal valence refers to.  Extended from ``_index.json``.
DEFAULT_SHELL = {'Ti': '3d', 'V': '3d', 'Cr': '3d', 'Mn': '3d', 'Fe': '3d',
                 'Co': '3d', 'Ni': '3d', 'Cu': '3d'}

#: edge label -> core sub-shell, in the order they are listed
EDGE_ORDER = {'1s': 'K', '2s': 'L1', '2p': 'L23', '3s': 'M1', '3p': 'M23',
              '3d': 'M45', '4s': 'N1', '4p': 'N23', '4d': 'N45', '4f': 'N67',
              '5s': 'O1', '5p': 'O23', '5d': 'O45'}


def _load_element_index() -> dict:
    path = os.path.join(ATOM_DATA_DIR, '_index.json')
    try:
        with open(path) as fh:
            index = json.load(fh)
    except (OSError, ValueError):
        return {}
    for sym, meta in index.items():
        TM_GROUP.setdefault(sym, meta['group'])
        DEFAULT_SHELL.setdefault(sym, meta['valence_shell'])
    return index


#: ``{symbol: {'Z', 'group', 'valence_shell'}}`` for every tabulated element
ELEMENT_INDEX = _load_element_index()

_TABLE_CACHE: dict = {}


def nd_from_valence(atom: str, valence: int) -> int:
    """Valence-shell occupancy of ``atom`` in formal oxidation state ``valence``."""
    atom = atom.strip()
    if atom not in TM_GROUP:
        raise ValueError(f"no group number tabulated for {atom!r}; set nd "
                         f"explicitly. Tabulated: {sorted(TM_GROUP)}")
    nd = TM_GROUP[atom] - valence
    letter = DEFAULT_SHELL.get(atom, '3d')[-1]
    norb = edrixs.info_atomic_shell()[letter][1]
    if not 0 <= nd <= norb:
        raise ValueError(f"{atom}{valence:+d} gives an impossible "
                         f"{DEFAULT_SHELL.get(atom, '3d')}^{nd}")
    return nd


def core_shell(edge: str) -> dict:
    """Describe the core level of an edge label.

    ``'L23'`` is the whole 2p shell; ``'L3'`` and ``'L2'`` are its
    :math:`p_{3/2}` and :math:`p_{1/2}` sub-shells, which edrixs can treat on
    their own (a smaller intermediate space that neglects L3-L2 mixing).

    Returns ``{'shell': '2p', 'letter': 'p', 'sub': 'p32', 'norb': 4,
    'norb_full': 6}``.
    """
    name = edrixs.edge_to_shell_name(edge.strip(), with_main_qn=True)  # '2p32'
    shell, sub = name[:2], name[1:]
    info = edrixs.info_atomic_shell()
    return dict(shell=shell, letter=shell[1], sub=sub, norb=info[sub][1],
                norb_full=info[shell[1]][1])


def _atom_table(atom: str, source: str) -> Optional[dict]:
    """The raw parameter table of one element from one source, or ``None``."""
    atom = atom.strip()
    if (atom, source) in _TABLE_CACHE:
        return _TABLE_CACHE[atom, source]
    try:
        if source == 'crispy':
            with open(os.path.join(ATOM_DATA_DIR, f'{atom}.json')) as fh:
                table = json.load(fh)
        elif source == 'edrixs':
            from importlib.resources import files
            table = json.loads(files('edrixs.atom_data')
                               .joinpath(f'{atom}.json').read_text())
        else:
            raise ValueError(f"unknown parameter source {source!r}; "
                             f"expected 'auto' or one of {SOURCES}")
    except (FileNotFoundError, ModuleNotFoundError):
        table = None
    _TABLE_CACHE[atom, source] = table
    return table


def _config_record(table: dict, source: str, key: str) -> dict:
    """``{'slater': {name: value}, 'soc_v': .., 'soc_c': ..}`` for one config.

    Crispy tables store the Slater integrals by name; edrixs tables store a
    bare list in the order of :func:`edrixs.slater_integrals_name`.
    """
    entry = table[key]
    if source == 'crispy':
        return dict(slater=dict(entry['slater']),
                    soc_v=float(entry['soc_v']), soc_c=float(entry['soc_c']))
    parts = key.split('_')
    if len(parts) > 2:
        raise ValueError(f"{key!r} names more than a valence and a core shell")
    letters = tuple(part[1] for part in parts)
    names = edrixs.slater_integrals_name(letters, label=('d', 'c')[:len(letters)])
    soc = list(entry['soc'])
    return dict(slater=dict(zip(names, entry['slater'])),
                soc_v=float(soc[0]), soc_c=float(soc[-1]) if len(soc) > 1 else 0.0)


def _find_config(table: dict, template: str, n: int) -> tuple:
    """``(key, exact)`` for the configuration nearest ``template.format(n)``."""
    key = template.format(n)
    if key in table:
        return key, True
    for dn in range(1, 15):
        for trial in (n + dn, n - dn):
            k = template.format(trial)
            if k in table:
                return k, False
    return None, False


def _pick_source(atom: str, source: str, tmpl_i: str, n_i: int,
                 tmpl_n: str, n_n: int) -> tuple:
    """Choose the table that best covers both configurations.

    Returns ``(source, table, key_i, key_n)``.  A source that has the
    intermediate (core-hole) configuration beats one that does not, and an
    exact occupancy beats a neighbouring one; ties go to :data:`SOURCES` order.
    """
    wanted = SOURCES if source == 'auto' else (source,)
    best = None
    for s in wanted:
        table = _atom_table(atom, s)
        if table is None:
            continue
        key_i, exact_i = _find_config(table, tmpl_i, n_i)
        if key_i is None:
            continue
        key_n, exact_n = _find_config(table, tmpl_n, n_n)
        score = (key_n is not None, exact_i, exact_n)
        if best is None or score > best[0]:
            best = (score, s, table, key_i, key_n, exact_i, exact_n)
    if best is None:
        known = [s for s in wanted if _atom_table(atom, s) is not None]
        if not known:
            raise ValueError(
                f"no atomic parameters for {atom!r} in {list(wanted)}. The "
                f"Crispy tables cover {', '.join(available_atoms())}.")
        raise ValueError(f"no initial-state configuration like "
                         f"{tmpl_i.format(n_i)!r} for {atom!r} in {known}")
    _, s, table, key_i, key_n, exact_i, exact_n = best
    for key, exact, tmpl, n, what in ((key_i, exact_i, tmpl_i, n_i, 'initial'),
                                      (key_n, exact_n, tmpl_n, n_n, 'intermediate')):
        if key is not None and not exact:
            warnings.warn(f"{atom} {what}-state configuration "
                          f"{tmpl.format(n)!r} is not tabulated; using {key!r} "
                          f"instead. Check the Slater integrals if this matters.")
    return s, table, key_i, key_n


def available_atoms(source: str = 'crispy') -> list:
    """Elements the parameter tables cover."""
    if source == 'crispy':
        return sorted(ELEMENT_INDEX)
    if source == 'edrixs':
        from importlib.resources import files
        return sorted(p.name[:-5] for p in files('edrixs.atom_data').iterdir()
                      if p.name.endswith('.json'))
    raise ValueError(f"unknown source {source!r}")


def available_edges(atom: str, source: str = 'auto',
                    shell: Optional[str] = None) -> list:
    """Core levels for which ``atom`` has core-hole Slater integrals.

    Whole-shell labels only (``'L23'``, not ``'L3'``); every one of them can
    also be computed as its spin-orbit sub-shells.
    """
    atom = atom.strip()
    shell = shell or DEFAULT_SHELL.get(atom, '3d')
    info = edrixs.info_atomic_shell()
    out = set()
    for s in (SOURCES if source == 'auto' else (source,)):
        for key in (_atom_table(atom, s) or {}):
            parts = key.split('_')
            if not key[0].isdigit() or len(parts) != 2 or \
                    not parts[0].startswith(shell):
                continue
            core, occ = parts[1][:2], parts[1][2:]
            # a core *hole*, not an empty orbital such as edrixs's '3d8_4p0'
            if occ.isdigit() and int(occ) == info[core[1]][1] - 1:
                out.add(core)
    return [EDGE_ORDER[c] for c in EDGE_ORDER if c in out]


def atomic_parameters(atom: str, shell: str = '3d', nd: int = 8,
                      edge: str = 'L23', source: str = 'auto') -> dict:
    """Slater integrals, SOC and edge data for an XAS / RIXS calculation.

    The initial state is :math:`v^{n}` and the intermediate state is
    :math:`v^{n+1}\\underline{c}`.

    Returns
    -------
    dict
        ``names``      : Slater integral names, ``['F0_dd', ..., 'F2_cc']``
        ``slater_i``   : ``{name: value}``, initial state (valence-only)
        ``slater_n``   : ``{name: value}``, intermediate state
        ``soc_v_i``, ``soc_v_n``, ``soc_c`` : spin-orbit couplings (eV)
        ``edge_ene``   : tabulated edge energies (eV), highest first
        ``gamma_c``    : tabulated core-hole widths as Lorentzian HWHM (eV)
        ``source``, ``edge_source``, ``config_i``, ``config_n``
        ``v_letter``, ``core`` (see :func:`core_shell`)
    """
    atom = atom.strip()
    v_letter = shell.strip()[-1]
    norb_v = edrixs.info_atomic_shell()[v_letter][1]
    if not 0 <= nd < norb_v:
        raise ValueError(f"{shell}^{nd}: absorption needs at least one valence hole")
    core = core_shell(edge)
    names = edrixs.slater_integrals_name((v_letter, core['sub']), label=('d', 'c'))
    tmpl_n = shell + '{}_' + core['shell'] + str(core['norb_full'] - 1)
    src, table, key_i, key_n = _pick_source(atom, source, shell + '{}', nd,
                                            tmpl_n, nd + 1)

    rec_i = _config_record(table, src, key_i)
    slater_i = {nm: (rec_i['slater'].get(nm, 0.0) if nm.endswith('_dd') else 0.0)
                for nm in names}
    if key_n is None:
        warnings.warn(
            f"no {core['shell']} core-hole configuration for {atom} in {src!r}: "
            f"the core-valence Slater integrals and the core spin-orbit coupling "
            f"are set to zero. Supply slater_override_n={{'G1_dc': ...}} and "
            f"soc_c=..., or use one of {available_edges(atom, source, shell)}.")
        slater_n = dict(slater_i)
        soc_v_n, soc_c = rec_i['soc_v'], 0.0
    else:
        rec_n = _config_record(table, src, key_n)
        slater_n = {nm: rec_n['slater'].get(nm, 0.0) for nm in names}
        soc_v_n, soc_c = rec_n['soc_v'], rec_n['soc_c']

    res = dict(names=names, slater_i=slater_i, slater_n=slater_n,
               soc_v_i=rec_i['soc_v'], soc_v_n=soc_v_n, soc_c=soc_c,
               v_letter=v_letter, core=core, config_i=key_i, config_n=key_n,
               source=src)

    res['edge_ene'], res['gamma_c'], res['edge_source'] = [0.0], [0.2], None
    for s in (EDGE_SOURCES if source == 'auto' else (source,)):
        t = _atom_table(atom, s)
        if t is not None and edge.strip() in t:
            res['edge_ene'] = list(t[edge.strip()]['ene'])
            # both tables give the FWHM; a Lorentzian HWHM is what we broaden with
            res['gamma_c'] = [g / 2 for g in t[edge.strip()]['gamma']]
            res['edge_source'] = s
            break
    return res
