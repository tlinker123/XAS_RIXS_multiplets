#!/usr/bin/env python
"""
Validation tests for :mod:`xas_rixs`.

Run with::

    pytest                       # everything
    pytest -s                    # ... showing every individual check
    pytest -m "not slow"         # skip the heavy cases

Two kinds of reference are used: edrixs itself -- its dense many-body
builders and its pure-python solvers ``ed_1v1c_py``, ``xas_1v1c_py`` and
``rixs_1v1c_py`` -- and exact sum rules.

Individual assertions go through :func:`check`, which *collects* failures so
that a test reports every bad check rather than stopping at the first one;
the ``_report_checks`` fixture turns a non-empty collection into a failure.
"""
import contextlib
import io
import warnings

import numpy as np
import pytest
import edrixs

import xas_rixs as xr
from xas_rixs import atomic

_FAILURES = []


def check(name, ok, detail=''):
    """Record one named check; see the module docstring."""
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ''))
    if not ok:
        _FAILURES.append(f"{name}   {detail}".rstrip())


@pytest.fixture(autouse=True)
def _report_checks():
    """Fail the test if any :func:`check` in it failed, listing all of them."""
    _FAILURES.clear()
    yield
    if _FAILURES:
        failed, n = list(_FAILURES), len(_FAILURES)
        _FAILURES.clear()
        pytest.fail(f"{n} check(s) failed:\n  " + "\n  ".join(failed), pytrace=False)


def rel(a, b):
    """Largest difference relative to the largest value."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def quiet(fn, *args, **kw):
    """Call an edrixs solver without its progress chatter."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


# ---------------------------------------------------------------------------
# the many-body layer
# ---------------------------------------------------------------------------

def _perm(basis):
    keys = np.array([sum(int(b) << i for i, b in enumerate(s)) for s in basis])
    return keys, np.argsort(keys)


def test_operators():
    """Sparse Fock spaces and operators must equal edrixs' dense ones."""
    print("\nmany-body operators vs. edrixs")
    rng = np.random.default_rng(3)
    for args in [(10, 8, 6, 6), (10, 9, 6, 5), (14, 2, 10, 9), (10, 3, 2, 1)]:
        ref = edrixs.get_fock_bin_by_N(*args)
        keys, _ = _perm(ref)
        check(f"Fock space {args}", np.array_equal(np.sort(keys),
                                                   xr.make_space(*args).states))

    basis_i = edrixs.get_fock_bin_by_N(10, 8, 6, 6)
    basis_n = edrixs.get_fock_bin_by_N(10, 9, 6, 5)
    _, pi = _perm(basis_i)
    _, pn = _perm(basis_n)
    space_i, space_n = xr.make_space(10, 8, 6, 6), xr.make_space(10, 9, 6, 5)

    emat = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    emat = emat + emat.conj().T
    ref = edrixs.build_opers(2, emat, basis_n)[np.ix_(pn, pn)]
    mine = xr.two_fermion_sparse(emat, space_n, space_n).toarray()
    check('two_fermion', np.allclose(ref, mine, atol=1e-10))

    umat = edrixs.get_umat_slater('dp', 1.3, 9.1, 5.7, 0.8, 6.2, 4.4, 2.7, 0.0, 0.0)
    ref = edrixs.build_opers(4, umat, basis_n)[np.ix_(pn, pn)]
    mine = xr.four_fermion_sparse(umat, space_n).toarray()
    check('four_fermion', np.allclose(ref, mine, atol=1e-10))

    # the transition operator f^dag_v c_c between the two spaces
    t = np.zeros((16, 16), dtype=complex)
    t[:10, 10:] = edrixs.get_trans_oper('dp')[2]
    ref = edrixs.two_fermion(t, basis_n, basis_i)[np.ix_(pn, pi)]
    mine = xr.two_fermion_sparse(t, space_n, space_i).toarray()
    check('transition operator', np.allclose(ref, mine, atol=1e-12))


# ---------------------------------------------------------------------------
# against the edrixs pure-python solvers
# ---------------------------------------------------------------------------

def _edrixs_reference(ion):
    """The same ion through ``edrixs.ed_1v1c_py``."""
    info = ion.info
    m = ion.model
    names = info['par']['names']
    v_letter, sub = info['par']['v_letter'], info['par']['core']['sub']
    cf = None
    if info['norb_v'] == 10 and m.cf in ('auto', 'cubic'):
        cf = edrixs.cf_cubic_d(m.ten_dq)
    zee = None if not np.any(m.ext_B) else list(m.ext_B)
    return quiet(edrixs.ed_1v1c_py, (v_letter, sub),
                 v_soc=(info['zeta_v_i'], info['zeta_v_n']), c_soc=info['zeta_c'],
                 v_noccu=info['nd'],
                 slater=([info['slater_i'][n] for n in names],
                         [info['slater_n'][n] for n in names]),
                 v_cfmat=cf, ext_B=zee, on_which=m.on_which, loc_axis=m.local_axis)


def _compare_with_edrixs(name, model, xas_pols, rixs_pols, thin=0.7, thout=1.1,
                         phi=0.4, scatter_axis=None):
    ion = xr.solve_ion(model, verbose=False)
    ev_i, ev_n, top = _edrixs_reference(ion)
    gs = list(range(int(model.n_gs)))
    check(f"{name}: initial energies", np.allclose(ev_i, ion.E_i, atol=1e-9))
    if ion.method == 'exact':
        check(f"{name}: intermediate energies", np.allclose(ev_n, ion.E_n, atol=1e-9))

    gamma = 0.3
    raw = np.linspace(ion.to_raw(-8), ion.to_raw(25), 700)
    for pol in xas_pols:
        ref = quiet(edrixs.xas_1v1c_py, ev_i, ev_n, top, raw, gamma_c=gamma,
                    thin=thin, phi=phi, pol_type=[pol], gs_list=gs,
                    temperature=model.temperature, scatter_axis=scatter_axis)[:, 0]
        spec = 'isotropic' if pol[0] == 'isotropic' else \
            (pol if pol[0] == 'linear' else pol[0])
        mine = xr.xas(ion, ion.from_raw(raw), pol=spec, thin=thin, phi=phi,
                      scatter_axis=scatter_axis, gamma_c=gamma).intensity
        check(f"{name}: XAS {pol}", rel(mine, ref) < 1e-9, f"rel={rel(mine, ref):.1e}")

    xs = xr.xas(ion, gamma_c=gamma)
    om = xs.poles[np.argsort(xs.poles[:, 1])[-2:], 0]       # two strongest poles
    eloss = np.linspace(-0.5, 6, 400)
    for it, alpha, jt, beta in rixs_pols:
        ref = quiet(edrixs.rixs_1v1c_py, ev_i, ev_n, top, ion.to_raw(om), eloss,
                    gamma_c=gamma, gamma_f=0.08, thin=thin, thout=thout, phi=phi,
                    pol_type=[(it, alpha, jt, beta)], gs_list=gs,
                    temperature=model.temperature, scatter_axis=scatter_axis)[:, :, 0]
        p_in = ('linear', alpha) if it == 'linear' else it
        p_out = ('linear', beta) if jt == 'linear' else jt
        mine = xr.rixs(ion, om, eloss, pol_in=p_in, pol_out=p_out, thin=thin,
                       thout=thout, phi=phi, scatter_axis=scatter_axis,
                       gamma_c=gamma, gamma_f=0.08).intensity
        check(f"{name}: RIXS {it}({alpha:.2f}) -> {jt}({beta:.2f})",
              rel(mine, ref) < 1e-8, f"rel={rel(mine, ref):.1e}")


LIN, CIRC = [('isotropic', 0), ('linear', 0.0), ('linear', 1.2)], [('left', 0), ('right', 0)]
RIXS_POLS = [('linear', 0.0, 'linear', 0.0), ('linear', np.pi / 2, 'linear', 0.3),
             ('left', 0, 'linear', np.pi / 2)]


def test_ni2_L23_vs_edrixs():
    """Ni2+ L2,3 with a field and a rotated crystal, dipolar."""
    print("\nNi2+ L23 vs. edrixs")
    axis = edrixs.euler_to_rmat(0.3, 0.5, -0.2)
    model = xr.IonModel(atom='Ni', valence=2, edge='L23', ten_dq=1.1,
                        ext_B=(0.01, 0.0, 0.03), local_axis=axis.tolist(),
                        temperature=40, n_gs=3, reference='peak')
    _compare_with_edrixs('Ni2+ L23', model, LIN + CIRC, RIXS_POLS)


def test_subshell_and_quadrupole_vs_edrixs():
    """A spin-orbit sub-shell (Cu2+ L3) and a quadrupolar edge (Ni2+ K pre-edge)."""
    print("\nsub-shell and quadrupolar edges vs. edrixs")
    _compare_with_edrixs('Cu2+ L3', xr.IonModel(atom='Cu', valence=2, edge='L3',
                                                ten_dq=1.5, n_gs=4, temperature=10),
                         LIN, RIXS_POLS[:2])
    _compare_with_edrixs('Ni2+ K', xr.IonModel(atom='Ni', valence=2, edge='K',
                                               ten_dq=1.0, n_gs=3, temperature=10,
                                               local_axis=edrixs.euler_to_rmat(
                                                   0.2, 0.9, 0.1).tolist()),
                         LIN, RIXS_POLS[:2], scatter_axis=edrixs.euler_to_rmat(
                             0.0, 0.3, 0.0))


def test_f_shell_vs_edrixs():
    """Ce3+ M4,5 (3d -> 4f), no crystal field, with an exchange field."""
    print("\nCe3+ M45 vs. edrixs")
    model = xr.IonModel(atom='Ce', valence=3, edge='M45', ext_B=(0, 0, 0.02),
                        n_gs=1, temperature=1)
    _compare_with_edrixs('Ce3+ M45', model, LIN + CIRC, RIXS_POLS[:2])


# ---------------------------------------------------------------------------
# large spaces
# ---------------------------------------------------------------------------

def test_lanczos_vs_exact():
    """The Ritz/Lanczos path must reproduce full diagonalisation."""
    print("\nLanczos vs. exact")
    # a Krylov space as large as the intermediate space is exact
    base = xr.IonModel(atom='Ni', valence=2, ten_dq=1.1, temperature=10)
    ex = xr.solve_ion(base, verbose=False)
    lz = xr.solve_ion(xr.replace_model(base, method='lanczos', n_kryl=100), verbose=False)
    check('Ni2+ reference energy', abs(ex.offset - lz.offset) < 1e-8)
    a, b = xr.xas(ex, pol='LV'), xr.xas(lz, pol='LV', omega=ex.xas(pol='LV').omega)
    check('Ni2+ XAS', rel(b.intensity, a.intensity) < 1e-9, f"{rel(b.intensity, a.intensity):.1e}")
    a = ex.rixs([0.0, 1.5], pol_in='LV')
    b = lz.rixs([0.0, 1.5], pol_in='LV', eloss=a.eloss)
    check('Ni2+ RIXS', rel(b.intensity, a.intensity) < 1e-8, f"{rel(b.intensity, a.intensity):.1e}")

    # a genuinely truncated Krylov space: Fe3+ (1260 intermediate states)
    base = xr.IonModel(atom='Fe', valence=3, ten_dq=1.5, temperature=10)
    ex = xr.solve_ion(base, verbose=False)
    lz = xr.solve_ion(xr.replace_model(base, method='lanczos', n_kryl=300), verbose=False)
    a = ex.xas()
    b = lz.xas(omega=a.omega)
    check('Fe3+ XAS (300 Lanczos steps)', rel(b.intensity, a.intensity) < 1e-6,
          f"{rel(b.intensity, a.intensity):.1e}")
    a = ex.rixs([-1.0, 0.0, 2.0])
    b = lz.rixs([-1.0, 0.0, 2.0], eloss=a.eloss)
    check('Fe3+ RIXS (300 Ritz states)', rel(b.intensity, a.intensity) < 1e-5,
          f"{rel(b.intensity, a.intensity):.1e}")


# ---------------------------------------------------------------------------
# exact sum rules and limits
# ---------------------------------------------------------------------------

def test_hole_sum_rule():
    """Integrated isotropic XAS is proportional to the number of holes."""
    print("\nisotropic XAS weight per hole")
    cases = [(f'{a}{v:+d} L23', dict(atom=a, valence=v, edge='L23'))
             for a, v in [('Ti', 4), ('V', 3), ('Cr', 3), ('Mn', 2), ('Fe', 2),
                          ('Co', 2), ('Ni', 2), ('Cu', 2)]]
    cases += [(f'{a}{v:+d} M45', dict(atom=a, valence=v, edge='M45'))
              for a, v in [('Ce', 3), ('Pr', 3), ('Yb', 3)]]
    for name, kw in cases:
        ion = xr.solve_ion(xr.IonModel(**kw), verbose=False)
        per_hole = ion.info['xas_sum_iso'] / ion.info['n_holes']
        # 1/3 per hole for any dipolar edge, with the edrixs normalisation
        check(f"{name}: weight per hole", abs(per_hole - 1 / 3) < 1e-10,
              f"{per_hole:.12f}")


def test_branching_ratio():
    """Without core-valence multipoles or valence SOC, L3:L2 is exactly 2:1."""
    print("\nstatistical branching ratio")
    for atom, val in [('Fe', 2), ('Co', 2), ('Ni', 2)]:
        ion = xr.solve_ion(xr.IonModel(
            atom=atom, valence=val, ten_dq=1.0, soc_v_i=0.0, soc_v_n=0.0,
            slater_override_n={'F2_dc': 0.0, 'G1_dc': 0.0, 'G3_dc': 0.0},
            reference='raw', temperature=10), verbose=False)
        p = ion.xas().poles
        mid = 0.5 * (p[:, 0].min() + p[:, 0].max())
        l3 = p[p[:, 0] < mid, 1].sum() / p[:, 1].sum()
        check(f"{atom}{val:+d}: L3 fraction", abs(l3 - 2 / 3) < 1e-10, f"{l3:.12f}")


def test_xmcd_orbital_sum_rule():
    """|int(mu+ - mu-)| / int(mu+ + mu- + mu0) = |<Lz>| / (2 n_h) for p -> d."""
    print("\nXMCD orbital sum rule")
    for atom, val in [('Co', 2), ('Fe', 2), ('Ni', 2)]:
        model = xr.IonModel(atom=atom, valence=val, ten_dq=0.8, ext_B=(0, 0, 0.05),
                            on_which='spin', temperature=5)
        ion = xr.solve_ion(model, verbose=False)
        mu = {}
        for name, vec in [('+', [1, 1j, 0]), ('-', [1, -1j, 0]), ('0', [0, 0, np.sqrt(2)])]:
            mu[name] = xr.xas(ion, pol=np.asarray(vec) / np.sqrt(2)).poles[:, 1].sum()
        ratio = abs(mu['+'] - mu['-']) / (mu['+'] + mu['-'] + mu['0'])

        H = xr.build_hamiltonians(model)
        lz = np.zeros((16, 16), dtype=complex)
        lz[:10, :10] = edrixs.get_lz(2, True)
        Lz = xr.two_fermion_sparse(lz, H['space_i'], H['space_i'])
        lz_gs = sum(pg * np.real(np.vdot(ion.V_i[:, g], Lz @ ion.V_i[:, g]))
                    for g, pg in zip(ion.gs, ion.p))
        expect = abs(lz_gs) / (2 * ion.info['n_holes'])
        check(f"{atom}{val:+d}: orbital sum rule", abs(ratio - expect) < 1e-10,
              f"ratio={ratio:.10f}  <Lz>={lz_gs:.6f}")


def test_symmetry_limits():
    """Cubic ions show no linear dichroism; isotropic XAS ignores orientation;
    a d0 ion scatters only elastically."""
    print("\nsymmetry limits")
    ion = xr.solve_ion(xr.IonModel(atom='Ni', valence=2, ten_dq=1.2), verbose=False)
    a, b = ion.xas(pol='LH'), ion.xas(pol='LV')
    check('Ni2+ Oh: no linear dichroism', rel(a.intensity, b.intensity) < 1e-10)
    rot = xr.solve_ion(xr.IonModel(atom='Ni', valence=2, ten_dq=1.2,
                                   cf='tetragonal', cf_d1=0.1, cf_d3=0.2,
                                   local_axis=edrixs.euler_to_rmat(0.4, 1.0, 0.3).tolist()),
                       verbose=False)
    ref = xr.solve_ion(xr.IonModel(atom='Ni', valence=2, ten_dq=1.2,
                                   cf='tetragonal', cf_d1=0.1, cf_d3=0.2), verbose=False)
    a, b = ref.xas(), rot.xas(omega=ref.xas().omega)
    check('isotropic XAS independent of local_axis', rel(b.intensity, a.intensity) < 1e-9)
    a, b = ref.xas(pol=[1, 0, 0]), ref.xas(pol=[0, 1, 0])
    check('D4h: no dichroism within the xy plane', rel(a.intensity, b.intensity) < 1e-10)
    b = ref.xas(pol=[0, 0, 1])
    check('D4h: E||x and E||z differ', rel(a.intensity, b.intensity) > 1e-2,
          f"{rel(a.intensity, b.intensity):.2f}")

    ti = xr.solve_ion(xr.IonModel(atom='Ti', valence=4, ten_dq=2.0), verbose=False)
    r = ti.rixs([0.0, 3.0], pol_in='LV')
    check('d0: only an elastic line', np.all(np.abs(r.pole_e[r.pole_w.max(axis=0) > 1e-12]) < 1e-9))


def test_energy_reference():
    """'peak' puts the strongest XAS feature at zero, 'edge' at the L3 energy."""
    print("\nenergy reference")
    ion = xr.solve_ion(xr.IonModel(atom='Ni', valence=2, gamma_c=0.3), verbose=False)
    s = ion.xas(gamma_c=0.3, omega=np.linspace(-3, 3, 6001))
    check("reference='peak'", abs(s.omega[np.argmax(s.intensity)]) < 2e-3)
    ion = xr.solve_ion(xr.IonModel(atom='Ni', valence=2, reference='edge',
                                   gamma_c=0.3), verbose=False)
    s = ion.xas(gamma_c=0.3, omega=np.linspace(850, 856, 6001))
    check("reference='edge'", abs(s.omega[np.argmax(s.intensity)] - 852.7) < 2e-3,
          f"{s.omega[np.argmax(s.intensity)]:.3f}")


# ---------------------------------------------------------------------------
# the parameter tables
# ---------------------------------------------------------------------------

def test_atom_tables_cover_absorption():
    """Every tabulated ion has its exact v^(n+1) core-hole configuration.

    The one known gap: Crispy has no v^(n+1) row at the highest occupancy for
    a d valence shell with a d core hole (the quadrupolar M45 / N45 edges of
    the 4d and 5d metals); those fall back to the neighbouring occupancy with
    a warning.
    """
    print("\nintermediate-state coverage of the atom tables")
    missing, total = [], 0
    for atom in atomic.available_atoms():
        table = atomic._atom_table(atom, 'crispy')
        shell = atomic.DEFAULT_SHELL[atom]
        norb = edrixs.info_atomic_shell()[shell[1]][1]
        occ = sorted({int(k[2:]) for k in table if k[0].isdigit() and '_' not in k})
        for edge in atomic.available_edges(atom, 'crispy'):
            core = atomic.core_shell(edge)
            for n in occ:
                if n >= norb:
                    continue
                total += 1
                key = f"{shell}{n + 1}_{core['shell']}{core['norb_full'] - 1}"
                if key not in table and not (shell[1] == 'd' and core['letter'] == 'd'):
                    missing.append(f"{atom} {shell}{n} {edge}")
    check(f"{total} (ion, edge) pairs", not missing, ', '.join(missing[:10]))


def test_parameters_vs_get_atom_data():
    """For Ni L23 the intermediate state matches edrixs.get_atom_data."""
    print("\nintermediate-state parameters vs. edrixs.get_atom_data")
    for nd in (7, 8, 9):
        ref = edrixs.get_atom_data('Ni', '3d', nd, edge='L3')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            par = xr.atomic_parameters('Ni', '3d', nd, 'L23', source='edrixs')
        # get_atom_data labels the integrals F2_11, G1_12, ... in the same order
        mine = [par['slater_n'][nm] for nm in par['names']]
        theirs = [v for _, v in ref['slater_n']]
        check(f"Ni 3d{nd} -> {par['config_n']}", np.allclose(mine[:7], theirs[:7]),
              f"{np.round(mine[:7], 3)}")
