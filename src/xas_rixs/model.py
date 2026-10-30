"""
The single-ion model and its Hamiltonians.

:class:`IonModel` collects every parameter of a crystal-field multiplet
calculation; :func:`build_hamiltonians` turns it into one-body matrices,
Coulomb tensors, Fock spaces and transition operators for the initial
(:math:`v^n`) and intermediate (:math:`v^{n+1}\\underline{c}`) states.
"""

from __future__ import annotations

__all__ = ['IonModel', 'build_hamiltonians', 'replace_model']

import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

import edrixs

from .atomic import DEFAULT_SHELL, TM_GROUP, atomic_parameters, nd_from_valence
from .fock import make_space, two_fermion_sparse


@dataclass
class IonModel:
    """Everything that defines the ion: atom, edge, interactions, fields.

    Only ``atom`` and ``valence`` (or ``nd``) are really needed; every other
    field has a documented default.  The spectroscopy itself -- incident
    energies, polarisation, geometry, broadening -- is given to
    :func:`xas_rixs.xas` and :func:`xas_rixs.rixs`, so that one solved ion
    serves any number of spectra.

    The suffix ``_i`` marks the initial (ground) state and ``_n`` the
    intermediate (core-hole) state, as in edrixs.
    """

    # ---- the ion and the core level -------------------------------------
    atom: str = 'Ni'                      #: element symbol
    valence: Optional[int] = 2            #: formal oxidation state -> nd
    nd: Optional[int] = None              #: valence occupancy (overrides valence)
    shell: Optional[str] = None           #: '3d', '4f', ... (default from the atom)
    edge: str = 'L23'
    #: 'L23' (2p), 'M45' (3d), 'M23' (3p), 'N45' (4d), 'K' (1s, quadrupolar)...;
    #: a spin-orbit sub-shell ('L3', 'L2', 'M5', 'M4') keeps only that part of
    #: the core shell, which is smaller but neglects the mixing of the two edges
    param_source: str = 'auto'            #: 'auto'|'crispy'|'edrixs' Slater/SOC table

    # ---- Coulomb interaction --------------------------------------------
    scale_dd: float = 0.8                 #: scaling of the F^k_dd multipoles
    scale_dc: float = 0.8                 #: scaling of the F^k_dc / G^k_dc multipoles
    U_dd: float = 0.0
    #: monopole valence-valence U.  In the single-ion model it only shifts
    #: the absorption energy rigidly, so it is irrelevant on a relative axis
    U_dc: float = 0.0                     #: monopole core-hole potential (likewise)
    slater_override: dict = field(default_factory=dict)
    #: pin individual Slater integrals in *both* states, e.g. {'F2_dd': 11.0}
    #: (core-valence terms are always zero in the initial state)
    slater_override_n: dict = field(default_factory=dict)
    #: pin individual Slater integrals of the intermediate (core-hole) state

    # ---- spin-orbit coupling --------------------------------------------
    scale_soc_v: float = 1.0
    scale_soc_c: float = 1.0
    soc_v_i: Optional[float] = None       #: eV, overrides the tabulated value
    soc_v_n: Optional[float] = None
    soc_c: Optional[float] = None

    # ---- crystal field ---------------------------------------------------
    cf: str = 'auto'
    #: 'auto'|'none'|'cubic'|'tetragonal'|'square_planar'|'matrix';
    #: 'auto' is 'cubic' for a d shell and 'none' otherwise
    ten_dq: float = 1.0                   #: 10Dq (eV)
    ten_dq_n: Optional[float] = None      #: 10Dq with the core hole (default: same)
    cf_d1: float = 0.0                    #: tetragonal parameters (edrixs convention)
    cf_d3: float = 0.0
    cf_ds: float = 0.0                    #: square-planar parameter
    cf_matrix: Optional[Union[str, Sequence]] = None
    #: file or array in *real* harmonics, orbital-only or with spin

    # ---- external field, orientation and temperature --------------------
    ext_B: Sequence = (0.0, 0.0, 0.0)     #: exchange / Zeeman field in eV
    on_which: str = 'spin'                #: 'spin' | 'orbital' | 'both'
    local_axis: Optional[Sequence] = None
    #: 3x3 matrix whose columns are the crystal x, y, z axes in the frame of
    #: the scattering geometry (identity: the two frames coincide)
    temperature: float = 300.0            #: K, Boltzmann weighting of initial states
    p_tol: float = 1e-3                   #: drop initial states with smaller population
    n_gs: Optional[int] = None
    #: if set, thermally average over exactly this many lowest initial states

    # ---- numerics --------------------------------------------------------
    method: str = 'auto'                  #: 'auto' | 'exact' | 'lanczos'
    max_dense: int = 6000                 #: largest space diagonalised densely
    n_kryl: int = 400                     #: Krylov dimension of the continued fraction

    # ---- energy axis -----------------------------------------------------
    gamma_c: Optional[float] = None
    #: core-hole Lorentzian HWHM (eV) used to locate the reference peak, and
    #: the default width of every spectrum; ``None`` takes the tabulated value
    reference: str = 'peak'               #: 'peak'|'centroid'|'first_pole'|'edge'|'raw'
    shift: float = 0.0                    #: rigid shift added to the photon-energy axis

    # ---- Anderson impurity model (in development) -----------------------
    nbath: int = 0
    #: ligand shells; only 0 (the single-ion model) is implemented

    # ---- housekeeping ----------------------------------------------------
    label: Optional[str] = None

    # -------------------------------------------------------------------
    def resolved(self) -> dict:
        """Fill in the derived quantities (nd, shell, label, ...)."""
        atom = self.atom.strip()
        shell = self.shell or DEFAULT_SHELL.get(atom, '3d')
        if self.nd is not None:
            nd = int(self.nd)
        elif self.valence is not None:
            nd = nd_from_valence(atom, int(self.valence))
        else:
            raise ValueError("give either nd or valence")
        val = TM_GROUP[atom] - nd if atom in TM_GROUP else self.valence
        ion = f"{atom}{val:+d}" if val is not None else f"{atom}_{shell}{nd}"
        label = self.label or f"{ion}_{self.edge}"
        return dict(atom=atom, shell=shell, nd=nd, valence=val, ion=ion, label=label)

    def describe(self) -> str:
        r = self.resolved()
        return f"{r['ion']}  {r['shell']}^{r['nd']}  {self.edge} edge  [single ion]"


def replace_model(model: IonModel, **kw) -> IonModel:
    """Copy of ``model`` with some fields replaced (``label`` is reset)."""
    d = asdict(model)
    d.update(kw)
    if 'label' not in kw:
        d['label'] = None
    return IonModel(**d)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _l(letter: str) -> int:
    return edrixs.info_atomic_shell()[letter][0]


def _load_matrix(m) -> np.ndarray:
    if isinstance(m, str):
        return np.loadtxt(m, delimiter=',' if m.endswith('.csv') else None)
    return np.asarray(m)


def _expand_spin(mat: np.ndarray) -> np.ndarray:
    """Orbital-only matrix -> spin-orbital matrix with edrixs' interleaved order."""
    n = mat.shape[0]
    out = np.zeros((2 * n, 2 * n), dtype=complex)
    out[0::2, 0::2] = mat
    out[1::2, 1::2] = mat
    return out


def _slater(model: IonModel, par: dict) -> tuple:
    """Scaled Slater integrals of the initial and intermediate states.

    Order of operations: tabulated Hartree-Fock value -> scaling -> user
    override of the multipoles -> monopoles from ``F0 = U + <atomic limit>``
    -> user override of the monopoles.
    """
    names = par['names']
    v, c = par['v_letter'], par['core']['letter']
    s_i, s_n = dict(par['slater_i']), dict(par['slater_n'])
    for nm in names:
        if nm.startswith('F0'):
            continue
        scale = model.scale_dd if nm.endswith('_dd') else model.scale_dc
        s_i[nm] *= scale
        s_n[nm] *= scale
    # the core-core interaction is a constant for a full shell or a single hole
    for nm in names:
        if nm.endswith('_cc'):
            s_i[nm] = s_n[nm] = 0.0

    over_both = dict(model.slater_override or {})
    over_n = dict(model.slater_override_n or {})
    for nm in list(over_both) + list(over_n):
        if nm not in names:
            raise ValueError(f"unknown Slater integral {nm!r}; expected one of {names}")
    for nm, val in over_both.items():
        if not nm.startswith('F0'):
            s_n[nm] = float(val)
            if not nm.endswith('_dc'):
                s_i[nm] = float(val)
    for nm, val in over_n.items():
        if not nm.startswith('F0'):
            s_n[nm] = float(val)

    ks = range(2, 2 * _l(v) + 1, 2)
    s_i['F0_dd'] = model.U_dd + edrixs.get_F0(v, *[s_i[f'F{k}_dd'] for k in ks])
    s_n['F0_dd'] = model.U_dd + edrixs.get_F0(v, *[s_n[f'F{k}_dd'] for k in ks])
    gk = [nm for nm in names if nm.startswith('G') and nm.endswith('_dc')]
    s_n['F0_dc'] = model.U_dc + edrixs.get_F0(v + c, *[s_n[nm] for nm in gk])
    s_i['F0_dc'] = 0.0        # no core hole in the initial state

    for nm, val in over_both.items():
        if nm.startswith('F0'):
            s_n[nm] = float(val)
            if not nm.endswith('_dc'):
                s_i[nm] = float(val)
    for nm, val in over_n.items():
        if nm.startswith('F0'):
            s_n[nm] = float(val)
    return s_i, s_n


def _cf_real(model: IonModel, ten_dq: float, norb_v: int) -> np.ndarray:
    """Crystal field in the *real* harmonic basis (spin-interleaved)."""
    kind = (model.cf or 'none').strip().lower()
    if kind == 'auto':
        kind = 'cubic' if norb_v == 10 else 'none'
    if kind in ('none', 'off', ''):
        return np.zeros((norb_v, norb_v), dtype=complex)
    if kind == 'matrix':
        if model.cf_matrix is None:
            raise ValueError("cf='matrix' needs cf_matrix")
        m = _load_matrix(model.cf_matrix)
        if m.shape[0] == norb_v // 2:
            m = _expand_spin(m)
        if m.shape != (norb_v, norb_v):
            raise ValueError(f"cf_matrix should be {norb_v // 2}x{norb_v // 2} "
                             f"or {norb_v}x{norb_v}, got {m.shape}")
        return np.asarray(m, dtype=complex)
    if norb_v != 10:
        raise ValueError(f"cf={kind!r} is only implemented for a d shell; use "
                         f"cf='none' or cf='matrix' with a "
                         f"{norb_v // 2}x{norb_v // 2} cf_matrix in real harmonics")
    if kind == 'cubic':
        cf = edrixs.cf_cubic_d(ten_dq)
    elif kind == 'tetragonal':
        cf = edrixs.cf_tetragonal_d(ten_dq, model.cf_d1, model.cf_d3)
    elif kind == 'square_planar':
        cf = edrixs.cf_square_planar_d(ten_dq, model.cf_ds)
    else:
        raise ValueError(f"unknown crystal field type {kind!r}")
    # edrixs returns complex harmonics; the matrix input is real harmonics
    return edrixs.cb_op(cf, edrixs.tmat_c2r('d', True))


def _zeeman(model: IonModel, v_letter: str) -> np.ndarray:
    B = np.asarray(model.ext_B, dtype=float)
    ll = _l(v_letter)
    norb_v = 2 * (2 * ll + 1)
    if not np.any(np.abs(B) > 0):
        return np.zeros((norb_v, norb_v), dtype=complex)
    lx, ly, lz = (edrixs.get_lx(ll, True), edrixs.get_ly(ll, True),
                  edrixs.get_lz(ll, True))
    sx, sy, sz = edrixs.get_sx(ll), edrixs.get_sy(ll), edrixs.get_sz(ll)
    which = model.on_which.strip().lower()
    if which == 'spin':
        op = [2 * sx, 2 * sy, 2 * sz]
    elif which == 'orbital':
        op = [lx, ly, lz]
    elif which == 'both':
        op = [lx + 2 * sx, ly + 2 * sy, lz + 2 * sz]
    else:
        raise ValueError(f"on_which={model.on_which!r}")
    return B[0] * op[0] + B[1] * op[1] + B[2] * op[2]


def transition_operators(case: str, local_axis=None) -> np.ndarray:
    """Single-particle transition operators in the global (scattering) frame.

    Dipolar (``npol = 3``, Cartesian x, y, z) when the valence and core
    angular momenta differ by one, quadrupolar (``npol = 5``) when they differ
    by an even number.  The rotation from the crystal (``local_axis``) frame
    is the one :func:`edrixs.ed_1v1c_py` applies.
    """
    op = edrixs.get_trans_oper(case)
    npol = op.shape[0]
    if local_axis is None:
        return op
    axis = np.asarray(local_axis, dtype=float)
    out = np.zeros_like(op)
    if npol == 3:
        for i in range(3):
            for j in range(3):
                out[i] += axis[i, j] * op[j]
    else:
        alpha, beta, gamma = edrixs.rmat_to_euler(axis)
        wigner = edrixs.get_wigner_dmat(4, alpha, beta, gamma)
        rot = edrixs.tmat_r2c('d') @ wigner @ np.conj(edrixs.tmat_r2c('d')).T
        for i in range(5):
            for j in range(5):
                out[i] += rot[i, j] * op[j]
    return out


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------

def build_hamiltonians(model: IonModel, verbose: bool = False) -> dict:
    """One-body matrices, Coulomb tensors, Fock spaces and transition operators.

    Orbital order is ``[valence | core]``; the arrays are ``(ntot, ntot)``
    and ``(ntot,)*4``.  ``trans`` holds one sparse ``<n|T_k|i>`` per
    polarisation component in the Fock bases ``space_n`` x ``space_i``.
    """
    if int(model.nbath) != 0:
        raise NotImplementedError(
            "the Anderson impurity model (nbath > 0) is in development; only "
            "the single-ion model (nbath = 0) is available in this release")
    r = model.resolved()
    par = atomic_parameters(r['atom'], r['shell'], r['nd'], model.edge,
                            model.param_source)
    v_letter, core = par['v_letter'], par['core']
    norb_v = edrixs.info_atomic_shell()[v_letter][1]
    c_norb = core['norb']
    ntot = norb_v + c_norb
    case = v_letter + core['sub']

    # ---- Coulomb ---------------------------------------------------------
    s_i, s_n = _slater(model, par)
    names = par['names']
    umat_i = edrixs.get_umat_slater(case, *[s_i[nm] for nm in names])
    umat_n = edrixs.get_umat_slater(case, *[s_n[nm] for nm in names])

    # ---- one-body ----------------------------------------------------------
    r2c = edrixs.tmat_r2c(v_letter, True)
    ten_dq_n = model.ten_dq if model.ten_dq_n is None else model.ten_dq_n
    zeta_i = (par['soc_v_i'] if model.soc_v_i is None else model.soc_v_i) * model.scale_soc_v
    zeta_n = (par['soc_v_n'] if model.soc_v_n is None else model.soc_v_n) * model.scale_soc_v
    zeta_c = (par['soc_c'] if model.soc_c is None else model.soc_c) * model.scale_soc_c
    zee = _zeeman(model, v_letter)

    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)
    emat_i[:norb_v, :norb_v] = (edrixs.cb_op(_cf_real(model, model.ten_dq, norb_v), r2c)
                                + edrixs.atom_hsoc(v_letter, zeta_i) + zee)
    emat_n[:norb_v, :norb_v] = (edrixs.cb_op(_cf_real(model, ten_dq_n, norb_v), r2c)
                                + edrixs.atom_hsoc(v_letter, zeta_n) + zee)
    # a spin-orbit sub-shell (p32, d52, ...) is already a j eigenspace
    if core['sub'] in ('p', 'd', 'f') and abs(zeta_c) > 0:
        emat_n[norb_v:, norb_v:] += edrixs.atom_hsoc(core['letter'], zeta_c)

    # ---- Fock spaces and transition operators ------------------------------
    space_i = make_space(norb_v, r['nd'], c_norb, c_norb, label='initial')
    space_n = make_space(norb_v, r['nd'] + 1, c_norb, c_norb - 1, label='intermediate')
    op = transition_operators(case, model.local_axis)
    trans = []
    for k in range(op.shape[0]):
        t = np.zeros((ntot, ntot), dtype=complex)
        t[:norb_v, norb_v:] = op[k]
        trans.append(two_fermion_sparse(t, space_n, space_i))

    info = dict(par=par, slater_i=s_i, slater_n=s_n, norb_v=norb_v,
                c_norb=c_norb, ntot=ntot, case=case, npol=op.shape[0],
                transition='dipolar' if op.shape[0] == 3 else 'quadrupolar',
                zeta_v_i=zeta_i, zeta_v_n=zeta_n, zeta_c=zeta_c,
                v_mask=(1 << norb_v) - 1, dim_i=len(space_i), dim_n=len(space_n),
                **r)
    if par['config_n'] is None:
        warnings.warn("intermediate-state parameters are missing; see above")
    if verbose:
        print(f"  {par['config_i']} -> {par['config_n']}  ({info['transition']}, "
              f"Slater/SOC from {par['source']})")
        print(f"  dim(initial) = {len(space_i)}, dim(intermediate) = {len(space_n)}")
    return dict(emat_i=emat_i, emat_n=emat_n, umat_i=umat_i, umat_n=umat_n,
                space_i=space_i, space_n=space_n, trans=trans, info=info)
