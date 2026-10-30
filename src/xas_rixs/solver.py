"""
Exact diagonalisation of the single-ion model.

:func:`solve_ion` diagonalises the initial Hamiltonian completely (its
dimension is at most a few thousand for any single d or f shell) and treats
the intermediate (core-hole) Hamiltonian in one of two ways:

``'exact'``
    full diagonalisation; the transition operators are stored in the two
    eigenbases.
``'lanczos'``
    for intermediate spaces above ``max_dense``.  For every initial state and
    incident polarisation a Lanczos run on :math:`H_n` from
    :math:`T_\\epsilon|i\\rangle` gives Ritz energies :math:`\\theta_j` and
    vectors.  XAS is then :math:`\\sum_j |c_j|^2 \\delta(\\omega-\\theta_j+E_i)`
    -- identical to the continued fraction -- and the Kramers-Heisenberg
    amplitude uses the same Ritz states as its intermediate states.

Both paths hand the spectroscopy code the same thing: a list of intermediate
poles ``(theta, c)`` and, for RIXS, the emission matrix
:math:`\\langle f|T^\\dagger_{\\epsilon'}|\\theta_j\\rangle`.
"""

from __future__ import annotations

__all__ = ['IonSolution', 'solve_ion']

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

import edrixs

from .fock import four_fermion_sparse, two_fermion_sparse
from .lineshape import broaden
from .model import IonModel, build_hamiltonians
from .polarization import polarization_terms


def _lanczos_ritz(H: sp.csr_matrix, v0: np.ndarray, nkryl: int,
                  keep_basis: bool, tol: float = 1e-10):
    """Ritz energies, first components and (optionally) Ritz vectors of ``H``.

    Lanczos from ``v0``.  With ``keep_basis`` the Krylov basis is fully
    reorthogonalised and stored, so that the Ritz *vectors* are meaningful;
    without it only three vectors are held and only the Ritz energies and
    their weights in ``v0`` -- which is all XAS needs -- are returned.
    """
    n = v0.size
    m = int(min(nkryl, n))
    nrm = np.linalg.norm(v0)
    Q = np.zeros((m, n), dtype=complex) if keep_basis else None
    alpha, beta = np.zeros(m), np.zeros(m)
    q, q_prev = v0 / nrm, np.zeros(n, dtype=complex)
    used = m
    for k in range(m):
        if keep_basis:
            Q[k] = q
        w = H @ q
        alpha[k] = np.real(np.vdot(q, w))
        w = w - alpha[k] * q - (beta[k] * q_prev if k else 0.0)
        if keep_basis:
            w -= (Q[:k + 1].conj() @ w) @ Q[:k + 1]
            w -= (Q[:k + 1].conj() @ w) @ Q[:k + 1]
        b = np.linalg.norm(w)
        if k + 1 == m:
            break
        if b < tol:
            used = k + 1
            break
        beta[k + 1] = b
        q_prev, q = q, w / b
    alpha, beta = alpha[:used], beta[:used]
    theta, S = la.eigh_tridiagonal(alpha, beta[1:])
    c = nrm * S[0]
    R = Q[:used].T @ S if keep_basis else None
    return theta, c, R


@dataclass
class IonSolution:
    """A solved ion: eigenstates and transition operators.

    Hand it to :func:`xas_rixs.xas` and :func:`xas_rixs.rixs` (or use the
    :meth:`xas` / :meth:`rixs` shortcuts) for spectra in any polarisation and
    geometry without re-diagonalising.

    Energies: ``E_i`` are absolute eigenvalues of the initial Hamiltonian.
    Photon energies on the user-facing axis relate to raw excitation energies
    :math:`E_n - E_i` through :meth:`to_raw` / :meth:`from_raw`, as fixed by
    ``model.reference`` and ``model.shift``.
    """
    model: IonModel
    info: dict
    E_i: np.ndarray                 #: all initial eigenvalues (eV)
    V_i: np.ndarray                 #: initial eigenvectors, Fock basis
    gs: np.ndarray                  #: indices of the thermally populated states
    p: np.ndarray                   #: their Boltzmann populations
    method: str                     #: 'exact' or 'lanczos'
    E_n: Optional[np.ndarray] = None     #: intermediate eigenvalues (exact)
    T: Optional[np.ndarray] = None
    #: (npol, dim_n, dim_i) transition operators in the eigenbases (exact)
    H_n: Optional[sp.csr_matrix] = field(default=None, repr=False)
    trans: Optional[list] = field(default=None, repr=False)
    offset: float = 0.0             #: raw energy placed at zero of the axis
    axis_origin: float = 0.0        #: tabulated edge energy if reference='edge'
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- energy axis -----------------------------------------------------
    def to_raw(self, omega):
        """Photon energy on the result axis -> raw :math:`E_n - E_i`."""
        return np.asarray(omega, dtype=float) - self.model.shift - self.axis_origin \
            + self.offset

    def from_raw(self, e):
        return np.asarray(e, dtype=float) + self.model.shift + self.axis_origin \
            - self.offset

    @property
    def gamma_c(self) -> float:
        """Default core-hole Lorentzian HWHM (eV)."""
        g = self.model.gamma_c
        return float(min(self.info['par']['gamma_c'])) if g is None else float(g)

    @property
    def npol(self) -> int:
        return self.info['npol']

    # ---- intermediate-state poles ----------------------------------------
    def absorption_poles(self, g: int, vec: np.ndarray, keep_basis: bool = False):
        """``(theta, c, R)`` for absorption from initial state ``g``.

        ``theta`` are intermediate energies and ``c`` the amplitudes
        :math:`\\langle n|\\epsilon\\cdot T|g\\rangle`; ``R`` holds the Ritz
        vectors in the Fock basis (``None`` for the exact eigenbasis, or when
        ``keep_basis`` is false).
        """
        key = ('abs', int(g), np.round(vec, 12).tobytes(), keep_basis)
        if key in self._cache:
            return self._cache[key]
        if self.method == 'exact':
            c = np.tensordot(vec, self.T[:, :, g], axes=(0, 0))
            out = (self.E_n, c, None)
        else:
            b = sum(vec[k] * (self.trans[k] @ self.V_i[:, g]) for k in range(self.npol))
            if np.linalg.norm(b) < 1e-14:
                out = (np.zeros(0), np.zeros(0, dtype=complex),
                       np.zeros((self.H_n.shape[0], 0), dtype=complex) if keep_basis else None)
            else:
                out = _lanczos_ritz(self.H_n, b, self.model.n_kryl, keep_basis)
        self._cache[key] = out
        return out

    def emission_matrix(self, vec: np.ndarray, R: Optional[np.ndarray]) -> np.ndarray:
        """:math:`\\langle f|(\\epsilon'\\cdot T)^\\dagger|n\\rangle`, shape (dim_i, m).

        Rows are initial-Hamiltonian eigenstates ``f``; columns the
        intermediate states of :meth:`absorption_poles` (``R`` from it).
        """
        if self.method == 'exact':
            key = ('emi', np.round(vec, 12).tobytes())
            if key not in self._cache:
                self._cache[key] = np.tensordot(vec, self.T, axes=(0, 0)).conj().T
            return self._cache[key]
        A = sum(vec[k] * self.trans[k] for k in range(self.npol))
        return self.V_i.conj().T @ (A.conj().T @ R)

    # ---- shortcuts ---------------------------------------------------------
    def xas(self, omega=None, **kw):
        """Shortcut for :func:`xas_rixs.xas`."""
        from .spectra import xas
        return xas(self, omega, **kw)

    def rixs(self, ominc, eloss=None, **kw):
        """Shortcut for :func:`xas_rixs.rixs`."""
        from .spectra import rixs
        return rixs(self, ominc, eloss, **kw)

    def describe(self) -> str:
        return self.model.describe()


def solve_ion(model: IonModel, verbose: bool = True) -> IonSolution:
    """Diagonalise the initial and intermediate Hamiltonians of ``model``."""
    t0 = time.time()
    if verbose:
        print(f"[xas_rixs] {model.describe()}")
    H = build_hamiltonians(model, verbose=verbose)
    info = H['info']
    space_i, space_n = H['space_i'], H['space_n']
    if len(space_i) > model.max_dense:
        raise ValueError(f"initial space of {len(space_i)} states exceeds "
                         f"max_dense={model.max_dense}; raise max_dense")

    # ---- initial state: always dense ------------------------------------
    Hi = (four_fermion_sparse(H['umat_i'], space_i)
          + two_fermion_sparse(H['emat_i'], space_i, space_i))
    E_i, V_i = la.eigh(Hi.toarray())
    p_all = edrixs.boltz_dist(E_i, model.temperature)
    if model.n_gs is not None:
        gs = np.arange(min(int(model.n_gs), len(E_i)))
    else:
        gs = np.flatnonzero(p_all > model.p_tol)
        if len(gs) == 0:
            gs = np.array([0])
    p = p_all[gs] / p_all[gs].sum()
    nv = space_i.occupation(info['v_mask'])
    info['E0'] = float(E_i[0])
    info['gs_degeneracy'] = int(np.sum(np.abs(E_i - E_i[0]) < 1e-6))
    info['n_initial_states'] = int(len(gs))
    info['populations'] = p.tolist()
    info['n_holes'] = int(info['norb_v'] - nv[0])

    # ---- intermediate state ----------------------------------------------
    method = model.method.strip().lower()
    if method == 'auto':
        method = 'exact' if len(space_n) <= model.max_dense else 'lanczos'
    if method not in ('exact', 'lanczos'):
        raise ValueError(f"method={model.method!r}")
    Hn = (four_fermion_sparse(H['umat_n'], space_n)
          + two_fermion_sparse(H['emat_n'], space_n, space_n)).tocsr()
    info['method'] = method
    info['nnz_Hn'] = int(Hn.nnz)
    if verbose:
        print(f"  H_n: {Hn.shape[0]} x {Hn.shape[0]}, {Hn.nnz} non-zeros, "
              f"method={method}")
    sol = IonSolution(model=model, info=info, E_i=E_i, V_i=V_i, gs=gs, p=p,
                      method=method)
    if method == 'exact':
        E_n, V_n = la.eigh(Hn.toarray())
        sol.E_n = E_n
        sol.T = np.array([V_n.conj().T @ (t @ V_i) for t in H['trans']])
    else:
        sol.H_n, sol.trans = Hn, H['trans']

    # ---- isotropic XAS: sum rule and the energy reference ------------------
    gamma = sol.gamma_c
    e_all, w_all = [], []
    for g, pg in zip(gs, p):
        for vec, wt in polarization_terms('isotropic', 0.0, npol=sol.npol):
            theta, c, _ = sol.absorption_poles(g, vec)
            e_all.append(theta - E_i[g])
            w_all.append(pg * wt * np.abs(c) ** 2)
    e_all, w_all = np.concatenate(e_all), np.concatenate(w_all)
    keep = w_all > 1e-10 * w_all.max()
    e_all, w_all = e_all[keep], w_all[keep]
    info['xas_sum_iso'] = float(w_all.sum())
    centroid = float(np.sum(e_all * w_all) / w_all.sum())
    info['centroid_raw'] = centroid

    ref = model.reference.strip().lower()
    if ref == 'raw':
        offset = 0.0
    elif ref == 'centroid':
        offset = centroid
    elif ref == 'first_pole':
        offset = float(e_all[w_all > 1e-4 * w_all.max()].min())
    elif ref in ('peak', 'edge'):
        scan = np.linspace(e_all.min() - 5 * gamma, e_all.max() + 5 * gamma,
                           max(4000, int((e_all.max() - e_all.min()) / (0.02 * gamma))))
        offset = float(scan[np.argmax(broaden(e_all, w_all, scan, gamma))])
    else:
        raise ValueError(f"reference={model.reference!r}")
    sol.offset = offset
    sol.axis_origin = float(min(info['par']['edge_ene'])) if ref == 'edge' else 0.0
    info['reference'] = ref
    info['offset'] = offset
    info['axis_origin'] = sol.axis_origin
    info['t_solve'] = time.time() - t0
    if verbose:
        print(f"  isotropic XAS weight = {info['xas_sum_iso']:.6f} "
              f"({info['n_holes']} holes), {len(gs)} initial state(s), "
              f"done in {info['t_solve']:.1f} s")
    return sol
