"""
Many-body layer: Fock spaces held as int64 bitmasks and the second-quantised
operators built on them as scipy sparse matrices.

This is the sparse re-implementation of ``edrixs.get_fock_bin_by_N``,
``edrixs.build_opers`` and ``edrixs.two_fermion``.  The fermion sign convention is
edrixs's exactly -- annihilating orbital ``i`` carries
:math:`(-1)^{n_{<i}}` -- and ``four_fermion_sparse`` follows its
``(l, k, j, i)`` index order and its ``i == j or k == l`` skip.
``tests/test_xas_rixs.py::test_operators`` pins all of it against the edrixs
dense builders.

Orbital order is ``[impurity valence | bath valence | core]`` everywhere.
The single-ion model has no bath; the ordering is kept so that the Anderson
impurity model (in development) can reuse this layer as it is.
"""

from __future__ import annotations

__all__ = ['FockSpace', 'make_space', 'two_fermion_sparse',
           'four_fermion_sparse', 'annihilation_sparse']

from itertools import combinations
from math import comb
from typing import Optional

import numpy as np
import scipy.sparse as sp

_BYTE_BITS = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)


def _popcount(x: np.ndarray, nbits: int = 64) -> np.ndarray:
    """Number of set bits of an int64 array."""
    x = np.asarray(x, dtype=np.int64)
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(x).astype(np.int64)
    out = np.zeros(x.shape, dtype=np.int64)
    y = x
    for _ in range((nbits + 7) // 8):
        out += _BYTE_BITS[(y & 0xFF).astype(np.uint8)]
        y = y >> 8
    return out


def _sign(masked: np.ndarray) -> np.ndarray:
    """(-1)**(number of occupied orbitals below the one being acted on)."""
    return 1.0 - 2.0 * (_popcount(masked) & 1)


def _block_states(norb: int, nocc: int, offset: int = 0) -> np.ndarray:
    """All bitmasks with ``nocc`` of ``norb`` orbitals filled, shifted by ``offset``."""
    if nocc < 0 or nocc > norb:
        return np.zeros(0, dtype=np.int64)
    out = np.empty(_binom(norb, nocc), dtype=np.int64)
    for n, occ in enumerate(combinations(range(norb), nocc)):
        s = 0
        for o in occ:
            s |= 1 << (o + offset)
        out[n] = s
    return out


def _binom(n: int, k: int) -> int:
    return comb(n, k) if 0 <= k <= n else 0

def _outer_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, dtype=np.int64)
    return (a[:, None] | b[None, :]).ravel()


class FockSpace:
    """A sorted list of Fock states held as bitmasks (orbital ``i`` -> bit ``i``).

    The fermion sign convention is that of edrixs: annihilating orbital ``i``
    carries :math:`(-1)^{n_{<i}}` where :math:`n_{<i}` is the number of occupied
    orbitals with a *smaller* index.
    """

    def __init__(self, states: np.ndarray, norb: int, label: str = ''):
        self.states = np.sort(np.asarray(states, dtype=np.int64))
        self.norb = int(norb)
        self.label = label

    def __len__(self):
        return len(self.states)

    def index(self, states: np.ndarray):
        """Positions of ``states``; returns (position, found-mask)."""
        pos = np.searchsorted(self.states, states)
        pos_clip = np.minimum(pos, max(len(self.states) - 1, 0))
        ok = (pos < len(self.states)) & (self.states[pos_clip] == states)
        return pos_clip, ok

    def occupation(self, mask: int) -> np.ndarray:
        """Number of electrons inside the orbital subset ``mask``, per state."""
        return _popcount(self.states & mask)


def make_space(norb_v: int, nocc_v: int, norb_c: int, nocc_c: int,
               norb_imp: int = 0, n_imp: Optional[int] = None,
               max_extra_holes: Optional[int] = None,
               label: str = '') -> FockSpace:
    """Build a Fock space with fixed occupancy per shell.

    Orbital order is ``[impurity valence | bath valence | core]`` -- the same
    order used for the one-body and Coulomb matrices.

    Parameters
    ----------
    norb_v, nocc_v : int
        Size and electron count of the *whole* valence space (impurity + bath).
    norb_c, nocc_c : int
        Size and electron count of the core shell.
    norb_imp, n_imp : int
        Size and nominal occupancy of the impurity block.  Only used when
        ``max_extra_holes`` restricts the configuration space.
    max_extra_holes : int or None
        If given, keep only configurations with at most this many holes in the
        bath block, i.e. :math:`d^{n}L^{0} \\ldots d^{n+k}L^{k}`.  This is the
        standard truncation of charge-transfer multiplet calculations and is
        what makes mid-shell impurity models tractable.
    """
    if max_extra_holes is None or norb_v == norb_imp:
        val = _block_states(norb_v, nocc_v)
    else:
        norb_bath = norb_v - norb_imp
        kmax = min(max_extra_holes, norb_imp - n_imp, norb_bath)
        blocks = []
        for k in range(0, kmax + 1):
            imp = _block_states(norb_imp, n_imp + k)
            bath = _block_states(norb_bath, norb_bath - k, offset=norb_imp)
            blocks.append(_outer_or(imp, bath))
        val = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int64)
    core = _block_states(norb_c, nocc_c, offset=norb_v)
    return FockSpace(_outer_or(val, core), norb_v + norb_c, label)


# ----------------------------------------------------------------------------
# 2.  many-body operators as sparse matrices
# ----------------------------------------------------------------------------

def two_fermion_sparse(emat: np.ndarray, left: FockSpace, right: FockSpace,
                       tol: float = 1e-10) -> sp.csr_matrix:
    r"""``<L| sum_ij E_ij f^dag_i f_j |R>`` as a sparse matrix."""
    rows, cols, vals = [], [], []
    nz = np.argwhere(np.abs(emat) > tol)
    all_idx = np.arange(len(right), dtype=np.int64)
    for i, j in nz:
        i, j = int(i), int(j)
        st, col = right.states, all_idx
        m = ((st >> j) & 1) == 1
        st, col = st[m], col[m]
        sgn = _sign(st & ((1 << j) - 1))
        st = st ^ (1 << j)
        m = ((st >> i) & 1) == 0
        st, col, sgn = st[m], col[m], sgn[m]
        sgn = sgn * _sign(st & ((1 << i) - 1))
        st = st | (1 << i)
        pos, ok = left.index(st)
        if ok.any():
            rows.append(pos[ok])
            cols.append(col[ok])
            vals.append(emat[i, j] * sgn[ok])
    return _assemble(rows, cols, vals, (len(left), len(right)))


def four_fermion_sparse(umat: np.ndarray, left: FockSpace,
                        right: Optional[FockSpace] = None,
                        tol: float = 1e-10) -> sp.csr_matrix:
    r"""``<L| sum_ijkl U_ijkl f^dag_i f^dag_j f_k f_l |R>`` as a sparse matrix.

    The index convention matches :func:`edrixs.four_fermion`.
    """
    right = left if right is None else right
    rows, cols, vals = [], [], []
    nz = np.argwhere(np.abs(umat) > tol)
    all_idx = np.arange(len(right), dtype=np.int64)
    for lorb, korb, jorb, iorb in nz:
        lorb, korb, jorb, iorb = int(lorb), int(korb), int(jorb), int(iorb)
        if iorb == jorb or korb == lorb:
            continue
        st, col = right.states, all_idx
        sgn = np.ones(len(st))
        for orb, create in ((iorb, False), (jorb, False), (korb, True), (lorb, True)):
            want = 0 if create else 1
            m = ((st >> orb) & 1) == want
            st, col, sgn = st[m], col[m], sgn[m]
            if len(st) == 0:
                break
            sgn = sgn * _sign(st & ((1 << orb) - 1))
            st = st ^ (1 << orb)
        if len(st) == 0:
            continue
        pos, ok = left.index(st)
        if ok.any():
            rows.append(pos[ok])
            cols.append(col[ok])
            vals.append(umat[lorb, korb, jorb, iorb] * sgn[ok])
    return _assemble(rows, cols, vals, (len(left), len(right)))


def annihilation_sparse(iorb: int, left: FockSpace, right: FockSpace) -> sp.csr_matrix:
    """``<L| f_iorb |R>`` as a sparse matrix (one non-zero per column at most)."""
    st = right.states
    col = np.arange(len(right), dtype=np.int64)
    m = ((st >> iorb) & 1) == 1
    st, col = st[m], col[m]
    sgn = _sign(st & ((1 << iorb) - 1))
    st = st ^ (1 << iorb)
    pos, ok = left.index(st)
    return _assemble([pos[ok]], [col[ok]], [sgn[ok]], (len(left), len(right)))


def _assemble(rows, cols, vals, shape) -> sp.csr_matrix:
    if not rows:
        return sp.csr_matrix(shape, dtype=np.complex128)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    v = np.concatenate(vals).astype(np.complex128)
    out = sp.coo_matrix((v, (r, c)), shape=shape, dtype=np.complex128).tocsr()
    out.sum_duplicates()
    return out
