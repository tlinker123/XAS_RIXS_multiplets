# Single-ion spectral data generation with `xas_rixs`

How to use this package for the single-ion (crystal-field multiplet) part of a
multiplet spectral-data campaign: 3d transition-metal ions Ti–Cu, L and K
edges, many sampled Hamiltonian parameter sets θ, with every spectrum labelled
by the θ that produced it.

## 1. Coverage

| Observable | Status |
|---|---|
| L-XAS, XMCD, XLD | ✅ available, validated against EDRIXS |
| L-RIXS | ✅ available, validated against EDRIXS |
| K pre-edge XAS (1s→3d, quadrupole) | ✅ `edge='K'`, validated against EDRIXS |
| K-RIXS (1s→3d→1s) | ✅ `edge='K'`, validated against EDRIXS |
| L-XPS, K-XPS | 🚧 in development |
| L-XES (Lα,β), Kα and Kβ XES | ❌ not yet implemented |
| K main edge (1s→4p) | ❌ needs a second valence shell (3d + 4p) |
| 1s2p RIXS | ❌ not planned for the first release |

**Atomic parameters.** The bundled tables (`src/xas_rixs/atom_data/`) hold
Cowan Hartree–Fock Slater integrals and spin–orbit couplings for the ground
state and for the 1s, 2p and 3p core-hole configurations of every Ti–Cu ion
listed below. The only missing entry is Cu¹⁺ 3d¹⁰, which has no L- or K-edge
absorption anyway. The 3d + 4p configurations for the K main edge are not
tabulated yet.

| Element | Oxidation states | 3dⁿ |
|---|---|---|
| Ti | 2+ 3+ 4+ | d² d¹ d⁰ |
| V | 2+ 3+ 4+ 5+ | d³ d² d¹ d⁰ |
| Cr | 2+ 3+ 4+ | d⁴ d³ d² |
| Mn | 2+ 3+ 4+ | d⁵ d⁴ d³ |
| Fe | 2+ 3+ 4+ | d⁶ d⁵ d⁴ |
| Co | 2+ 3+ 4+ | d⁷ d⁶ d⁵ |
| Ni | 2+ 3+ | d⁸ d⁷ |
| Cu | 1+ 2+ 3+ | d¹⁰ d⁹ d⁸ |

**Crystal-field symmetries.**

| Symmetry | How to set it | Status |
|---|---|---|
| O_h | `cf='cubic'`, `ten_dq` | ✅ |
| T_d | `cf='cubic'` with a negative `ten_dq` | ✅ |
| D₄h tetragonal | `cf='tetragonal'`, `ten_dq`, `cf_d1`, `cf_d3` (EDRIXS parametrisation) | ✅, Ds/Dt input planned |
| D₄h square planar | `cf='square_planar'`, `ten_dq`, `cf_ds` | ✅ |
| D₃d trigonal (Dσ, Dτ) | `cf='matrix'` with a hand-built `cf_matrix` | built-in form planned |
| anything else | `cf='matrix'`, 5×5 in real harmonics (dz², dzx, dzy, dx²−y², dxy) | ✅ |

## 2. Throughput

Measured on one laptop core for the largest single-ion L-edge cases:

| Ion | Core-hole states | Full solve | Isotropic XAS | 41-point RIXS map |
|---|---|---|---|---|
| Mn³⁺ | 1 512 | 1.9 s | 0.2 s | 0.3 s |
| Fe³⁺ | 1 260 | 1.3 s | 0.1 s | 0.3 s |
| Ni²⁺ | 60 | 0.06 s | 3 ms | 0.02 s |

The dense diagonalisation of the core-hole Hamiltonian dominates. Building
the Hamiltonian takes about 0.04 s, so precomputing operators does not pay.
10⁵ parameter sets per (ion, symmetry) is roughly 1–2 CPU-days for the worst
ions. Parallelise over θ with one process per core and one BLAS thread per
process (`OMP_NUM_THREADS=1`).

## 3. Recommended workflow

**One θ → one solve per core level → every spectrum of that edge.** Each edge
needs its own `solve_ion`, because the core-hole block differs. The ground
state (at most 252 states for 3d) is recomputed per edge, which is cheap.
Polarisations, geometries and RIXS incident energies all reuse the same
solved ion.

```python
import numpy as np
from xas_rixs import IonModel, solve_ion, xas, rixs

OMINC_L = np.linspace(-5, 25, 151)     # fixed grids: every row of the dataset
OMINC_K = np.linspace(-5, 10, 76)      # must line up on the same axes
ELOSS = np.linspace(-0.5, 12, 600)

def run_theta(atom, valence, sym, theta):
    common = dict(
        atom=atom, valence=valence,
        scale_dd=theta['k_dd'], scale_dc=theta['k_pd'], scale_soc_v=theta['s_z3d'],
        cf=sym, ten_dq=theta['10Dq'], cf_d1=theta.get('d1', 0.0),
        cf_d3=theta.get('d3', 0.0),
        ext_B=(0.0, 0.0, theta['H_ex']), temperature=theta['T'],
        reference='raw',                        # see section 4.2
    )
    L = solve_ion(IonModel(edge='L23', gamma_c=theta['G_L3'], **common), verbose=False)
    K = solve_ion(IonModel(edge='K', gamma_c=theta['G_1s'], **common), verbose=False)
    return {
        'L_XAS_iso':  xas(L).poles,                                  # (energy, weight) sticks
        'L_XAS_pol':  {p: xas(L, pol=p).poles for p in ('LH', 'LV', 'left', 'right')},
        'L_RIXS':     rixs(L, OMINC_L, ELOSS, pol_in='LV', gamma_f=theta['G_f']),
        'K_XAS_iso':  xas(K).poles,
        'K_RIXS':     rixs(K, OMINC_K, ELOSS, pol_in='LV', gamma_f=theta['G_f']),
        'centroid_L': L.info['centroid_raw'],
        'centroid_K': K.info['centroid_raw'],
    }
```

What to store per θ:

* **XAS**: the stick spectra (`.poles`). Any core-hole width and resolution
  can be applied afterwards with `xas_rixs.broaden`.
* **RIXS**: `pole_e` (energy loss of every final state) and `pole_w` (its
  weight at each incident energy). The final-state width and the loss
  resolution can be applied afterwards; the core-hole width cannot (see 4.1).
* **Provenance**: the package version, the EDRIXS commit, the full θ and a
  hash of it, and the energy reference used.

## 4. Physics points to settle before generating data

### 4.1 The RIXS core-hole width must be fixed at generation time

Stick spectra allow any broadening to be applied afterwards for XAS. This is
not true of the core-hole width Γ_c in RIXS. Γ_c sits inside the
Kramers–Heisenberg sum and controls the interference between intermediate
states, so changing it changes the relative weights of the final states, not
just their line shape. Sample Γ_L3, Γ_L2 and Γ_1s as part of θ when the RIXS
is generated.

An energy-dependent `gamma_c` (given as `[[energy, gamma], ...]` breakpoints,
e.g. a broader L₂) is applied per incident energy. That is a good
approximation for the L₂ Coster–Kronig broadening, but it is an
approximation and should be recorded as such.

### 4.2 Choose the energy reference once

A single-ion model predicts relative energies only; it has no chemical shift.

* The default `reference='peak'` puts the strongest XAS feature at zero for
  every ion, which also removes the relative position between ions.
* For a dataset, use `reference='raw'` and store `info['centroid_raw']`, then
  apply one documented alignment convention (e.g. the tabulated L₃ or K edge
  energy of the element plus a sampled shift).
* Incoherent mixtures of ions (mixed valence such as Fe²⁺/Fe³⁺, or
  multi-site spinels) need their relative energy offset as an explicit
  sampled parameter; otherwise the mixtures are meaningless.

### 4.3 The T_d K pre-edge is underestimated

The K pre-edge here is a pure 1s→3d quadrupole transition. Without an
inversion centre (T_d), 3d–4p mixing adds dipole intensity that dominates the
measured pre-edge. Single-ion T_d K pre-edges should be labelled qualitative
for intensities; peak positions and multiplet structure remain useful.

### 4.4 Map θ to the package's fields explicitly

| Proposal symbol | Field | Note |
|---|---|---|
| κ_dd | `scale_dd` | scales F², F⁴ (dd) in both states |
| κ_pd, κ_sd | `scale_dc` | scales the core–valence F^k, G^k |
| ζ3d scaling | `scale_soc_v` | scales ground and core-hole ζ3d together; use `soc_v_i` / `soc_v_n` to free them |
| ζ2p scaling | `scale_soc_c` | |
| 10Dq | `ten_dq` | `ten_dq_n` for a different value in the core-hole state |
| Ds, Dt | `cf_d1`, `cf_d3` | **different parametrisation**: needs a tested converter |
| Dσ, Dτ | `cf_matrix` | until the built-in D₃d form exists |
| H_ex | `ext_B`, `on_which='spin'` | couples as B·2S; fix what "H_ex = 0.1 eV" means against this |
| T | `temperature` | Boltzmann average over states above `p_tol` |
| Γ_L3/Γ_L2, Γ_1s | `gamma_c` | scalar or breakpoints; default is the tabulated value |
| Γ_f, σ_instr | `gamma_f`, `gauss_fwhm` | arguments of `rixs` / `xas` |
| polarisation, geometry | `pol`, `pol_in`, `pol_out`, `thin`, `thout`, `phi` | EDRIXS conventions; see `polarization.py` |

## 5. Planned additions, in priority order

1. **D₃d crystal field (Dσ, Dτ) and Ds/Dt input for D₄h**, each tested
   against a hand-built matrix.
2. **`ground_state_labels(ion)`** returning ⟨S²⟩, ⟨L·S⟩, n_d, e_g/t₂g
   occupancies, a high/low-spin flag and the ground-state degeneracy. The
   initial-state eigenvectors are already stored on `IonSolution`.
3. **XPS** (L and K core levels), in development, under the same
   `IonModel` so one θ drives every observable.
4. **XES drivers**: Lα (from the thermalised 2p-hole states to 3dⁿ⁻¹), and
   Kα/Kβ (from 1s¹3dⁿ to 2p⁵3dⁿ / 3p⁵3dⁿ). These need a two-core-shell
   orbital layout, which the many-body layer already supports.
5. **A batch driver** writing Zarr/HDF5 with θ, sticks, maps and labels on a
   shared leading axis, plus the provenance listed in section 3.
6. **Validation against reference oxides** in the single-ion limit: SrTiO₃
   (Ti⁴⁺), Cr₂O₃ (Cr³⁺), α-Fe₂O₃ (Fe³⁺), CoO (Co²⁺) and NiO (Ni²⁺) L-edge
   XAS and RIXS.

The Anderson impurity model (charge-transfer satellites, metal–ligand
covalency) is in development; see the README.
