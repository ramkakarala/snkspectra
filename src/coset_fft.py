"""
Vibe coded (Claude)
Fast Fourier Transform of right-S_{n-k}-invariant functions on S_n.

Setup
-----
S_{n-k} ≤ S_n is the subgroup that fixes the last k positions pointwise:

    S_{n-k} = { τ ∈ S_n : τ(i) = i  for  i = n-k+1, …, n }

A function f : S_n → ℂ is *right-S_{n-k}-invariant* if

    f(σ·τ) = f(σ)   for all τ ∈ S_{n-k}.

Such functions are constant on right cosets of S_{n-k} and are identified
with functions on the coset space

    S_n / S_{n-k}  ≅  { ordered k-tuples of distinct elements of {1,…,n} },

labelled by  (σ(n-k+1), …, σ(n)).  The coset space has  n!/(n-k)!  elements.

Fourier transform
-----------------
The Fourier transform is still

    f̂(ρ_λ) = Σ_{σ ∈ S_n}  f(σ) ρ_λ(σ)          (d_λ × d_λ  matrix)

but the S_{n-k}-invariance means the sum collapses:

    f̂(ρ_λ) = (n-k)! · Σ_{coset reps c}  f̃(c) · ρ_λ(c_rep)

where the effective computation uses the partial Clausen recursion below.

Algorithm (k steps of Clausen's coset decomposition)
------------------------------------------------------
At each level (n, k) with k > 0:

  1. Partition the n!/(n-k)! k-tuples by their last element j = σ(n):
         g_j(b₁,…,b_{k-1}) = f̃(τ_j(b₁),…,τ_j(b_{k-1}), j)
     where τ_j = transposition (j n)  and  b_i ∈ {1,…,n-1}.

  2. Recursively compute  ĝ_j = FFT on S_{n-1} / S_{n-k}  for each j.

  3. Assemble using the Clausen formula:
         f̂(ρ_λ) = Σ_{j=1}^{n}  ρ_λ(τ_j) · BlockDiag_λ( ĝ_j(μ) : μ ≺ λ )

Base case k = 0 (constant function, no remaining summation variable):
  f is constant on S_n (= fully S_n-invariant).
  Only the trivial irrep (n,) gets a nonzero coefficient:
      f̂((n,)) = f̃(()) · n! · I₁

Special cases:
  k = 0 : only trivial Fourier component survives.
  k = n : no invariance; recovers the full Clausen FFT (sn_fft).

Complexity
----------
The recursion makes n·(n-1)·…·(n-k+1) = n!/(n-k)! leaf calls, each O(1),
plus O(k · n!/(n-k)! · Σ_λ d_λ²) ≈ O(k · n! · n!/(n-k)!) for assembly.
The sum over cosets is therefore "linear" in the domain size n!/(n-k)!,
compared with O((n!)²) for the naive DFT.

References
----------
  Clausen, "Fast generalized Fourier transforms" (1989).
  Maslen & Rockmore, "Generalized FFTs — a survey of applications" (1997).
"""

import math
import numpy as np

from permutations import Permutation
from symmetric_group import SymmetricGroup
from irreps import irrep, partitions_of, standard_young_tableaux
from fft import _tau, _block_structure


# ---------------------------------------------------------------------------
# Helpers: convert between full S_n functions and coset functions
# ---------------------------------------------------------------------------

def to_coset_function(f: dict, n: int, k: int) -> dict:
    """
    Convert a full function f : S_n → ℂ to its coset representation.

    Parameters
    ----------
    f : dict {Permutation → value}
    n : int
    k : int  (0 ≤ k ≤ n)

    Returns
    -------
    f_tilde : dict { (a_1,…,a_k) → value }
        where (a_1,…,a_k) = (σ(n-k+1),…,σ(n)).

    Raises ValueError if f is not actually right-S_{n-k}-invariant.
    """
    f_tilde = {}
    Sn = SymmetricGroup(n)
    for sigma in Sn:
        tup = tuple(sigma(i) for i in range(n - k + 1, n + 1))
        val = f.get(sigma, 0.0)
        if tup in f_tilde:
            if not np.isclose(f_tilde[tup], val):
                raise ValueError(
                    f"f is not right-S_{{n-k}}-invariant: "
                    f"key {tup} maps to both {f_tilde[tup]!r} and {val!r}"
                )
        else:
            f_tilde[tup] = val
    return f_tilde


def from_coset_function(f_tilde: dict, n: int, k: int) -> dict:
    """
    Expand a coset function f̃ to the full right-S_{n-k}-invariant function
    on S_n by setting  f(σ) = f̃(σ(n-k+1),…,σ(n)).

    Parameters
    ----------
    f_tilde : dict { (a_1,…,a_k) → value }
    n : int
    k : int

    Returns
    -------
    f : dict {Permutation → value}
    """
    Sn = SymmetricGroup(n)
    f = {}
    for sigma in Sn:
        tup = tuple(sigma(i) for i in range(n - k + 1, n + 1))
        f[sigma] = f_tilde.get(tup, 0.0)
    return f


# ---------------------------------------------------------------------------
# Core: S_n / S_{n-k} FFT
# ---------------------------------------------------------------------------

def sn_mod_sk_fft(f_tilde: dict, n: int, k: int) -> dict[tuple, np.ndarray]:
    """
    Fourier transform of a right-S_{n-k}-invariant function on S_n.

    Parameters
    ----------
    f_tilde : dict { (a_1,…,a_k) → complex }
        The coset function.  (a_1,…,a_k) = (σ(n-k+1),…,σ(n)) are k distinct
        elements from {1,…,n}.  Missing tuples are treated as 0.
    n : int
        Degree of S_n.
    k : int
        Coset dimension; 0 ≤ k ≤ n.

    Returns
    -------
    f_hat : dict { partition λ ⊢ n → np.ndarray (d_λ × d_λ) }
        Same format as fourier.sn_dft and fft.sn_fft.
        The inverse sn_idft from fourier.py works unchanged.
    """
    if k < 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 ≤ k ≤ n={n}")

    # --- Base case: k = 0, f is constant on S_n ----------------------------
    # f̂(λ) = val · Σ_{σ ∈ S_n} ρ_λ(σ)  =  val · n! · [λ = trivial] · I₁
    # The trivial partition of n is (n,) for n ≥ 1, and () for n = 0
    # (since (0,) is not a valid partition — parts must be positive).
    if k == 0:
        val = f_tilde.get((), 0.0)
        factorial_n = math.factorial(n)
        trivial_lam = () if n == 0 else (n,)
        result = {}
        for lam in partitions_of(n):
            d = len(standard_young_tableaux(lam))
            if lam == trivial_lam:
                result[lam] = np.array([[val * factorial_n]], dtype=complex)
            else:
                result[lam] = np.zeros((d, d), dtype=complex)
        return result

    # --- Step 1: split by the last tuple entry j = σ(n) -------------------
    # For each j ∈ {1,…,n}, build the (k-1)-tuple function on S_{n-1}:
    #
    #   g_j(b₁,…,b_{k-1}) = f̃(τ_j(b₁),…,τ_j(b_{k-1}), j)
    #
    # where τ_j = (j n) maps b ∈ {1,…,n-1} as:
    #   b ↦ j  if  b = j  (impossible: b ∈ {1,…,n-1} and a_{i<k} ≠ j)
    #   b ↦ n  if  b = j  (τ_j sends j → n)   [handled below]
    # More precisely: a_i = τ_j(b_i) so b_i = τ_j⁻¹(a_i) = τ_j(a_i) and
    #   b_i = j  if  a_i = n
    #   b_i = a_i  otherwise  (since a_i ≠ j by distinctness)

    g_j_tildes: dict[int, dict] = {j: {} for j in range(1, n + 1)}

    for tup, val in f_tilde.items():
        j = tup[-1]                              # a_k = σ(n)
        b_tup = tuple(j if a == n else a         # τ_j⁻¹ applied to each a_i
                      for a in tup[:-1])
        g_j_tildes[j][b_tup] = val

    # --- Step 2: recursively FFT each g_j on S_{n-1} -----------------------
    g_j_hats: dict[int, dict] = {
        j: sn_mod_sk_fft(g_j_tildes[j], n - 1, k - 1)
        for j in range(1, n + 1)
    }

    # --- Step 3: assemble using Clausen's formula --------------------------
    # f̂(ρ_λ) = Σ_j  ρ_λ(τ_j) · BlockDiag_λ( ĝ_j(μ) : μ ≺ λ )
    f_hat: dict[tuple, np.ndarray] = {}

    for lam in partitions_of(n):
        rho = irrep(lam)
        d = len(standard_young_tableaux(lam))
        block_struct = _block_structure(lam)

        F = np.zeros((d, d), dtype=complex)
        for j in range(1, n + 1):
            # Build d × d block-diagonal matrix for this j
            B_j = np.zeros((d, d), dtype=complex)
            for (mu, lam_indices) in block_struct:
                F_mu = g_j_hats[j].get(mu)
                if F_mu is None:
                    continue
                d_mu = len(lam_indices)
                for ii in range(d_mu):
                    for jj in range(d_mu):
                        B_j[lam_indices[ii], lam_indices[jj]] = F_mu[ii, jj]

            F += rho(_tau(j, n)) @ B_j

        f_hat[lam] = F

    return f_hat


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from fourier import sn_dft

    rng = random.Random(7)

    for n in range(2, 6):
        for k in range(0, n + 1):
            # Build a random right-S_{n-k}-invariant function
            tuples = []
            def _k_tuples(chosen, remaining):
                if len(chosen) == k:
                    tuples.append(tuple(chosen))
                    return
                for x in remaining:
                    _k_tuples(chosen + [x], [y for y in remaining if y != x])
            _k_tuples([], list(range(1, n + 1)))

            f_tilde = {tup: rng.uniform(-1, 1) for tup in tuples}
            if k == 0:
                f_tilde = {(): rng.uniform(-1, 1)}

            # Run the coset FFT
            f_hat_coset = sn_mod_sk_fft(f_tilde, n, k)

            # Compare with full DFT on the expanded function
            f_full = from_coset_function(f_tilde, n, k)
            f_hat_dft = sn_dft(f_full, n)

            max_err = max(
                np.max(np.abs(f_hat_coset[lam] - f_hat_dft[lam]))
                for lam in partitions_of(n)
            )
            print(f"S_{n} / S_{n-k}  (k={k}, domain size={len(tuples) if k>0 else 1:4d})"
                  f"  max|coset_FFT − DFT| = {max_err:.2e}")
        print()
