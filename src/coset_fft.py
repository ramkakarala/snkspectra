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
# Linear-time FFT for right S_{n-1}-invariant functions on S_n
# Clausen & Kakarala (2010), Applied Mathematics Letters 23, 183–187.
# ---------------------------------------------------------------------------
#
# Setup
# -----
# S_{n-1} ≤ S_n is the stabiliser of n (permutations that fix n).
# A right S_{n-1}-invariant function is constant on the LEFT cosets
#
#     L_j = { σ ∈ S_n | σ(n) = j },   j = 1, …, n ,
#
# so it is identified with a length-n vector  f̃_j = f(σ)  for any σ ∈ L_j.
# (Same parameterisation as  sn_mod_sk_fft  with  k = 1.)
#
# Key observation (Clausen & Kakarala §3)
# ----------------------------------------
# The idempotent D^λ(ι_{n-1}) is zero for every irrep λ ⊢ n except
#   (n)    — the trivial irrep       (degree 1)
#   (n-1,1) — the standard irrep     (degree n-1)
# so all other Fourier matrices vanish.
#
# The matrix D^{(n-1,1)}(f) is (n-1)×(n-1) with only its LAST column
# non-zero; that column equals y = C · r, where
#
#   r_j = (n-1)! · f̃_j                     (scaled signal, j = 1…n)
#
#   C  is the (n-1)×n matrix (eq. 6 of the paper):
#       C[i, j] = ε_{i+1}   for j ≤ i      (ε_d = −1/d)
#       C[i, i+1] = 1
#       C[i, j] = 0         for j > i+1
#
# Straight-line program for y = C · r (2n−3 additions, n−2 multiplications):
#   Step 1:  s_k = r_1 + … + r_k  for k = 1, …, n-1   (n-2 additions)
#   Step 2:  y_j = ε_j · s_j + r_{j+1}  for j = 1, …, n-1
#                                                (n-1 additions, n-2 mults)
#   Trivial: D^{(n)}(f) = s_{n-1} + r_n          (1 addition)
# Total: 2n-2 additions + n-2 multiplications.


def _straight_line_program(r: np.ndarray) -> tuple[np.ndarray, complex]:
    """
    Straight-line program for the matrix C from Clausen & Kakarala (2010) eq. (6).

    Computes  y = C · r  and the trivial Fourier coefficient  Σ_j r_j.

    Parameters
    ----------
    r : np.ndarray of shape (n,), dtype complex
        Scaled signal vector  r_j = (n-1)! · f̃_j  (1-indexed, stored 0-indexed).

    Returns
    -------
    y     : np.ndarray of shape (n-1,)
        The last column of D^{(n-1,1)}(f).
    total : complex
        Σ_j r_j  =  D^{(n)}(f)  (trivial Fourier coefficient).
    """
    n = len(r)
    if n == 1:
        return np.empty(0, dtype=complex), complex(r[0])

    # Step 1: cumulative sums  s[k] = r[0] + … + r[k]  for k = 0, …, n-2
    s = np.cumsum(r[:-1])           # length n-1

    # Step 2: y[i] = (−1/(i+1)) · s[i] + r[i+1]  for i = 0, …, n-2
    eps = -1.0 / np.arange(1, n, dtype=float)   # ε_{i+1} = −1/(i+1)
    y = eps * s + r[1:]

    # Trivial component
    total = s[-1] + r[-1]

    return y, complex(total)


def sn_mod_sn1_fft(f_tilde: dict, n: int) -> dict[tuple, np.ndarray]:
    """
    Linear-time Fourier transform for right S_{n-1}-invariant functions on S_n.

    Implements the straight-line algorithm of Clausen & Kakarala (2010),
    *Applied Mathematics Letters* 23 (2010) 183–187, Theorem 3.1.

    Representation convention
    -------------------------
    The algorithm uses Young's *seminormal* form and its dual D̃^{semi}
    (not the orthogonal form used by ``irrep``/``sn_mod_sk_fft``).  For n ≥ 3
    the Fourier matrices differ from those of ``sn_mod_sk_fft`` for the
    standard irrep (n-1, 1) — they are equivalent representations related by
    a change of basis.  The trivial component (n,) and all character values are
    identical in both conventions.

    Cost: 2n − 2 additions + n − 2 scalar multiplications.

    Parameters
    ----------
    f_tilde : dict { (j,) → complex },  j ∈ {1, …, n}
        The coset function.  f_tilde[(j,)] = f(σ) for any σ with σ(n) = j.
        Same parameterisation as the k=1 input to  sn_mod_sk_fft.
        Missing entries are treated as 0.
    n : int
        Degree of S_n (n ≥ 1).

    Returns
    -------
    f_hat : dict { partition λ ⊢ n → np.ndarray (d_λ × d_λ) }
        Fourier matrices in Young's seminormal dual (D̃^{semi}) convention.
        Non-zero only for λ = (n) (trivial) and λ = (n-1, 1) (standard).
        For λ = (n-1, 1) only the last column is non-zero; it equals C · r
        where C is the (n-1)×n matrix of eq. (6) and r_j = (n-1)! · f̃_j.
    """
    if n < 1:
        raise ValueError(f"n={n} must be ≥ 1")

    # Scale: paper's r_j = (n-1)! · f̃_j
    fact = math.factorial(n - 1)
    r = np.array([f_tilde.get((j,), 0.0) * fact for j in range(1, n + 1)],
                 dtype=complex)

    # Initialise all Fourier matrices to zero
    f_hat: dict[tuple, np.ndarray] = {}
    for lam in partitions_of(n):
        d = len(standard_young_tableaux(lam))
        f_hat[lam] = np.zeros((d, d), dtype=complex)

    # n = 1: one coset, one irrep (1,)
    if n == 1:
        f_hat[(1,)][0, 0] = r[0]
        return f_hat

    # Straight-line program
    y, total = _straight_line_program(r)

    # Trivial irrep (n,): 1×1 matrix = Σ_j r_j
    f_hat[(n,)][0, 0] = total

    # Standard irrep (n-1, 1): (n-1)×(n-1) matrix, last column = y, rest zero
    standard = (n - 1, 1)
    f_hat[standard][:, n - 2] = y

    return f_hat


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from fourier import sn_dft

    rng = random.Random(7)

    # --- Existing coset FFT (all k) -----------------------------------------
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

    # --- Clausen & Kakarala (2010) straight-line FFT (k=1 only) -------------
    # The paper uses Young's *seminormal* form (and its dual D̃^semi).
    # The existing `irrep` uses Young's *orthogonal* form; these differ for
    # n ≥ 3.  We verify the straight-line output by independently constructing
    # the C matrix from the D̃^semi transposition matrices (J_k^T blocks)
    # and comparing against the closed-form C of eq. (6) and against y = C·r.
    print("Straight-line FFT  (Clausen & Kakarala 2010, right S_{n-1}-invariant)")
    print("  [D̃^semi convention; compared against C built from J_k^T blocks]")
    rng2 = random.Random(42)

    def _C_from_Dtilde_semi(n):
        """
        Build the (n-1)×n matrix C̃ by backward recursion with D̃^semi blocks.
        D̃^semi(τ_j) = I_{j-2} ⊕ J_j^T ⊕ I_{n-j-1},  J_j^T[0,1] = 1.
        Starting vector: e_{n-2} (last standard basis vector).
        """
        c = np.zeros(n - 1, dtype=float)
        c[n - 2] = 1.0
        cols = [c.copy()]                # column n (j = n), stored first
        for j in range(n - 1, 0, -1):   # j = n-1 down to 1
            if j == 1:
                c[0] = -c[0]            # D̃^semi(τ_1) = diag(-1, 1, …, 1)
            else:
                i0, i1 = j - 2, j - 1
                x0, x1 = c[i0], c[i1]
                # J_j^T = [[1/j, 1], [1-1/j^2, -1/j]]
                c[i0] = x0 / j + x1
                c[i1] = (1.0 - 1.0 / j**2) * x0 - x1 / j
            cols.insert(0, c.copy())    # column j prepended
        return np.column_stack(cols)    # shape (n-1) × n

    def _C_closed_form(n):
        """Paper's eq. (6): C̃[i, j] = ε_{i+1} for j ≤ i, 1 for j = i+1, 0 else."""
        C = np.zeros((n - 1, n), dtype=float)
        for i in range(n - 1):
            eps = -1.0 / (i + 1)
            C[i, :i + 1] = eps
            C[i, i + 1] = 1.0
        return C

    for n in range(1, 8):
        f_tilde = {(j,): rng2.uniform(-1, 1) for j in range(1, n + 1)}
        f_hat_sl = sn_mod_sn1_fft(f_tilde, n)

        if n == 1:
            # Only trivial irrep; no standard rep
            fact = math.factorial(n - 1)
            r = np.array([f_tilde[(j,)] * fact for j in range(1, n + 1)])
            err_C, err_y = 0.0, 0.0
        else:
            # Build reference C matrix two ways and compare
            C_semi = _C_from_Dtilde_semi(n)       # from J_k^T blocks
            C_paper = _C_closed_form(n)            # eq. (6) closed form

            fact = math.factorial(n - 1)
            r = np.array([f_tilde.get((j,), 0.0) * fact for j in range(1, n + 1)])

            err_C = np.max(np.abs(C_semi - C_paper))   # both C's agree?
            y_ref = C_semi @ r                          # last col from C_semi
            y_sl  = f_hat_sl[(n - 1, 1)][:, n - 2]    # last col from straight-line
            err_y = np.max(np.abs(y_sl - y_ref))

        print(f"  n={n}  max|C_semi − C_paper|={err_C:.2e}"
              f"  max|y_sl − C_semi·r|={err_y:.2e}")
