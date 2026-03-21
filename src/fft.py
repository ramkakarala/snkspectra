
"""
Vibe coded (Claude)
Fast Fourier Transform (FFT) on the symmetric group S_n.

Implements Clausen's algorithm (1989), which computes the same Fourier
coefficients as the naive DFT but in O(n² · n!) operations instead of
O((n!)²), a speedup by a factor of roughly n!/n².

Background
----------
The non-abelian Fourier transform on S_n is

    f̂(ρ_λ) = Σ_{g ∈ S_n}  f(g) · ρ_λ(g)          (d_λ × d_λ matrix)

Clausen's algorithm exploits the subgroup chain S₁ ⊂ S₂ ⊂ ⋯ ⊂ S_n.
S_{n-1} ≤ S_n is the stabiliser of n (permutations that fix n).

Right-coset decomposition
    S_n = τ_1·S_{n-1}  ⊔  τ_2·S_{n-1}  ⊔  ⋯  ⊔  τ_n·S_{n-1}

where τ_k is any permutation with τ_k(n) = k.  We choose:
    τ_n = identity,     τ_k = transposition (k n)  for k < n.

Every σ ∈ S_n is uniquely written σ = τ_k · h  where k = σ(n) and
h = τ_k⁻¹ · σ ∈ S_{n-1} (h fixes n).

Define shifted functions  g_k : S_{n-1} → ℂ  by  g_k(h) = f(τ_k · h).

Then

    f̂(ρ_λ) = Σ_{k=1}^{n}  ρ_λ(τ_k) · BlockDiag_λ(ĝ_k)          (*)

where BlockDiag_λ(ĝ_k) is the d_λ × d_λ block-diagonal matrix whose
μ-block (for each child μ ≺ λ in the Young lattice) is the S_{n-1} Fourier
coefficient ĝ_k(μ).  The block positions follow the restriction rule

    ρ_λ |_{S_{n-1}} = ⊕_{μ ≺ λ}  ρ_μ                     (branching rule)

which is manifested in Young's orthogonal form: each SYT T of shape λ
restricts to the SYT T' of shape μ obtained by removing the cell containing n.

Algorithm
---------
    sn_fft(f, n):
        if n == 1: return directly
        for k = 1..n:
            define g_k(h) = f(τ_k · h)
            ĝ_k = sn_fft(g_k, n-1)          ← recursive call
        for each λ ⊢ n:
            f̂(ρ_λ) = Σ_k ρ_λ(τ_k) · BlockDiag_λ(ĝ_k)

Complexity: T(n) = n·T(n-1) + O(n · Σ_λ d_λ²) = O(n² · n!).
Naive DFT:  O(n! · Σ_λ d_λ²) = O((n!)²).

Output format
-------------
Identical to fourier.sn_dft: {partition: np.ndarray}.
The inverse and other utilities from fourier.py work unchanged.

References
----------
  Clausen, "Fast generalized Fourier transforms" (1989).
  Maslen & Rockmore, "Generalized FFTs — a survey of applications" (1997).
  Diaconis, "Group Representations in Probability and Statistics" (1988).
"""

import numpy as np
from functools import cache
from permutations import Permutation
from symmetric_group import SymmetricGroup
from irreps import irrep, partitions_of, standard_young_tableaux, YoungTableau


# ---------------------------------------------------------------------------
# Helpers: coset representatives and permutation embedding/projection
# ---------------------------------------------------------------------------

def _tau(k: int, n: int) -> Permutation:
    """
    Coset representative τ_k on {1,...,n} with τ_k(n) = k.
      - τ_n = identity
      - τ_k = transposition (k, n)  for k < n
    """
    if k == n:
        return Permutation.identity(n)
    mapping = {i: i for i in range(1, n + 1)}
    mapping[k] = n
    mapping[n] = k
    return Permutation(mapping)


def _extend(perm: Permutation) -> Permutation:
    """Embed a permutation on {1,...,n-1} into S_n by fixing n."""
    n = perm.n + 1
    mapping = {i: perm(i) for i in range(1, n)}
    mapping[n] = n
    return Permutation(mapping)


# ---------------------------------------------------------------------------
# Block structure: how the SYT basis of λ splits under restriction to S_{n-1}
# ---------------------------------------------------------------------------

def _restricted_shape(tab: YoungTableau, n: int) -> tuple[int, ...]:
    """
    Shape of the SYT obtained by removing the cell containing n.
    In a standard Young tableau n always sits at a removable corner.
    """
    r, _ = tab.position(n)
    parts = list(tab.shape)
    parts[r] -= 1
    while parts and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


@cache
def _block_structure(lam: tuple[int, ...]) -> list[tuple]:
    """
    For partition lam ⊢ n, compute the correspondence between SYTs of shape
    lam and the child shapes μ ≺ lam under the restriction S_n → S_{n-1}.

    Returns a list of (mu, lam_indices) pairs, one per child μ ≺ lam:
      - mu         : child partition (lam with one box removed)
      - lam_indices: list of length d_μ giving, for each j = 0,...,d_μ-1,
                     the index in standard_young_tableaux(lam) of the SYT
                     whose restricted shape is μ and whose restricted tableau
                     is the j-th SYT in standard_young_tableaux(μ).

    I.e., if B = BlockDiag_lam(matrices), then
        B[lam_indices[i], lam_indices[j]] = F_mu[i, j].
    """
    n = sum(lam)
    tabs_lam = standard_young_tableaux(lam)

    # Build: mu -> list of (lam_index, restricted YoungTableau)
    mu_data: dict[tuple, list] = {}
    for i, T in enumerate(tabs_lam):
        mu = _restricted_shape(T, n)
        rows_prime = [[v for v in row if v != n] for row in T.rows]
        rows_prime = [row for row in rows_prime if row]
        T_prime = YoungTableau(rows_prime)
        mu_data.setdefault(mu, []).append((i, T_prime))

    result = []
    for mu, pairs in mu_data.items():
        tabs_mu = standard_young_tableaux(mu)
        tab_idx_mu = {T: j for j, T in enumerate(tabs_mu)}
        # Sort by the mu-ordering of the restricted tableau so that
        # lam_indices[j] corresponds to the j-th SYT of shape mu.
        pairs_sorted = sorted(pairs, key=lambda p: tab_idx_mu[p[1]])
        lam_indices = [p[0] for p in pairs_sorted]
        result.append((mu, lam_indices))

    return result


# ---------------------------------------------------------------------------
# Core FFT (Clausen's algorithm)
# ---------------------------------------------------------------------------

def sn_fft(f: dict, n: int) -> dict[tuple, np.ndarray]:
    """
    Compute the Fourier transform of f on S_n via Clausen's FFT.

    Produces the same output as fourier.sn_dft but with complexity
    O(n² · n!) instead of O((n!)²).

    Parameters
    ----------
    f : dict {Permutation → complex}
        Function on S_n.  Missing elements are treated as 0.
    n : int
        Degree of the symmetric group.

    Returns
    -------
    f_hat : dict {partition → np.ndarray}
        f̂(ρ_λ) for every partition λ ⊢ n, as a d_λ × d_λ complex array.
        (Same format as fourier.sn_dft — the inverse sn_idft works unchanged.)
    """
    # --- Base case ---
    if n == 1:
        e = Permutation.identity(1)
        return {(1,): np.array([[f.get(e, 0.0)]], dtype=complex)}

    # --- Step 1: build the n shifted functions g_k : S_{n-1} → ℂ ---
    # g_k(h) = f(τ_k · h),   h ∈ S_{n-1}  (viewed as perms on {1,...,n-1})
    Sn1 = SymmetricGroup(n - 1)
    g_k_funcs: dict[int, dict] = {}
    for k in range(1, n + 1):
        tk = _tau(k, n)
        g_k = {}
        for h in Sn1:
            sigma = tk * _extend(h)
            val = f.get(sigma, 0.0)
            if val != 0.0:
                g_k[h] = val
        g_k_funcs[k] = g_k

    # --- Step 2: recursively FFT each g_k on S_{n-1} ---
    g_k_hats: dict[int, dict] = {
        k: sn_fft(g_k_funcs[k], n - 1) for k in range(1, n + 1)
    }

    # --- Step 3: assemble f̂(ρ_λ) = Σ_k ρ_λ(τ_k) · BlockDiag_λ(ĝ_k) ---
    f_hat: dict[tuple, np.ndarray] = {}
    for lam in partitions_of(n):
        rho = irrep(lam)
        d = len(standard_young_tableaux(lam))
        block_struct = _block_structure(lam)

        F = np.zeros((d, d), dtype=complex)
        for k in range(1, n + 1):
            # Build the d × d block-diagonal matrix B_k
            B_k = np.zeros((d, d), dtype=complex)
            for (mu, lam_indices) in block_struct:
                F_mu = g_k_hats[k].get(mu)
                if F_mu is None:
                    continue
                d_mu = len(lam_indices)
                # Scatter F_mu into the appropriate rows/cols of B_k
                for ii in range(d_mu):
                    for jj in range(d_mu):
                        B_k[lam_indices[ii], lam_indices[jj]] = F_mu[ii, jj]

            F += rho(_tau(k, n)) @ B_k

        f_hat[lam] = F

    return f_hat


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import math
    import time
    from fourier import sn_dft, sn_idft

    rng = random.Random(42)

    for n in range(2, 9 ):
        Sn = SymmetricGroup(n)
        elements = Sn.elements()
        f = {g: rng.uniform(-1, 1) for g in elements}

        # Time DFT
        t0 = time.perf_counter()
        f_hat_dft = sn_dft(f, n)
        t_dft = time.perf_counter() - t0

        # Time FFT
        t0 = time.perf_counter()
        f_hat_fft = sn_fft(f, n)
        t_fft = time.perf_counter() - t0

        # Check agreement
        max_err = max(
            np.max(np.abs(f_hat_fft[lam] - f_hat_dft[lam]))
            for lam in f_hat_dft
        )

        # Check inversion
        f_rec = sn_idft(f_hat_fft, n)
        inv_err = max(abs(f[g] - f_rec[g]) for g in elements)

        print(
            f"S_{n} (order {math.factorial(n):4d}):"
            f"  max|FFT−DFT|={max_err:.2e}"
            f"  inv_err={inv_err:.2e}"
            f"  DFT={t_dft*1e3:.1f}ms  FFT={t_fft*1e3:.1f}ms"
        )
