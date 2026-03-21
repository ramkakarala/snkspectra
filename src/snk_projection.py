"""
Vibe coded (Claude)
Projection matrices for the DFT coefficients of right-S_{n-k}-invariant functions.

Mathematical background
-----------------------
Let  S_{n-k} ≤ S_n  be the subgroup that fixes the last k positions pointwise:

    S_{n-k} = { τ ∈ S_n : τ(i) = i  for  i = n-k+1, …, n }

A function  f : S_n → ℂ  is right-S_{n-k}-invariant if

    f(σ·τ) = f(σ)   for all τ ∈ S_{n-k}.

Claim
-----
The Fourier coefficient  f̂(ρ_λ) = Σ_{σ ∈ S_n} f(σ) ρ_λ(σ)  satisfies

    f̂(ρ_λ) · ρ_λ(τ) = f̂(ρ_λ)   for all τ ∈ S_{n-k}.

Proof:
    f̂(λ) · ρ_λ(τ) = Σ_σ f(σ) ρ_λ(σ) ρ_λ(τ)
                   = Σ_σ f(σ) ρ_λ(σ·τ)
                   = Σ_{σ'} f(σ'·τ⁻¹) ρ_λ(σ')   [σ' = σ·τ]
                   = Σ_{σ'} f(σ') ρ_λ(σ')         [right invariance]
                   = f̂(λ).

Projection matrix
-----------------
Define, for each irrep λ ⊢ n, the d_λ × d_λ matrix

    P_λ  =  (1/|S_{n-k}|) Σ_{τ ∈ S_{n-k}} ρ_λ(τ)

This is the orthogonal projection onto the S_{n-k}-fixed subspace of V_λ:

    Im(P_λ) = { v ∈ V_λ : ρ_λ(τ)v = v  for all τ ∈ S_{n-k} }

Any right-S_{n-k}-invariant function satisfies

    f̂(ρ_λ) = f̂(ρ_λ) · P_λ   (the columns of f̂ lie in Im(P_λ)).

Properties of P_λ
-----------------
  • Orthogonal projection:   P_λ² = P_λ,   P_λᵀ = P_λ
  • Eigenvalues:             0 and 1 only
  • Rank:                    dim Im(P_λ) = multiplicity of the trivial rep of
                             S_{n-k} in ρ_λ|_{S_{n-k}}
                           = multiplicity of ρ_λ in Ind^{S_n}_{S_{n-k}}(trivial)
                             [Frobenius reciprocity]
  • Counting:                Σ_{λ ⊢ n} d_λ · rank(P_λ) = n!/(n-k)!
                             (= size of the coset space S_n/S_{n-k})

Special cases
-------------
  k = 0  :  S_{n-k} = S_n  →  P_λ = I_{1} for λ=(n,), else 0  (only trivial survives)
  k = n  :  S_{n-k} = {e}  →  P_λ = I_{d_λ}  for all λ       (no constraint)
"""

import math
import numpy as np
from itertools import permutations as _perms

from permutations import Permutation
from symmetric_group import SymmetricGroup
from irreps import irrep, partitions_of, standard_young_tableaux


# ---------------------------------------------------------------------------
# Core: projection matrices
# ---------------------------------------------------------------------------

def snk_projection(n: int, k: int) -> dict[tuple, np.ndarray]:
    """
    Compute the projection matrix P_λ for each irrep λ ⊢ n.

        P_λ = (1/|S_{n-k}|) Σ_{τ ∈ S_{n-k}} ρ_λ(τ)

    where  S_{n-k} = {τ ∈ S_n : τ(i) = i for i = n-k+1,...,n}.

    Parameters
    ----------
    n : int   degree of S_n
    k : int   coset dimension  (0 ≤ k ≤ n)

    Returns
    -------
    projections : dict { partition λ ⊢ n  →  d_λ × d_λ np.ndarray }
        Each matrix is the orthogonal projection onto the S_{n-k}-fixed
        subspace of V_λ.  Any right-S_{n-k}-invariant f satisfies
            f̂(ρ_λ) = f̂(ρ_λ) · projections[λ].
    """
    if k < 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 ≤ k ≤ n={n}")

    m = n - k                        # degree of the subgroup
    order_snk = math.factorial(m)    # |S_{n-k}|

    # Enumerate S_{n-k}: all permutations of {1,...,m}, fixing {m+1,...,n}
    snk_elements = [
        Permutation(list(p) + list(range(m + 1, n + 1)))
        for p in _perms(range(1, m + 1))
    ] if m > 0 else [Permutation.identity(n)]

    projections: dict[tuple, np.ndarray] = {}
    for lam in partitions_of(n):
        rho = irrep(lam)
        d = len(standard_young_tableaux(lam))
        P = sum(rho(tau) for tau in snk_elements) / order_snk
        projections[lam] = P

    return projections


def projection_summary(n: int, k: int) -> dict[tuple, dict]:
    """
    Compute P_λ and its key properties for each λ ⊢ n.

    Returns a dict { λ → { 'P': matrix, 'rank': int, 'dim': int } } where
      'rank' = rank of P_λ = dim of S_{n-k}-fixed subspace of V_λ
      'dim'  = d_λ = dimension of V_λ
    """
    projs = snk_projection(n, k)
    summary = {}
    for lam, P in projs.items():
        d = P.shape[0]
        rank = int(round(np.trace(P).real))   # rank = trace for a projection
        summary[lam] = {'P': P, 'rank': rank, 'dim': d}
    return summary


def apply_projection(f_hat: dict, n: int, k: int) -> dict:
    """
    Project DFT coefficients onto the right-S_{n-k}-invariant subspace.

    Given any f̂ = {λ: matrix}, returns { λ: f̂(λ) · P_λ }.
    If f was truly right-S_{n-k}-invariant then the result equals f̂.

    Parameters
    ----------
    f_hat : dict { partition → np.ndarray }   (output of sn_dft or sn_fft)
    n, k  : int

    Returns
    -------
    projected : dict { partition → np.ndarray }
    """
    projs = snk_projection(n, k)
    return {lam: F @ projs[lam] for lam, F in f_hat.items()}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from fourier import sn_dft
    from coset_fft import from_coset_function

    print("=" * 70)
    print("Projection matrices P_λ for right-S_{n-k}-invariant functions")
    print("=" * 70)

    for n in range(2, 6):
        print(f"\n── S_{n}  (order {math.factorial(n)}) ──")
        for k in range(1, n):
            summ = projection_summary(n, k)
            coset_size = math.factorial(n) // math.factorial(n - k)
            total_active = sum(info['rank'] * info['dim'] for info in summ.values())

            print(f"\n  k={k}  S_{n}/S_{n-k}  coset size={coset_size}"
                  f"  Σ d_λ·rank(P_λ)={total_active}"
                  f"  (should equal {coset_size})")
            for lam, info in summ.items():
                P = info['P']
                rank, dim = info['rank'], info['dim']

                # Verify P is an orthogonal projection
                is_proj = np.allclose(P @ P, P)
                is_sym  = np.allclose(P, P.T)

                print(f"    λ={str(lam):12s}  dim={dim}  rank(P_λ)={rank}"
                      f"  P²=P:{is_proj}  Pᵀ=P:{is_sym}")
                # Print the matrix, indented
                if np.allclose(P, 0):
                    print(f"      np.zeros(({dim}, {dim}))")
                elif np.allclose(P, np.eye(dim)):
                    print(f"      np.eye({dim})")
                else:
                    for row in np.round(P, 6):
                        print("      [" + "  ".join(f"{v:8.4f}" for v in row) + "]")

    # -----------------------------------------------------------------------
    # Verification: for random right-S_{n-k}-invariant f,
    #               f̂(λ) = f̂(λ) · P_λ  for all λ.
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Verification: f̂(λ) = f̂(λ) · P_λ  for random invariant functions")
    print("=" * 70)

    rng = random.Random(42)
    for n in range(2, 6):
        for k in range(1, n):
            # Build a random right-S_{n-k}-invariant function via coset reps
            from itertools import permutations as _pt
            tuples = list(_pt(range(1, n + 1), k)) if k > 0 else [()]
            f_tilde = {tup: rng.uniform(-1, 1) for tup in tuples}
            f_full = from_coset_function(f_tilde, n, k)
            f_hat = sn_dft(f_full, n)

            projs = snk_projection(n, k)
            max_err = max(
                np.max(np.abs(f_hat[lam] - f_hat[lam] @ projs[lam]))
                for lam in partitions_of(n)
            )
            status = "✓" if max_err < 1e-10 else "✗"
            print(f"  S_{n}/S_{n-k}  k={k}: max |f̂(λ) - f̂(λ)·P_λ| = {max_err:.2e}  {status}")
