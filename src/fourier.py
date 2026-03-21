
"""
Vibe coded (Claude)
Discrete Fourier Transform (DFT) on the symmetric group S_n.

For a function f: S_n → ℂ the non-abelian Fourier transform is a collection
of matrices, one per irreducible representation (irrep) of S_n:

    f̂(ρ_λ) = Σ_{g ∈ S_n}  f(g) · ρ_λ(g)          (a d_λ × d_λ matrix)

The irreps of S_n are indexed by partitions λ ⊢ n; d_λ is the dimension of
irrep λ (= number of standard Young tableaux of shape λ).

Inverse DFT:
    f(g) = (1/|S_n|) Σ_{λ ⊢ n}  d_λ · tr( f̂(ρ_λ) · ρ_λ(g⁻¹) )

Plancherel / Parseval identity:
    Σ_{g ∈ S_n} |f(g)|² = (1/|S_n|) Σ_{λ ⊢ n}  d_λ · ‖f̂(ρ_λ)‖²_F

where ‖·‖_F is the Frobenius norm.

References:
  Diaconis, "Group Representations in Probability and Statistics" (1988).
  Maslen & Rockmore, "Generalized FFTs — a survey of applications" (1997).
  Sagan, "The Symmetric Group" (2001), Ch. 1–2.
"""

import numpy as np
from symmetric_group import SymmetricGroup
from irreps import irrep, partitions_of
from permutations import Permutation


# ---------------------------------------------------------------------------
# Forward DFT
# ---------------------------------------------------------------------------

def sn_dft(
    f: dict,
    n: int,
) -> dict[tuple, np.ndarray]:
    """
    Compute the Fourier transform of f on S_n.

    Parameters
    ----------
    f : dict  {Permutation: complex}
        A function on S_n.  Missing permutations are treated as 0.
    n : int
        The degree of the symmetric group.

    Returns
    -------
    f_hat : dict  {partition: np.ndarray}
        Keyed by partitions λ ⊢ n.  Each value is a d_λ × d_λ complex matrix
            f̂(ρ_λ) = Σ_{g ∈ S_n}  f(g) · ρ_λ(g)
    """
    Sn = SymmetricGroup(n)
    f_hat = {}
    for lam in partitions_of(n):
        rho = irrep(lam)
        # Determine matrix dimension from the identity element
        dim = rho(Sn.identity()).shape[0]
        F = np.zeros((dim, dim), dtype=complex)
        for g in Sn:
            val = f.get(g, 0)
            if val != 0:
                F += val * rho(g)
        f_hat[lam] = F
    return f_hat


# ---------------------------------------------------------------------------
# Inverse DFT
# ---------------------------------------------------------------------------

def sn_idft(
    f_hat: dict[tuple, np.ndarray],
    n: int,
) -> dict:
    """
    Recover f from its Fourier transform f̂ on S_n.

    Parameters
    ----------
    f_hat : dict  {partition: np.ndarray}
        Output of sn_dft (or any dict of d_λ × d_λ matrices keyed by
        partitions of n).
    n : int
        Degree of the symmetric group.

    Returns
    -------
    f : dict  {Permutation: complex}
        The recovered function on S_n.  Imaginary parts that are numerically
        zero (< 1e-12) are discarded to give real values where appropriate.
    """
    Sn = SymmetricGroup(n)
    order = Sn.order()            # |S_n| = n!
    f = {}
    for g in Sn:
        g_inv = g.inverse()
        value = 0.0 + 0.0j
        for lam, F in f_hat.items():
            rho = irrep(lam)
            d = F.shape[0]
            value += d * np.trace(F @ rho(g_inv))
        value /= order
        # Drop negligible imaginary part for real-valued functions
        f[g] = value.real if abs(value.imag) < 1e-10 else value
    return f


# ---------------------------------------------------------------------------
# Utility: Plancherel norm
# ---------------------------------------------------------------------------

def plancherel_norm_sq(f_hat: dict[tuple, np.ndarray], n: int) -> float:
    """
    Compute (1/|S_n|) Σ_λ d_λ · ‖f̂(ρ_λ)‖²_F.

    By the Plancherel identity this equals Σ_{g ∈ S_n} |f(g)|².
    """
    from math import factorial
    order = factorial(n)
    total = 0.0
    for lam, F in f_hat.items():
        d = F.shape[0]
        total += d * np.linalg.norm(F, "fro") ** 2
    return total / order


# ---------------------------------------------------------------------------
# Utility: convolution via Fourier transform
# ---------------------------------------------------------------------------

def sn_convolve(f: dict, h: dict, n: int) -> dict:
    """
    Compute the convolution (f * h)(g) = Σ_{x ∈ S_n} f(x) h(x⁻¹ g)
    using the pointwise product of Fourier transforms:

        (f * h)^(ρ) = f̂(ρ) · ĥ(ρ)   (matrix product)

    Parameters
    ----------
    f, h : dict  {Permutation: complex}
    n : int

    Returns
    -------
    conv : dict  {Permutation: complex}
    """
    f_hat = sn_dft(f, n)
    h_hat = sn_dft(h, n)
    conv_hat = {lam: f_hat[lam] @ h_hat[lam] for lam in f_hat}
    return sn_idft(conv_hat, n)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    for n in range(2, 5):
        Sn = SymmetricGroup(n)
        order = Sn.order()
        elements = Sn.elements()
        parts = partitions_of(n)

        print(f"\n{'='*60}")
        print(f"S_{n}  (order {order})  partitions: {parts}")

        # --- Example 1: uniform function f ≡ 1 --------------------------
        # f̂(trivial) = n!,  f̂(all other λ) = 0 × 0 zero matrix
        f_uniform = {g: 1.0 for g in elements}
        f_hat = sn_dft(f_uniform, n)
        print(f"\n  f ≡ 1  (uniform)")
        for lam, F in f_hat.items():
            print(f"    λ={lam}  f̂ =\n{np.round(F, 6)}")

        # --- Example 2: delta at identity --------------------------------
        # f̂(ρ_λ) = I_λ  (identity matrix of the irrep dimension)
        e = Sn.identity()
        f_delta = {g: (1.0 if g == e else 0.0) for g in elements}
        f_hat_delta = sn_dft(f_delta, n)
        print(f"\n  f = δ_e  (delta at identity)")
        for lam, F in f_hat_delta.items():
            d = F.shape[0]
            is_identity = np.allclose(F, np.eye(d))
            print(f"    λ={lam}  f̂ = I_{d}? {is_identity}")

        # --- Verify Plancherel identity ----------------------------------
        f_values = list(f_uniform.values())
        l2_sq = sum(abs(v) ** 2 for v in f_values)
        planch_sq = plancherel_norm_sq(f_hat, n)
        print(f"\n  Plancherel check (f≡1):  Σ|f|²={l2_sq:.4f}  (1/|G|)Σd_λ‖f̂‖²={planch_sq:.4f}  ok={np.isclose(l2_sq, planch_sq)}")

        # --- Verify inversion --------------------------------------------
        if n <= 3:
            # Use sign function as a more interesting example
            f_sign = {g: float(g.sign()) for g in elements}
            f_hat_sign = sn_dft(f_sign, n)
            f_recovered = sn_idft(f_hat_sign, n)
            recon_ok = all(
                np.isclose(f_sign[g], f_recovered[g]) for g in elements
            )
            print(f"  Inversion check (f=sign): {'✓' if recon_ok else '✗'}")

        # --- Verify convolution theorem ----------------------------------
        if n <= 3:
            # Direct convolution: (f*h)(g) = Σ_x f(x) h(x⁻¹g)
            f_test = {g: float(i) for i, g in enumerate(elements)}
            h_test = {g: float(g.sign()) for g in elements}

            conv_direct = {}
            for g in elements:
                val = sum(f_test.get(x, 0) * h_test.get(x.inverse() * g, 0) for x in elements)
                conv_direct[g] = val

            conv_fft = sn_convolve(f_test, h_test, n)
            conv_ok = all(
                np.isclose(conv_direct[g], conv_fft[g]) for g in elements
            )
            print(f"  Convolution theorem check: {'✓' if conv_ok else '✗'}")
