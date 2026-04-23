#!/usr/bin/env python3
"""
Coordinate function matrix for all Young irreps of S_n.

The coordinate functions (matrix coefficients) of an irrep rho_lam are
    phi^lam_{ij}(sigma) = rho_lam(sigma)[i, j].

By Schur orthogonality, the scaled functions sqrt(d_lam) * phi^lam_{ij} form
an orthonormal basis of L^2(S_n) (inner product (1/n!) sum_sigma f(sigma)g(sigma)).
Since sum_lam d_lam^2 = n!, there are exactly n! such functions.

This script arranges them as an n! x n! matrix F:
    rows  ~ (lambda, i, j) for lambda |- n, 0 <= i,j < d_lambda, lex order
    cols  ~ sigma in S_n in lex order of one-line notation
    F[row, col] = sqrt(d_lambda) * rho_lambda(sigma)[i, j]

F is real and orthogonal: F @ F.T = F.T @ F = I.
"""

import sys
import numpy as np
from itertools import permutations as iperms

from permutations import Permutation
from irreps import irrep, partitions_of


def all_permutations(n):
    """All elements of S_n in lex order of one-line notation."""
    return [Permutation(list(p)) for p in iperms(range(1, n + 1))]


def coord_matrix(n):
    """
    Build the n! x n! coordinate function matrix for S_n.

    Returns
    -------
    F : ndarray, shape (n!, n!), dtype float64
    row_labels : list of (partition, i, j) tuples
    col_perms  : list of Permutation objects (column order)
    """
    parts = partitions_of(n)
    col_perms = all_permutations(n)
    N = len(col_perms)  # = n!

    F = np.zeros((N, N), dtype=float)
    row_labels = []

    row = 0
    for lam in parts:
        rho = irrep(lam)
        # compute rep matrix for every permutation once
        mats = [rho(sigma) for sigma in col_perms]
        d = mats[0].shape[0]
        # Schur orthogonality: sum_sigma rho(sigma)_ij * rho(sigma)_kl = (n!/d) delta_ik delta_jl
        # Scaling by sqrt(d/n!) makes the rows orthonormal.
        scale = np.sqrt(d / N)

        for i in range(d):
            for j in range(d):
                row_labels.append((lam, i, j))
                F[row] = scale * np.array([mat[i, j] for mat in mats])
                row += 1

    assert row == N, f"Row count mismatch: {row} != {N}"
    return F, row_labels, col_perms


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print(f"Building coordinate matrix for S_{n}  (size {__import__('math').factorial(n)}x{__import__('math').factorial(n)})...")
    F, row_labels, col_perms = coord_matrix(n)
    N = len(col_perms)

    print(F)
    print()

    # Verify orthogonality
    FtF = F.T @ F
    FFt = F @ F.T
    I = np.eye(len(col_perms))
    print(f"F.T @ F = I:  {np.allclose(FtF, I)}")
    print(f"F @ F.T = I:  {np.allclose(FFt, I)}")
    print(f"max |F.T@F - I| = {np.max(np.abs(FtF - I)):.2e}")
    print()

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(F)
    # Sort by real part, then imaginary part
    idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print("Eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        if abs(ev.imag) < 1e-10:
            print(f"  {i:3d}: {ev.real:+.6f}")
        else:
            print(f"  {i:3d}: {ev.real:+.6f} + {ev.imag:+.6f}j")

    # Summarise unique eigenvalues and their multiplicities
    unique, counts = np.unique(np.round(eigenvalues, 8), return_counts=True)
    print()
    print("Unique eigenvalues and multiplicities:")
    for uv, cnt in zip(unique, counts):
        if abs(uv.imag) < 1e-8:
            print(f"  {uv.real:+.6f}  (multiplicity {cnt})")
        else:
            print(f"  {uv.real:+.6f} + {uv.imag:+.6f}j  (multiplicity {cnt})")
    print()
    print("Eigenvectors (columns), sorted by eigenvalue:")
    print(np.round(eigenvectors, 6))

    # --- Check if F has finite order (F^k = I for some k) ---
    # F is orthogonal so all eigenvalues lie on the unit circle.
    # F^k = I iff every eigenvalue is a k-th root of unity,
    # i.e. its angle is a rational multiple of 2*pi.
    from fractions import Fraction
    from math import gcd
    from functools import reduce

    def lcm(a, b):
        return a * b // gcd(a, b)

    angles = np.angle(eigenvalues)           # in (-pi, pi]
    max_denom = 1000                         # search up to this denominator
    periods = []
    rational = True
    for ang in angles:
        frac = ang / (2 * np.pi)             # eigenvalue = e^{2*pi*i * frac}
        # approximate frac as p/q with denominator <= max_denom
        f = Fraction(frac).limit_denominator(max_denom)
        # verify the approximation is tight
        if abs(float(f) - frac) > 1e-6:
            rational = False
            break
        periods.append(abs(f.denominator) if f.numerator != 0 else 1)

    print()
    if not rational:
        print("Period check: eigenvalue angles are NOT rational multiples of 2π.")
        print("  F does not appear to have finite order.")
    else:
        period = reduce(lcm, periods)
        print(f"Eigenvalue angles as fractions of 2π:")
        seen = set()
        for ang, p in zip(angles, periods):
            frac = Fraction(ang / (2 * np.pi)).limit_denominator(max_denom)
            if frac not in seen:
                seen.add(frac)
                print(f"  {float(frac):+.6f}  =  {frac}  (order {abs(frac.denominator) if frac.numerator != 0 else 1})")

        # Verify numerically
        Fk = np.linalg.matrix_power(F, period)
        is_identity = np.allclose(Fk, np.eye(N), atol=1e-8)
        print(f"\nPeriod k = {period}")
        print(f"F^{period} = I:  {is_identity}  (max error {np.max(np.abs(Fk - np.eye(N))):.2e})")

        # Also check all divisors to find the minimal period
        divisors = sorted(d for d in range(1, period + 1) if period % d == 0)
        print(f"\nMinimal period search over divisors of {period}: {divisors}")
        for d in divisors:
            Fd = np.linalg.matrix_power(F, d)
            if np.allclose(Fd, np.eye(N), atol=1e-8):
                print(f"  F^{d} = I  ← minimal period")
                break
            else:
                print(f"  F^{d} ≠ I  (max error {np.max(np.abs(Fd - np.eye(N))):.2e})")

    # --- Plot eigenvalues on the complex plane ---
    import matplotlib.pyplot as plt

    # Group by proximity to find unique eigenvalues and multiplicities
    tol = 1e-6
    visited = np.zeros(len(eigenvalues), dtype=bool)
    unique_evs, multiplicities = [], []
    for i, ev in enumerate(eigenvalues):
        if not visited[i]:
            close = np.abs(eigenvalues - ev) < tol
            unique_evs.append(ev)
            multiplicities.append(int(close.sum()))
            visited |= close

    fig, ax = plt.subplots(figsize=(6, 6))

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="lightgray", lw=1, zorder=0)
    ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.5, zorder=0)

    # Scatter: size and label encode multiplicity
    xs = [ev.real for ev in unique_evs]
    ys = [ev.imag for ev in unique_evs]
    ms = multiplicities
    sc = ax.scatter(xs, ys, s=[60 * m for m in ms], c="steelblue",
                    zorder=3, edgecolors="navy", linewidths=0.8)

    for x, y, m in zip(xs, ys, ms):
        ax.annotate(str(m), (x, y),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=9, color="navy")

    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(f"Eigenvalues of coord matrix $F$ for $S_{{{n}}}$\n"
                 f"(dot size ∝ multiplicity; label = multiplicity)")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    plt.tight_layout()
    out = f"eigenvalues_S{n}.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()
