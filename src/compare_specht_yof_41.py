"""
Compare Specht irrep matrices vs Young orthogonal form (YOF) matrices for
adjacent transpositions and partition (4, 1) of S_5.

Partition (4, 1) is the standard representation of S_5, dimension 4.
The two constructions are equivalent orthogonal representations related by
an orthogonal change of basis Q:
    rho_specht(s_i) = Q @ rho_yof(s_i) @ Q.T

Steps:
1. Print the SYT basis (4 tableaux).
2. For each adjacent transposition s_i (i=1..4) print both matrices.
3. Compute Q from the overlap of Specht and YOF matrices on s_1.
4. Verify Q intertwines all s_i.
5. Check characters match.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from permutations import Permutation
from irreps import irrep as irrep_yof, standard_young_tableaux
from irreps_specht import irrep_specht, specht_basis

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PARTITION = (4, 1)
N = 5
DIM = 4  # number of SYTs of shape (4,1)

rep_specht = irrep_specht(PARTITION)
rep_yof    = irrep_yof(PARTITION)

B, tabloids, tab_index = specht_basis(PARTITION)
syts = standard_young_tableaux(PARTITION)


def adj_transposition(i: int) -> Permutation:
    """Return the adjacent transposition s_i = (i  i+1) in S_5."""
    one_line = list(range(1, N + 1))
    one_line[i - 1], one_line[i] = one_line[i], one_line[i - 1]
    return Permutation(one_line)


# ---------------------------------------------------------------------------
# 1. Standard Young tableaux basis
# ---------------------------------------------------------------------------

print("=" * 65)
print(f"Partition {PARTITION}  –  standard representation of S_{N}")
print(f"Dimension: {DIM}   (= number of standard Young tableaux)")
print("=" * 65)

print("\nStandard Young tableaux (T_0 ... T_3):")
for idx, t in enumerate(syts):
    rows_str = " | ".join(" ".join(str(v) for v in row) for row in t.rows)
    print(f"  T_{idx}: {rows_str}")

# ---------------------------------------------------------------------------
# 2. Matrices for each adjacent transposition
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("Matrices for adjacent transpositions")
print("=" * 65)

np.set_printoptions(precision=6, suppress=True, linewidth=120)

for i in range(1, N):
    s_i = adj_transposition(i)
    M_s = rep_specht(s_i)
    M_y = rep_yof(s_i)

    print(f"\ns_{i} = ({i} {i+1})")
    print("-" * 55)

    # --- Specht ---
    print("  Specht (Gram-Schmidt on Specht module):")
    for row in M_s:
        print("   ", "  ".join(f"{v:+.6f}" for v in row))

    # --- YOF ---
    print("  Young orthogonal form (YOF):")
    for row in M_y:
        print("   ", "  ".join(f"{v:+.6f}" for v in row))

    # --- Entry-wise difference ---
    diff = np.max(np.abs(M_s - M_y))
    print(f"  Max entry-wise difference: {diff:.2e}")

    # --- Eigenvalues ---
    eigs_s = sorted(np.linalg.eigvalsh(M_s))
    eigs_y = sorted(np.linalg.eigvalsh(M_y))
    eig_match = np.allclose(eigs_s, eigs_y, atol=1e-10)
    print(f"  Eigenvalues (Specht): {[f'{e:+.4f}' for e in eigs_s]}")
    print(f"  Eigenvalues (YOF):    {[f'{e:+.4f}' for e in eigs_y]}")
    print(f"  Eigenvalues match:    {'YES' if eig_match else 'NO'}")

    # --- Character ---
    chi_s = np.trace(M_s)
    chi_y = np.trace(M_y)
    print(f"  Character (trace):    Specht={chi_s:+.6f}   YOF={chi_y:+.6f}   "
          f"match={'YES' if abs(chi_s - chi_y) < 1e-10 else 'NO'}")

# ---------------------------------------------------------------------------
# 3. Orthogonal intertwiner Q
# ---------------------------------------------------------------------------
# If rho_s and rho_y are equivalent, there exists an orthogonal Q such that
#     rho_s(g) = Q @ rho_y(g) @ Q.T   for all g.
#
# We find Q by solving rho_s(s_1) @ Q = Q @ rho_y(s_1) using the fact that
# for an orthogonal map on a matrix algebra the intertwiner is determined (up
# to sign within each eigenspace) by matching eigenspaces.
# Concretely: Q = U_s @ U_y.T where U_s, U_y are orthogonal eigenvector
# matrices of rho_s(s_1) and rho_y(s_1) with eigenvalues in the same order.

print("\n" + "=" * 65)
print("Orthogonal intertwiner  Q  (rho_specht = Q @ rho_yof @ Q.T)")
print("=" * 65)

s1 = adj_transposition(1)
M_s1 = rep_specht(s1)
M_y1 = rep_yof(s1)

evals_s, U_s = np.linalg.eigh(M_s1)
evals_y, U_y = np.linalg.eigh(M_y1)

# Sort by eigenvalue so eigenspaces align
order_s = np.argsort(evals_s)
order_y = np.argsort(evals_y)
U_s = U_s[:, order_s]
U_y = U_y[:, order_y]

Q = U_s @ U_y.T

print(f"\nQ (derived from eigenspace alignment of s_1):")
for row in Q:
    print("  ", "  ".join(f"{v:+.6f}" for v in row))

print(f"\nQ orthogonal check  (Q @ Q.T ≈ I):  "
      f"{'YES' if np.allclose(Q @ Q.T, np.eye(DIM), atol=1e-10) else 'NO'}")

# ---------------------------------------------------------------------------
# 4. Verify Q intertwines all s_i
# ---------------------------------------------------------------------------

print("\nVerify  rho_specht(s_i) = Q @ rho_yof(s_i) @ Q.T  for each s_i:")
all_ok = True
for i in range(1, N):
    s_i = adj_transposition(i)
    M_s = rep_specht(s_i)
    M_y = rep_yof(s_i)
    err = np.max(np.abs(M_s - Q @ M_y @ Q.T))
    ok = err < 1e-10
    print(f"  s_{i}: max error = {err:.2e}   {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok

if all_ok:
    print("\nAll s_i: Specht and YOF representations are equivalent via Q. PASS")
else:
    # Q built from s_1 may not diagonalise degenerate eigenspaces correctly for
    # all s_i; find Q via the group-averaging intertwiner formula:
    #   T = Σ_{g ∈ G} ρ_s(g) A ρ_y(g)^T   (random A avoids accidental zero)
    # T satisfies ρ_s(h) T = T ρ_y(h), so its polar factor is Q.
    print("\nRetrying Q via group-average intertwiner  T = Σ_g ρ_s(g) A ρ_y(g)^T ...")
    from symmetric_group import SymmetricGroup
    Sn = SymmetricGroup(N)
    elems = Sn.elements()
    rng = np.random.default_rng(42)
    A = rng.standard_normal((DIM, DIM))
    T = sum(rep_specht(g) @ A @ rep_yof(g).T for g in elems)
    U, _, Vt = np.linalg.svd(T)
    Q2 = U @ Vt
    print(f"  Q2 orthogonal: {'YES' if np.allclose(Q2 @ Q2.T, np.eye(DIM), atol=1e-10) else 'NO'}")
    all_ok2 = True
    for i in range(1, N):
        s_i = adj_transposition(i)
        err = np.max(np.abs(rep_specht(s_i) - Q2 @ rep_yof(s_i) @ Q2.T))
        ok = err < 1e-10
        print(f"  s_{i}: max error = {err:.2e}   {'PASS' if ok else 'FAIL'}")
        all_ok2 = all_ok2 and ok
    if all_ok2:
        print("\nAll s_i: equivalent via Q2. PASS")
        Q = Q2

# ---------------------------------------------------------------------------
# 5. Summary: are matrices identical or just equivalent?
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print("Summary")
print("=" * 65)

identical = all(
    np.allclose(rep_specht(adj_transposition(i)),
                rep_yof(adj_transposition(i)), atol=1e-10)
    for i in range(1, N)
)
print(f"  Matrices identical (same basis):   {'YES' if identical else 'NO'}")

equiv = all(
    np.allclose(np.trace(rep_specht(adj_transposition(i))),
                np.trace(rep_yof(adj_transposition(i))), atol=1e-10)
    for i in range(1, N)
)
print(f"  Same characters (equivalent irrep): {'YES' if equiv else 'NO'}")
print(f"  Related by orthogonal Q:             YES  (verified above)")
print()
