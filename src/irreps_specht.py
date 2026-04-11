"""
Irreducible representations of S_n via Gram-Schmidt orthogonalization of the
Specht module.

Construction
------------
For a partition λ of n:

1.  Permutation module M^λ
    Basis = all tabloids of shape λ, i.e. fillings of the Young diagram with
    {1,...,n} where rows are unordered (equivalence classes under row
    permutation).  Dimension = n! / (λ_1! · λ_2! · ... · λ_k!).

2.  Polytabloids
    For each standard Young tableau T define
        e_T = Σ_{π ∈ C_T} sgn(π) · {π·T}
    where C_T is the column stabilizer of T (all permutations that map each
    column of T to itself) and {·} denotes the tabloid of a tableau.

3.  Specht module S^λ
    The polytabloids {e_T : T standard} span a submodule of M^λ called the
    Specht module.  They are linearly independent (Sagan Thm 2.6.5) but not
    orthogonal in general.  Gram-Schmidt on the inner product that makes
    tabloids orthonormal yields an orthonormal basis B.

4.  Representation matrices
    For adjacent transposition s_i = (i i+1), the matrix in the orthonormal
    Specht basis is
        ρ(s_i) = B · P_i · B^T
    where P_i is the permutation matrix of s_i on the tabloid basis.
    General permutations are factored into adjacent transpositions.

The result is an orthogonal representation equivalent to Young's orthogonal
form (they are both orthonormal bases for the same irreducible module), related
by an orthogonal change of basis.

Reference: Sagan, "The Symmetric Group", Chapters 2-3.
"""

import numpy as np
from itertools import combinations, permutations as iter_perms
from irreps import standard_young_tableaux, partitions_of, _to_adjacent_transpositions
from permutations import Permutation


# ---------------------------------------------------------------------------
# Tabloids
# ---------------------------------------------------------------------------

def all_tabloids(shape: tuple[int, ...]) -> list[tuple[frozenset, ...]]:
    """
    All tabloids of the given shape as tuples of frozensets.

    A tabloid {T} of shape λ is an assignment of {1,...,n} to rows of sizes
    λ_1, λ_2, ..., λ_k where order within rows is ignored.
    """
    result = []

    def backtrack(remaining: set, row_idx: int, rows_so_far: tuple):
        if row_idx == len(shape):
            result.append(rows_so_far)
            return
        for combo in combinations(sorted(remaining), shape[row_idx]):
            backtrack(remaining - set(combo), row_idx + 1,
                      rows_so_far + (frozenset(combo),))

    backtrack(set(range(1, sum(shape) + 1)), 0, ())
    return result


def tableau_to_tabloid(rows: list[list[int]]) -> tuple[frozenset, ...]:
    """Convert a list-of-lists tableau to its tabloid (tuple of frozensets)."""
    return tuple(frozenset(row) for row in rows)


def apply_perm_to_tabloid(perm_dict: dict, tabloid: tuple[frozenset, ...]) -> tuple[frozenset, ...]:
    """Apply a permutation (as a dict i → σ(i)) to a tabloid."""
    return tuple(frozenset(perm_dict[x] for x in row) for row in tabloid)


# ---------------------------------------------------------------------------
# Column stabilizer
# ---------------------------------------------------------------------------

def _sign_of_list_permutation(original: list, permuted: tuple) -> int:
    """
    Sign of the permutation that maps original[k] → permuted[k].
    Computed by counting inversions in the induced index permutation.
    """
    pos = {v: i for i, v in enumerate(original)}
    p = [pos[v] for v in permuted]
    inversions = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return 1 if inversions % 2 == 0 else -1


def column_stabilizer(syt) -> list[tuple[int, dict]]:
    """
    Return [(sign, perm_dict), ...] for all elements of the column stabilizer
    of the given standard Young tableau.

    The column stabilizer C_T is the subgroup of S_n consisting of all
    permutations that map each column of T to itself (as a set).
    """
    n = sum(len(row) for row in syt.rows)

    # Group entries by column index
    cols: dict[int, list[int]] = {}
    for row in syt.rows:
        for c, val in enumerate(row):
            cols.setdefault(c, []).append(val)

    # Start with the identity
    result: list[tuple[int, dict]] = [(1, {i: i for i in range(1, n + 1)})]

    for col_entries in cols.values():
        if len(col_entries) == 1:
            continue  # trivial column, no new permutations
        new_result = []
        for (sgn, perm) in result:
            for p in iter_perms(col_entries):
                s = _sign_of_list_permutation(col_entries, p)
                new_perm = dict(perm)
                for orig, img in zip(col_entries, p):
                    new_perm[orig] = img
                new_result.append((sgn * s, new_perm))
        result = new_result

    return result


# ---------------------------------------------------------------------------
# Polytabloid
# ---------------------------------------------------------------------------

def polytabloid(syt, tab_index: dict) -> np.ndarray:
    """
    Compute the polytabloid e_T as a coordinate vector in the tabloid basis.

        e_T = Σ_{π ∈ C_T} sgn(π) · {π·T}
    """
    vec = np.zeros(len(tab_index))
    base_tabloid = tableau_to_tabloid(syt.rows)
    for sgn, perm_dict in column_stabilizer(syt):
        t = apply_perm_to_tabloid(perm_dict, base_tabloid)
        vec[tab_index[t]] += sgn
    return vec


# ---------------------------------------------------------------------------
# Specht basis via Gram-Schmidt
# ---------------------------------------------------------------------------

def specht_basis(partition: tuple[int, ...]) -> tuple[np.ndarray, list, dict]:
    """
    Return (B, tabloids, tab_index) where:
      - B has shape (dim, num_tabloids); rows are the orthonormal Specht basis
        vectors expressed in the tabloid basis.
      - tabloids is the ordered list of all tabloids.
      - tab_index maps each tabloid to its integer index.

    Steps:
      1. Compute polytabloids e_T for each standard Young tableau T.
      2. Gram-Schmidt orthogonalize them with respect to the inner product
         that makes tabloids orthonormal.
    """
    syts = standard_young_tableaux(partition)
    tabloids = all_tabloids(partition)
    tab_index = {t: i for i, t in enumerate(tabloids)}

    # Compute polytabloids as rows
    polys = np.array([polytabloid(syt, tab_index) for syt in syts])

    # Gram-Schmidt
    basis = []
    for v in polys:
        for b in basis:
            v = v - np.dot(b, v) * b
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            basis.append(v / norm)

    B = np.array(basis)  # shape (dim, num_tabloids)
    return B, tabloids, tab_index


# ---------------------------------------------------------------------------
# Transposition action on tabloids
# ---------------------------------------------------------------------------

def transposition_matrix_on_tabloids(n: int, i: int,
                                     tabloids: list,
                                     tab_index: dict) -> np.ndarray:
    """
    Permutation matrix of s_i = (i i+1) acting on the tabloid basis.
    P[new_idx, old_idx] = 1 iff s_i · tabloids[old_idx] = tabloids[new_idx].
    """
    m = len(tabloids)
    P = np.zeros((m, m))
    swap = {k: k for k in range(1, n + 1)}
    swap[i], swap[i + 1] = i + 1, i
    for j, t in enumerate(tabloids):
        new_t = apply_perm_to_tabloid(swap, t)
        P[tab_index[new_t], j] = 1.0
    return P


# ---------------------------------------------------------------------------
# Irrep
# ---------------------------------------------------------------------------

def irrep_specht(partition: tuple[int, ...]):
    """
    Return a function rep(perm: Permutation) -> np.ndarray for the irrep of
    S_n indexed by partition, built via Gram-Schmidt on the Specht module.

    The matrix of s_i in the orthonormal Specht basis is:
        ρ(s_i) = B · P_i · B^T
    where B is the orthonormal basis matrix (rows) and P_i is the permutation
    matrix of s_i on tabloids.  Since B is an isometry (B B^T = I_dim), this
    is the standard change-of-basis formula.

    This representation is orthogonal and equivalent to Young's orthogonal
    form, related by an orthogonal change of basis (the overlap matrix B · B_YOF^T).
    """
    n = sum(partition)
    B, tabloids, tab_index = specht_basis(partition)

    # Precompute s_i matrices
    s_mats = {
        i: B @ transposition_matrix_on_tabloids(n, i, tabloids, tab_index) @ B.T
        for i in range(1, n)
    }

    def rep(perm: Permutation) -> np.ndarray:
        if perm.n != n:
            raise ValueError(f"Permutation has degree {perm.n}, expected {n}")
        M = np.eye(B.shape[0])
        for i in _to_adjacent_transpositions(perm):
            M = M @ s_mats[i]
        return M

    return rep


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from symmetric_group import SymmetricGroup
    from irreps import irrep as irrep_yof

    n = 5
    partition = (3, 2)

    print(f"Irrep of S_{n} for partition {partition} via Specht module")
    print("=" * 60)

    B, tabloids, tab_index = specht_basis(partition)
    syts = standard_young_tableaux(partition)
    print(f"\nPermutation module M^{partition}: {len(tabloids)} tabloids")
    print(f"Specht module S^{partition}: dimension {B.shape[0]}")

    print("\nTableaux (SYT basis, before Gram-Schmidt):")
    for idx, syt in enumerate(syts):
        tab = tableau_to_tabloid(syt.rows)
        poly = polytabloid(syt, tab_index)
        nonzero = {tabloids[j]: int(v) for j, v in enumerate(poly) if abs(v) > 1e-10}
        print(f"  T_{idx}: e_T = {nonzero}")

    print("\nOrthonormal Specht basis vectors (in tabloid basis):")
    for idx, row in enumerate(B):
        nonzero = {str(tuple(sorted(tabloids[j]))): f"{v:+.4f}"
                   for j, v in enumerate(row) if abs(v) > 1e-10}
        print(f"  b_{idx}: {nonzero}")

    # --- S matrices ---
    rep = irrep_specht(partition)

    def s(i):
        one_line = list(range(1, n + 1))
        one_line[i - 1], one_line[i] = one_line[i], one_line[i - 1]
        return Permutation(one_line)

    print(f"\nS matrices for adjacent transpositions:")
    for i in range(1, n):
        M = rep(s(i))
        print(f"\n  S_{i} = rho(({i} {i+1})):")
        print(np.array2string(M, precision=6, suppress_small=True, prefix="  "))

    # --- Verify homomorphism ---
    print("\n" + "=" * 60)
    print("Homomorphism check")
    Sn = SymmetricGroup(n)
    elements = Sn.elements()
    sample = [(elements[i], elements[j])
              for i in range(min(12, len(elements)))
              for j in range(min(12, len(elements)))]
    hom_ok = all(np.allclose(rep(a * b), rep(a) @ rep(b)) for a, b in sample)
    print(f"  rho(a*b) = rho(a) @ rho(b) for {len(sample)} pairs: {'PASS' if hom_ok else 'FAIL'}")

    # --- Verify braid relations ---
    print("\nBraid relations  s_i s_{i+1} s_i = s_{i+1} s_i s_{i+1}:")
    for i in range(1, n - 1):
        Si, Si1 = rep(s(i)), rep(s(i + 1))
        lhs = Si @ Si1 @ Si
        rhs = Si1 @ Si @ Si1
        err = np.max(np.abs(lhs - rhs))
        print(f"  i={i}: max error = {err:.2e}  {'PASS' if err < 1e-10 else 'FAIL'}")

    # --- Compare with Young's orthogonal form ---
    print("\n" + "=" * 60)
    print("Comparison with Young's orthogonal form")
    print("(same irrep up to orthogonal change of basis)")
    rep_yof = irrep_yof(partition)
    for i in range(1, n):
        M_specht = rep(s(i))
        M_yof = rep_yof(s(i))
        # Eigenvalues must match (both are orthogonal, same characteristic poly)
        eigs_s = sorted(np.linalg.eigvalsh(M_specht))
        eigs_y = sorted(np.linalg.eigvalsh(M_yof))
        match = np.allclose(eigs_s, eigs_y)
        print(f"  S_{i}: eigenvalues match YOF = {'yes' if match else 'NO'}")
        print(f"    Specht:  {[f'{e:+.4f}' for e in eigs_s]}")
        print(f"    YOF:     {[f'{e:+.4f}' for e in eigs_y]}")
