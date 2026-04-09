"""
Vibe coded (Claude)
Irreducible representations (irreps) of the symmetric group S_n.

The irreps of S_n are indexed by partitions of n (Young diagrams).
This module constructs them using Young's orthogonal form.

For a partition λ of n:
  - The representation space has one basis vector per standard Young tableau
    (SYT) of shape λ; its dimension equals the number of such tableaux.
  - Adjacent transpositions s_i = (i i+1) act by the rule:

        D(s_i)_{T,T}   =  1/d(T,i)
        D(s_i)_{T',T}  =  sqrt(1 - 1/d(T,i)^2)   if T' = swap(T, i, i+1) is standard

    where d(T,i) = (col(i+1) - col(i)) - (row(i+1) - row(i)) is the axial
    distance from i to i+1 in T (0-indexed rows/cols).   

  - Any permutation is factored into adjacent transpositions; the irrep matrix
    is the corresponding product of s_i matrices.

The result is an orthogonal representation over the reals.

Special cases recoverable from this module:
  - λ = (n,)      → trivial representation (dim 1)
  - λ = (1,...,1) → sign representation (dim 1)
  - λ = (n-1, 1)  → standard representation (dim n-1)

References:
  Sagan, "The Symmetric Group", Chapters 2-3.
  James & Kerber, "The Representation Theory of the Symmetric Group".
"""

import numpy as np
from functools import cache
from permutations import Permutation


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

def partitions_of(n: int) -> list[tuple[int, ...]]:
    """Return all partitions of n as tuples of parts in descending order."""
    result = []

    def _gen(remaining, max_part, current):
        if remaining == 0:
            result.append(tuple(current))
            return
        for p in range(min(remaining, max_part), 0, -1):
            _gen(remaining - p, p, current + [p])

    _gen(n, n, [])
    return result


# ---------------------------------------------------------------------------
# Standard Young tableaux
# ---------------------------------------------------------------------------

class YoungTableau:
    """
    A filling of a Young diagram of shape λ with integers 1,...,n.
    Rows and columns are 0-indexed internally.
    """

    def __init__(self, rows: list[list[int]]):
        self.rows = [list(r) for r in rows]
        self.shape = tuple(len(r) for r in self.rows)
        self._pos: dict[int, tuple[int, int]] = {}
        for r, row in enumerate(self.rows):
            for c, val in enumerate(row):
                self._pos[val] = (r, c)

    def position(self, val: int) -> tuple[int, int]:
        """Return (row, col) of val, 0-indexed."""
        return self._pos[val]

    def is_standard(self) -> bool:
        """Rows increase left-to-right; columns increase top-to-bottom."""
        for row in self.rows:
            if any(row[i] >= row[i + 1] for i in range(len(row) - 1)):
                return False
        n_cols = max(len(r) for r in self.rows) if self.rows else 0
        for c in range(n_cols):
            col = [self.rows[r][c] for r in range(len(self.rows)) if c < len(self.rows[r])]
            if any(col[i] >= col[i + 1] for i in range(len(col) - 1)):
                return False
        return True

    def swap(self, a: int, b: int) -> 'YoungTableau':
        """Return a new tableau with entries a and b swapped."""
        new_rows = [list(r) for r in self.rows]
        ra, ca = self._pos[a]
        rb, cb = self._pos[b]
        new_rows[ra][ca], new_rows[rb][cb] = b, a
        return YoungTableau(new_rows)

    def __eq__(self, other):
        return isinstance(other, YoungTableau) and self.rows == other.rows

    def __hash__(self):
        return hash(tuple(tuple(r) for r in self.rows))

    def __repr__(self):
        return '\n'.join(' '.join(f'{v:2}' for v in row) for row in self.rows)


def standard_young_tableaux(shape: tuple[int, ...]) -> list[YoungTableau]:
    """
    Generate all standard Young tableaux of the given shape by backtracking.

    Places 1, 2, ..., n one at a time in any cell whose left and top
    neighbours are already filled (so both row and column remain increasing).
    """
    n = sum(shape)
    num_rows = len(shape)
    result = []

    def backtrack(grid, row_counts, next_val):
        if next_val > n:
            result.append(YoungTableau(grid))
            return
        for r in range(num_rows):
            c = row_counts[r]
            if c >= shape[r]:
                continue                          # row is full
            if r > 0 and row_counts[r - 1] <= c:
                continue                          # cell above not yet filled
            grid[r].append(next_val)
            row_counts[r] += 1
            backtrack(grid, row_counts, next_val + 1)
            grid[r].pop()
            row_counts[r] -= 1

    backtrack([[] for _ in range(num_rows)], [0] * num_rows, 1)
    return result


# ---------------------------------------------------------------------------
# Young's orthogonal form
# ---------------------------------------------------------------------------

def _axial_distance(tab: YoungTableau, i: int, j: int) -> int:
    """
    Axial distance from entry i to entry j in tab:
        d = (col(j) - col(i)) - (row(j) - row(i))   [0-indexed]
    """
    ri, ci = tab.position(i)
    rj, cj = tab.position(j)
    return (cj - ci) - (rj - ri)


def _to_adjacent_transpositions(perm: Permutation) -> list[int]:
    """
    Express perm as s_{i_1} * ... * s_{i_k} (product of adjacent transpositions,
    where composition is (f*g)(x) = f(g(x))).

    Uses insertion sort on the one-line notation to right-multiply by successive
    s_j until the array is sorted; the resulting sequence is then reversed to
    recover the left-to-right factorisation of perm.
    """
    pi = perm.to_one_line()       # copy; pi[pos] = value (0-indexed pos)
    applied = []                  # right-multiplications s_j in order applied
    for target in range(1, perm.n + 1):
        pos = pi.index(target)    # 0-indexed position of value `target`
        for j in range(pos, target - 1, -1):   # bubble left
            pi[j], pi[j - 1] = pi[j - 1], pi[j]
            applied.append(j)                   # 1-indexed transposition s_j
    # perm * s_{applied[0]} * ... * s_{applied[-1]} = identity
    # => perm = s_{applied[-1]} * ... * s_{applied[0]}
    return list(reversed(applied))


@cache
def irrep(partition: tuple[int, ...]):
    """
    Return the irreducible representation of S_n indexed by `partition`.

    `partition` must be a tuple of positive integers in descending order
    that sum to n, e.g. (3, 1, 1) for n = 5.

    Returns a function  rep(perm: Permutation) -> np.ndarray  that maps each
    permutation to its matrix in Young's orthogonal form.  The matrix is
    dim x dim where dim = number of standard Young tableaux of shape partition.
    """
    tableaux = standard_young_tableaux(partition)
    dim = len(tableaux)
    tab_index = {t: idx for idx, t in enumerate(tableaux)}
    n = sum(partition)

    @cache
    def _s_matrix(i: int) -> np.ndarray:
        """Matrix of adjacent transposition s_i = (i i+1) in the SYT basis."""
        M = np.zeros((dim, dim))
        for col, T in enumerate(tableaux):
            d = _axial_distance(T, i, i + 1)
            M[col, col] = 1.0 / d
            T_prime = T.swap(i, i + 1)
            if T_prime.is_standard():
                row = tab_index[T_prime]
                M[row, col] = np.sqrt(1.0 - 1.0 / d ** 2)
        return M

    def rep(perm: Permutation) -> np.ndarray:
        """Matrix of perm in the irrep of S_n indexed by this partition."""
        if perm.n != n:
            raise ValueError(f"Permutation has degree {perm.n}, expected {n}")
        M = np.eye(dim)
        for i in _to_adjacent_transpositions(perm):
            S = _s_matrix(i)
            M = M @ S
        return M

    return rep


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from symmetric_group import SymmetricGroup

    for n in range(2, 7):
        Sn = SymmetricGroup(n)
        parts = partitions_of(n)
        print(f"\n=== S_{n}  (order {Sn.order()}) ===")
        print(f"partitions: {parts}")

        for lam in parts:
            rho = irrep(lam)
            elements = Sn.elements()
            dim = rho(elements[0]).shape[0]

            # Verify homomorphism on all pairs (small groups) or a sample
            pairs = [(elements[i], elements[j])
                     for i in range(min(8, len(elements)))
                     for j in range(min(8, len(elements)))]
            hom_ok = all(
                np.allclose(rho(s * t), rho(s) @ rho(t)) for s, t in pairs
            )
            
            # Character at identity = dimension
            chi_e = round(np.trace(rho(Sn.identity())).real)

            # Average over all group elements
            avg = sum(rho(g) for g in elements) / len(elements)

            print(f"  λ={lam}  dim={dim}  χ(e)={chi_e}  homomorphism={'✓' if hom_ok else '✗'}")
            if not np.allclose(avg, 0):
                print(f"    average over group:")
                print(np.array2string(avg, precision=4, suppress_small=True, prefix="    "))
