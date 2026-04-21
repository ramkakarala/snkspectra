#!/usr/bin/env python3
"""Character table of the symmetric group S_n via the Murnaghan-Nakayama rule."""

from functools import lru_cache
import numpy as np


def partitions(n):
    """All partitions of n as tuples, in decreasing lex order."""
    result = []

    def gen(remaining, max_val, curr):
        if remaining == 0:
            result.append(tuple(curr))
            return
        for v in range(min(remaining, max_val), 0, -1):
            gen(remaining - v, v, curr + [v])

    gen(n, n, [])
    return result


@lru_cache(maxsize=None)
def border_strips(lam, r):
    """
    All (mu, height) pairs from removing a border strip of size r from lam.

    A border strip (rim hook) is a connected skew shape with no 2x2 block.
    For consecutive strip rows k and k+1, the connectivity + no-2x2 condition
    forces mu[k] = lam[k+1] - 1.  The bottom row length is then determined by
    the total size constraint.

    height = (rows spanned) - 1; the MN sign is (-1)^height.
    """
    if r == 0:
        return ((lam, 0),)

    n = len(lam)

    def ext(k):
        return lam[k] if k < n else 0

    results = []
    for top in range(n):
        for bot in range(top, n):
            # Size equation: sum_{k=top}^{bot-1}(lam[k] - lam[k+1] + 1) + (lam[bot] - mu_bot) = r
            mu_bot = ext(top) + (bot - top) - r

            # mu_bot must be non-negative, strictly less than lam[bot] (nonempty strip
            # in the bottom row), and at least lam[bot+1] (partition stays valid).
            if mu_bot < 0 or mu_bot >= ext(bot) or mu_bot < ext(bot + 1):
                continue

            mu = list(lam)
            for k in range(top, bot):
                mu[k] = ext(k + 1) - 1   # forced by connectivity + no-2x2
            mu[bot] = mu_bot

            while mu and mu[-1] == 0:
                mu.pop()

            results.append((tuple(mu), bot - top))

    return tuple(results)


@lru_cache(maxsize=None)
def chi(lam, mu):
    """chi^lam(mu): character of irrep lam evaluated on conjugacy class mu."""
    if not mu:
        return 1 if not lam else 0
    r, rest = mu[0], mu[1:]
    return sum((-1) ** h * chi(new_lam, rest) for new_lam, h in border_strips(lam, r))


def character_table(n):
    """Return character table of S_n as an integer numpy matrix.

    Rows = irreps, columns = conjugacy classes (both indexed by partitions of n
    in decreasing lex order, columns reversed).
    """
    parts = partitions(n)
    X = np.array([[chi(lam, mu) for mu in reversed(parts)] for lam in parts], dtype=int)
    return X

if __name__ == "__main__":
    for n in range(4, 9):
        print(f"S_{n}:")
        X = character_table(n)
        print(X)
        print()
        D = X.T @ X
        off = D - np.diag(np.diag(D))
        if np.all(off == 0):
            print(f"D = X.T @ X is diagonal: {np.diag(D).tolist()}")
        else:
            nz = np.argwhere(off != 0)
            print(f"D = X.T @ X is NOT diagonal. Nonzero off-diagonal elements:")
            for i, j in nz:
                print(f"  D[{i},{j}] = {D[i,j]}")
        print()

