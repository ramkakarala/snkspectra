"""
vibe coded (Claude) 15 March 2026
    Convert a permutation to its corresponding binary permutation matrix.

    The permutation is 1-indexed: perm[i] = j means element i+1 maps to position j.
    The resulting matrix M has M[perm[i]-1,i] = 1 and 0 elsewhere.
    This matrix sends column vectors e_i = [0,..,1,..,0], with 1 in the ith place
    to e_perm(i), and hence forms a homomorphism with composition of perms, i.e,
    M(sigma*tau) = M(sigma)M(tau), where sigma, and tau are perms and 
    sigma*tau(x) = sigma(tau(x))
"""

import numpy as np
from permutations import Permutation


def perm_mat_rep(perm: Permutation) -> np.ndarray:
    """
    Args:
        perm: A Permutation instance on {1, 2, ..., n}

    Returns:
        An n×n binary numpy array representing the permutation.
    """
    n = perm.n
    matrix = np.zeros((n, n), dtype=int)
    for i in range(1, n + 1):
        matrix[perm(i) - 1, i - 1] = 1 # shuffle the rows according to the perm
    return matrix


if __name__ == "__main__":
    examples = [
        Permutation([1, 2, 3]),        # identity
        Permutation([3, 1, 2]),        # cyclic shift
        Permutation([2, 1, 3, 4]),     # swap first two
        Permutation([4, 3, 2, 1]),     # reverse
    ]

    for perm in examples:
        print(f"Permutation: {perm}")
        print(perm_mat_rep(perm))
        print()
