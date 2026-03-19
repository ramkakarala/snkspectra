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
from functools import cache
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


def trivial_rep(perm: Permutation) -> np.ndarray:
    """
    The trivial (principal) representation: every permutation maps to the
    1×1 identity matrix [[1]].
    """
    return np.array([[1]], dtype=int)


def sign_rep(perm: Permutation) -> np.ndarray:
    """
    The sign (alternating) representation: even permutations map to [[1]],
    odd permutations map to [[-1]].
    """
    return np.array([[perm.sign()]], dtype=int)


@cache
def _standard_basis(n: int) -> np.ndarray:
    """
    Return an n×(n-1) matrix whose columns are an orthonormal basis for the
    hyperplane {x in R^n : sum(x) = 0}, i.e. the orthogonal complement of
    (1, 1, ..., 1)/sqrt(n).

    Uses the explicit basis vectors:
        b_k = 1/sqrt(k(k+1)) * (1, 1, ..., 1, -k, 0, ..., 0),  k = 1, ..., n-1
    where the first k entries are 1 and the (k+1)-th entry is -k.
    """
    B = np.zeros((n, n - 1))
    for k in range(1, n):
        B[:k, k - 1] = 1.0
        B[k, k - 1] = -k
        B[:, k - 1] /= np.sqrt(k * (k + 1))
    return B


def standard_rep(perm: Permutation) -> np.ndarray:
    """
    The standard representation of S_n: the irreducible (n-1)-dimensional
    subrepresentation of the permutation representation.

    It acts on the invariant hyperplane V = {x in R^n : sum(x_i) = 0},
    which is the orthogonal complement of the trivial subspace span{(1,...,1)}.

    The matrix is computed by conjugating the permutation matrix with an
    orthonormal basis B for V:

        std_rep(sigma) = B^T @ M(sigma) @ B

    Its character satisfies chi_std = chi_perm - chi_trivial,
    i.e. (fixed points of sigma) - 1.
    """
    B = _standard_basis(perm.n)
    M = perm_mat_rep(perm).astype(float)
    return B.T @ M @ B


def character(perm: Permutation, rep) -> int | float:
    """
    Return the character of a representation evaluated at perm.

    The character of a representation rho at a group element g is the trace
    of the matrix rho(g).

    Args:
        perm: A Permutation instance.
        rep:  A callable perm -> np.ndarray (e.g. perm_mat_rep, trivial_rep,
              sign_rep).

    Returns:
        Trace of rep(perm). For the standard representations:
          - trivial_rep:  always 1
          - sign_rep:     +1 (even) or -1 (odd)
          - perm_mat_rep: number of fixed points of perm
          - standard_rep: (fixed points of perm) - 1
    """
    return np.trace(rep(perm))


if __name__ == "__main__":
    examples = [
        Permutation([1, 2, 3]),        # identity
        Permutation([3, 1, 2]),        # cyclic shift
        Permutation([2, 1, 3, 4]),     # swap first two
        Permutation([4, 3, 2, 1]),     # reverse
    ]

    for perm in examples:
        print(f"Permutation: {perm}")
        print(f"  perm_mat_rep:\n{perm_mat_rep(perm)}")
        print(f"  trivial_rep:   {trivial_rep(perm)}")
        print(f"  sign_rep:      {sign_rep(perm)}")
        print(f"  chi_trivial:   {character(perm, trivial_rep)}")
        print(f"  chi_sign:      {character(perm, sign_rep)}")
        print(f"  chi_perm:      {character(perm, perm_mat_rep)}  (fixed points)")
        print(f"  standard_rep:\n{standard_rep(perm)}")
        print(f"  chi_standard:  {character(perm, standard_rep):.4g}  (fixed points - 1)")
        print()
