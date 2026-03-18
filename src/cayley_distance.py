"""
Vibe coded (Claude)
Cayley distance on the symmetric group S_n.

The Cayley distance between two permutations sigma and tau is the minimum
number of transpositions needed to transform sigma into tau:

    d(sigma, tau) = n - c(sigma^{-1} * tau)

where c(pi) is the number of cycles of pi, including fixed points (1-cycles).
"""

from permutations import Permutation


def cayley_distance(sigma: Permutation, tau: Permutation) -> int:
    """
    Return the Cayley distance between sigma and tau.

    This equals n minus the number of cycles (including fixed points) of
    sigma^{-1} * tau.
    """
    if sigma.n != tau.n:
        raise ValueError(
            f"Permutations must have the same degree ({sigma.n} vs {tau.n})"
        )
    pi = sigma.inverse() * tau
    n_cycles = len(pi.cycles()) + (pi.n - sum(len(c) for c in pi.cycles()))
    return pi.n - n_cycles


def hamming_distance(sigma: Permutation, tau: Permutation) -> int:
    """
    Return the Hamming distance between sigma and tau: the number of positions
    where they disagree.

        d_H(sigma, tau) = #{j : sigma(j) != tau(j)}

    This equals (1/2) * ||M(sigma) - M(tau)||_F^2, where M(·) is the
    permutation matrix representation.
    """
    if sigma.n != tau.n:
        raise ValueError(
            f"Permutations must have the same degree ({sigma.n} vs {tau.n})"
        )
    return sum(1 for j in range(1, sigma.n + 1) if sigma(j) != tau(j))


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    e     = Permutation.identity(5)
    sigma = Permutation([2, 3, 1, 4, 5])   # (1 2 3)
    tau   = Permutation([2, 1, 3, 4, 5])   # (1 2)

    print(f"d(e,     e)     = {cayley_distance(e, e)}")
    print(f"d(e,     sigma) = {cayley_distance(e, sigma)}")
    print(f"d(e,     tau)   = {cayley_distance(e, tau)}")
    print(f"d(sigma, tau)   = {cayley_distance(sigma, tau)}")
    print(f"d(sigma, sigma) = {cayley_distance(sigma, sigma)}")

    # Symmetry check
    print(f"\nd(sigma, tau) == d(tau, sigma): "
          f"{cayley_distance(sigma, tau) == cayley_distance(tau, sigma)}")

    # Triangle inequality check
    rho = Permutation([1, 2, 4, 5, 3])   # (3 4 5)
    dst = cayley_distance(sigma, tau)
    dtr = cayley_distance(tau, rho)
    dsr = cayley_distance(sigma, rho)
    print(f"\nTriangle inequality d(s,t)+d(t,r) >= d(s,r): "
          f"{dst} + {dtr} >= {dsr} → {dst + dtr >= dsr}")

    # Hamming distance demo
    print(f"\n--- Hamming distance ---")
    print(f"d_H(e,     e)     = {hamming_distance(e, e)}")
    print(f"d_H(e,     sigma) = {hamming_distance(e, sigma)}")
    print(f"d_H(e,     tau)   = {hamming_distance(e, tau)}")
    print(f"d_H(sigma, tau)   = {hamming_distance(sigma, tau)}")
    print(f"d_H(sigma, sigma) = {hamming_distance(sigma, sigma)}")
