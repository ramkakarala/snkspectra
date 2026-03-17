"""
Vibe coded (Claude)
Symmetric group S_n: the group of all permutations on {1, 2, ..., n}.
"""

from itertools import permutations as _permutations
from permutations import Permutation


class SymmetricGroup:
    def __init__(self, n):
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self._elements = [
            Permutation(list(p)) for p in _permutations(range(1, n + 1))
        ]

    # ------------------------------------------------------------------
    # Group axioms
    # ------------------------------------------------------------------

    def identity(self):
        """Return the identity element of S_n."""
        return Permutation.identity(self.n)

    def op(self, sigma, tau):
        """Group operation: composition sigma * tau."""
        return sigma * tau

    def inv(self, sigma):
        """Return the inverse of sigma."""
        return sigma.inverse()

    # ------------------------------------------------------------------
    # Group properties
    # ------------------------------------------------------------------

    def order(self):
        """Return |S_n| = n!"""
        return len(self._elements)

    def elements(self):
        """Return all elements of S_n as a list of Permutations."""
        return list(self._elements)

    def __iter__(self):
        return iter(self._elements)

    def __len__(self):
        return len(self._elements)

    def __contains__(self, sigma):
        return isinstance(sigma, Permutation) and sigma.n == self.n

    def __repr__(self):
        return f"SymmetricGroup(n={self.n})"

    # ------------------------------------------------------------------
    # Subsets
    # ------------------------------------------------------------------

    def alternating(self):
        """Return the alternating group A_n: all even permutations in S_n."""
        return [sigma for sigma in self._elements if sigma.sign() == 1]

    def conjugacy_class(self, sigma):
        """
        Return the conjugacy class of sigma: { tau * sigma * tau^{-1} | tau in S_n }.
        Elements are in the same conjugacy class iff they have the same cycle type.
        """
        seen = set()
        result = []
        for tau in self._elements:
            conj = tau * sigma * tau.inverse()
            if conj not in seen:
                seen.add(conj)
                result.append(conj)
        return result

    def cycle_index(self):
        """
        Return the cycle index as a dict mapping cycle-type tuples to their counts.
        The cycle type of sigma is a sorted tuple of its cycle lengths (including fixed points).
        """
        index = {}
        for sigma in self._elements:
            cycle_lengths = sorted(len(c) for c in sigma.cycles())
            fixed = self.n - sum(cycle_lengths)
            cycle_type = tuple(sorted(cycle_lengths + [1] * fixed))
            index[cycle_type] = index.get(cycle_type, 0) + 1
        return index


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    for n in range(1, 5):
        Sn = SymmetricGroup(n)
        An = Sn.alternating()
        print(f"S_{n}: order={Sn.order()}, |A_{n}|={len(An)}")

    print()
    S3 = SymmetricGroup(3)
    print("Elements of S_3:")
    for sigma in S3:
        print(f"  {sigma.cycle_str() or 'e':12s}  sign={sigma.sign():+d}")

    print()
    print("Cycle index of S_3:")
    for cycle_type, count in sorted(S3.cycle_index().items()):
        print(f"  {cycle_type}: {count}")

    print()
    sigma = Permutation.from_cycles(3, [1, 2])
    print(f"Conjugacy class of (1 2) in S_3: {[p.cycle_str() for p in S3.conjugacy_class(sigma)]}")
