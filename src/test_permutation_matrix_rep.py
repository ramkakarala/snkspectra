"""
Vibe coded (Claude) 15 March 2026
Tests that the standard (permutation matrix) representation is a homomorphism:
    M(sigma * tau) == M(sigma) @ M(tau)
"""

import unittest
import numpy as np
from permutations import Permutation
from permutation_matrix_rep import perm_mat_rep
import random


class TestHomomorphism(unittest.TestCase):

    def _assert_hom(self, sigma, tau):
        lhs = perm_mat_rep(sigma * tau)
        rhs = perm_mat_rep(sigma) @ perm_mat_rep(tau)
        self.assertTrue(np.array_equal(lhs, rhs),
                        f"M({sigma} * {tau}) != M({sigma}) @ M({tau})")

    def test_two_3cycles(self):
        sigma = Permutation([2, 3, 1])
        tau   = Permutation([3, 1, 2])
        self._assert_hom(sigma, tau)

    def test_transposition_and_cycle(self):
        sigma = Permutation([2, 1, 3, 4, 5])
        tau   = Permutation([2, 3, 4, 5, 1])
        self._assert_hom(sigma, tau)

    def test_identity_is_neutral(self):
        sigma = Permutation([3, 1, 4, 2])
        e     = Permutation.identity(4)
        self._assert_hom(sigma, e)
        self._assert_hom(e, sigma)

    def test_sigma_and_its_inverse(self):
        sigma = Permutation([4, 2, 5, 1, 3])
        self._assert_hom(sigma, sigma.inverse())

    def test_identity_matrix_for_identity_permutation(self):
        e = Permutation.identity(4)
        self.assertTrue(np.array_equal(perm_mat_rep(e), np.eye(4, dtype=int)))

    def test_from_cycle(self):
        sigma = Permutation.from_cycles(6, [1, 3, 5])
        tau   = Permutation.from_cycles(6, [2, 4, 6])
        self._assert_hom(sigma, tau)

    def test_transpose_is_inverse(self):
        domain=list(range(1,7))
        rng = random.Random(42)
        I = np.eye(6,6)
        for _ in range(100):
            shuffled = domain[:]
            rng.shuffle(shuffled)
            p = Permutation(shuffled)
            M = perm_mat_rep(p)
            self.assertTrue(np.array_equal(I,M@M.T))


    def test_matrix_is_unitary(self):
        # A matrix M is unitary iff M @ M†  == I  (for real matrices, M† = Mᵀ)
        domain = list(range(1, 7))
        rng = random.Random(99)
        I = np.eye(6)
        for _ in range(100):
            shuffled = domain[:]
            rng.shuffle(shuffled)
            M = perm_mat_rep(Permutation(shuffled))
            self.assertTrue(np.allclose(M @ M.conj().T, I),
                            "M @ M† != I  (not unitary)")
            self.assertTrue(np.allclose(M.conj().T @ M, I),
                            "M† @ M != I  (not unitary)")

    def test_composition_is_associative_in_matrix_form(self):
        # M(sigma * (tau * rho)) == M(sigma) @ M(tau) @ M(rho)
        sigma = Permutation([2, 3, 1, 4])
        tau   = Permutation([1, 3, 4, 2])
        rho   = Permutation([4, 1, 2, 3])
        lhs = perm_mat_rep(sigma * tau * rho)
        rhs = perm_mat_rep(sigma) @ perm_mat_rep(tau) @ perm_mat_rep(rho)
        self.assertTrue(np.array_equal(lhs, rhs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
