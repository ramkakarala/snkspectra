"""
Vibe coded (Claude) 15 March 2026
Tests for representations.py:
  - perm_mat_rep: standard permutation matrix representation
  - trivial_rep:  trivial (principal) representation
  - sign_rep:     sign (alternating) representation
  - character:    character (trace) of a representation
"""

import unittest
import numpy as np
from permutations import Permutation
from representations import perm_mat_rep, trivial_rep, sign_rep, standard_rep, character
from symmetric_group import SymmetricGroup
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


class TestTrivialRep(unittest.TestCase):

    def test_always_returns_one(self):
        for perm in [
            Permutation.identity(4),
            Permutation([2, 3, 1]),
            Permutation([2, 1, 3, 4, 5]),
            Permutation([4, 3, 2, 1]),
        ]:
            self.assertTrue(np.array_equal(trivial_rep(perm), np.array([[1]])))

    def test_is_homomorphism(self):
        # trivial_rep(sigma * tau) == trivial_rep(sigma) @ trivial_rep(tau)
        sigma = Permutation([2, 3, 1, 4])
        tau   = Permutation([1, 3, 4, 2])
        lhs = trivial_rep(sigma * tau)
        rhs = trivial_rep(sigma) @ trivial_rep(tau)
        self.assertTrue(np.array_equal(lhs, rhs))


class TestSignRep(unittest.TestCase):

    def test_identity_is_plus_one(self):
        e = Permutation.identity(5)
        self.assertTrue(np.array_equal(sign_rep(e), np.array([[1]])))

    def test_transposition_is_minus_one(self):
        t = Permutation([2, 1, 3])   # single transposition → odd
        self.assertTrue(np.array_equal(sign_rep(t), np.array([[-1]])))

    def test_3cycle_is_plus_one(self):
        c = Permutation([2, 3, 1])   # 3-cycle → even
        self.assertTrue(np.array_equal(sign_rep(c), np.array([[1]])))

    def test_is_homomorphism(self):
        # sign_rep(sigma * tau) == sign_rep(sigma) @ sign_rep(tau)
        domain = list(range(1, 6))
        rng = random.Random(7)
        for _ in range(100):
            a = domain[:]
            b = domain[:]
            rng.shuffle(a)
            rng.shuffle(b)
            sigma = Permutation(a)
            tau   = Permutation(b)
            lhs = sign_rep(sigma * tau)
            rhs = sign_rep(sigma) @ sign_rep(tau)
            self.assertTrue(np.array_equal(lhs, rhs))


class TestCharacter(unittest.TestCase):

    def test_trivial_character_is_always_1(self):
        for perm in [
            Permutation.identity(4),
            Permutation([2, 3, 1]),
            Permutation([4, 3, 2, 1]),
        ]:
            self.assertEqual(character(perm, trivial_rep), 1)

    def test_sign_character_matches_sign(self):
        perms = [
            Permutation([2, 1, 3]),          # odd  → -1
            Permutation([2, 3, 1]),          # even → +1
            Permutation([4, 3, 2, 1]),       # odd  → -1  (two transpositions... wait)
            Permutation.identity(5),         # even → +1
        ]
        for perm in perms:
            self.assertEqual(character(perm, sign_rep), perm.sign())

    def test_perm_mat_character_counts_fixed_points(self):
        # Fixed points of sigma are positions i where sigma(i) == i
        cases = [
            (Permutation([1, 2, 3, 4]),   4),  # identity: 4 fixed points
            (Permutation([2, 1, 3, 4]),   2),  # swap 1,2: 2 fixed points
            (Permutation([2, 3, 1, 4]),   1),  # 3-cycle + fixed 4
            (Permutation([2, 3, 4, 1]),   0),  # 4-cycle: no fixed points
        ]
        for perm, expected in cases:
            self.assertEqual(character(perm, perm_mat_rep), expected)


class TestStandardRep(unittest.TestCase):

    def test_dimension(self):
        # standard_rep on S_n gives an (n-1)×(n-1) matrix
        for n in [2, 3, 4, 5]:
            perm = Permutation.identity(n)
            self.assertEqual(standard_rep(perm).shape, (n - 1, n - 1))

    def test_identity_maps_to_identity_matrix(self):
        for n in [2, 3, 4, 5]:
            e = Permutation.identity(n)
            self.assertTrue(np.allclose(standard_rep(e), np.eye(n - 1)))

    def test_is_homomorphism(self):
        # standard_rep(sigma * tau) == standard_rep(sigma) @ standard_rep(tau)
        domain = list(range(1, 6))
        rng = random.Random(17)
        for _ in range(100):
            a, b = domain[:], domain[:]
            rng.shuffle(a)
            rng.shuffle(b)
            sigma, tau = Permutation(a), Permutation(b)
            lhs = standard_rep(sigma * tau)
            rhs = standard_rep(sigma) @ standard_rep(tau)
            self.assertTrue(np.allclose(lhs, rhs))

    def test_matrices_are_orthogonal(self):
        # standard_rep(sigma) is an orthogonal matrix: M @ M.T == I
        domain = list(range(1, 6))
        rng = random.Random(23)
        I = np.eye(4)
        for _ in range(100):
            shuffled = domain[:]
            rng.shuffle(shuffled)
            M = standard_rep(Permutation(shuffled))
            self.assertTrue(np.allclose(M @ M.T, I))

    def test_character_is_fixed_points_minus_one(self):
        # chi_standard = chi_perm - chi_trivial = (fixed points) - 1
        cases = [
            Permutation([1, 2, 3, 4, 5]),   # identity:   5 fixed points → 4
            Permutation([2, 1, 3, 4, 5]),   # (1 2):      3 fixed points → 2
            Permutation([2, 3, 1, 4, 5]),   # (1 2 3):    2 fixed points → 1
            Permutation([2, 3, 4, 5, 1]),   # (1 2 3 4 5): 0 fixed points → -1
        ]
        for perm in cases:
            expected = character(perm, perm_mat_rep) - character(perm, trivial_rep)
            self.assertAlmostEqual(character(perm, standard_rep), expected)

    def test_orthogonal_to_trivial_over_S5(self):
        # standard is irreducible and distinct from trivial → inner product = 0
        S5 = SymmetricGroup(5)
        inner = sum(
            character(g, standard_rep) * character(g, trivial_rep) for g in S5
        ) / S5.order()
        self.assertAlmostEqual(inner, 0)

    def test_orthogonal_to_sign_over_S5(self):
        # standard is irreducible and distinct from sign → inner product = 0
        S5 = SymmetricGroup(5)
        inner = sum(
            character(g, standard_rep) * character(g, sign_rep) for g in S5
        ) / S5.order()
        self.assertAlmostEqual(inner, 0)

    def test_self_inner_product_is_one_over_S5(self):
        # <chi_standard, chi_standard> = 1 iff the representation is irreducible
        S5 = SymmetricGroup(5)
        inner = sum(
            character(g, standard_rep) ** 2 for g in S5
        ) / S5.order()
        self.assertAlmostEqual(inner, 1)


class TestCharacterOrthogonality(unittest.TestCase):

    @staticmethod
    def _inner_product(rep1, rep2, group):
        # <chi_rho, chi_rho'> = (1/|G|) * sum_{g in G} chi_rho(g) * chi_rho'(g)
        return sum(
            character(g, rep1) * character(g, rep2) for g in group
        ) / group.order()

    def test_trivial_and_sign_are_orthogonal_over_S5(self):
        S5 = SymmetricGroup(5)
        self.assertEqual(self._inner_product(trivial_rep, sign_rep, S5), 0)

    def test_sign_and_perm_mat_are_orthogonal_over_S5(self):
        # perm_mat_rep decomposes as trivial + standard; sign is irreducible and
        # distinct from both components, so <chi_sign, chi_perm> = 0.
        S5 = SymmetricGroup(5)
        self.assertEqual(self._inner_product(sign_rep, perm_mat_rep, S5), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
