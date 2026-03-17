"""
Vibe coded (Claude)
Unit tests for SymmetricGroup.
"""

import unittest
from math import factorial
from permutations import Permutation
from symmetric_group import SymmetricGroup


class TestConstruction(unittest.TestCase):

    def test_order_equals_n_factorial(self):
        for n in range(1, 7):
            self.assertEqual(SymmetricGroup(n).order(), factorial(n))

    def test_invalid_n_raises(self):
        with self.assertRaises(ValueError):
            SymmetricGroup(0)

    def test_elements_are_permutations(self):
        for sigma in SymmetricGroup(4):
            self.assertIsInstance(sigma, Permutation)

    def test_elements_have_correct_degree(self):
        for sigma in SymmetricGroup(4):
            self.assertEqual(sigma.n, 4)

    def test_no_duplicate_elements(self):
        elements = SymmetricGroup(4).elements()
        self.assertEqual(len(elements), len(set(elements)))

    def test_contains_permutation_of_correct_degree(self):
        S4 = SymmetricGroup(4)
        self.assertIn(Permutation([2, 1, 3, 4]), S4)

    def test_does_not_contain_permutation_of_wrong_degree(self):
        S4 = SymmetricGroup(4)
        self.assertNotIn(Permutation([2, 1, 3]), S4)

    def test_len_equals_order(self):
        S4 = SymmetricGroup(4)
        self.assertEqual(len(S4), S4.order())

    def test_repr(self):
        self.assertEqual(repr(SymmetricGroup(3)), "SymmetricGroup(n=3)")


class TestGroupAxioms(unittest.TestCase):
    """Verify that S_n satisfies the four group axioms."""

    def setUp(self):
        self.S3 = SymmetricGroup(3)

    def test_identity_is_in_group(self):
        self.assertIn(self.S3.identity(), self.S3)

    def test_identity_left_and_right(self):
        e = self.S3.identity()
        for sigma in self.S3:
            self.assertEqual(self.S3.op(e, sigma), sigma)
            self.assertEqual(self.S3.op(sigma, e), sigma)

    def test_closure(self):
        # Product of any two elements is still in S_n
        S3 = self.S3
        for sigma in S3:
            for tau in S3:
                self.assertIn(S3.op(sigma, tau), S3)

    def test_inverses(self):
        e = self.S3.identity()
        for sigma in self.S3:
            self.assertEqual(self.S3.op(sigma, self.S3.inv(sigma)), e)
            self.assertEqual(self.S3.op(self.S3.inv(sigma), sigma), e)

    def test_associativity(self):
        elements = self.S3.elements()[:3]   # sample to keep it fast
        op = self.S3.op
        for a in elements:
            for b in elements:
                for c in elements:
                    self.assertEqual(op(op(a, b), c), op(a, op(b, c)))


class TestAlternating(unittest.TestCase):

    def test_alternating_order_is_half(self):
        for n in range(2, 7):
            Sn = SymmetricGroup(n)
            self.assertEqual(len(Sn.alternating()), factorial(n) // 2)

    def test_alternating_contains_only_even_permutations(self):
        for sigma in SymmetricGroup(4).alternating():
            self.assertEqual(sigma.sign(), +1)

    def test_alternating_closed_under_multiplication(self):
        An = SymmetricGroup(4).alternating()
        for sigma in An:
            for tau in An:
                self.assertEqual((sigma * tau).sign(), +1)

    def test_identity_is_in_alternating(self):
        S4 = SymmetricGroup(4)
        self.assertIn(Permutation.identity(4), S4.alternating())


class TestConjugacyClass(unittest.TestCase):

    def test_identity_conjugacy_class_is_singleton(self):
        S4 = SymmetricGroup(4)
        cc = S4.conjugacy_class(Permutation.identity(4))
        self.assertEqual(len(cc), 1)
        self.assertEqual(cc[0], Permutation.identity(4))

    def test_transpositions_form_one_conjugacy_class_in_S3(self):
        S3 = SymmetricGroup(3)
        tau = Permutation.from_cycles(3, [1, 2])
        cc = S3.conjugacy_class(tau)
        # All three transpositions in S_3
        self.assertEqual(len(cc), 3)
        for p in cc:
            self.assertEqual(len(p.cycles()), 1)
            self.assertEqual(len(p.cycles()[0]), 2)

    def test_conjugacy_class_elements_have_same_cycle_type(self):
        S4 = SymmetricGroup(4)
        sigma = Permutation.from_cycles(4, [1, 2, 3])
        ref_type = tuple(sorted(len(c) for c in sigma.cycles()))
        for p in S4.conjugacy_class(sigma):
            p_type = tuple(sorted(len(c) for c in p.cycles()))
            self.assertEqual(p_type, ref_type)

    def test_conjugacy_class_size_S4_transposition(self):
        # There are 6 transpositions in S_4
        S4 = SymmetricGroup(4)
        tau = Permutation.from_cycles(4, [1, 2])
        self.assertEqual(len(S4.conjugacy_class(tau)), 6)


class TestCycleIndex(unittest.TestCase):

    def test_cycle_index_counts_sum_to_order(self):
        for n in range(1, 6):
            Sn = SymmetricGroup(n)
            self.assertEqual(sum(Sn.cycle_index().values()), Sn.order())

    def test_cycle_index_S3_known_values(self):
        # S_3 has: 1 identity (1,1,1), 3 transpositions (1,2), 2 three-cycles (3,)
        ci = SymmetricGroup(3).cycle_index()
        self.assertEqual(ci[(1, 1, 1)], 1)
        self.assertEqual(ci[(1, 2)],    3)
        self.assertEqual(ci[(3,)],      2)

    def test_cycle_index_keys_partition_n(self):
        S4 = SymmetricGroup(4)
        for cycle_type in S4.cycle_index():
            self.assertEqual(sum(cycle_type), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
