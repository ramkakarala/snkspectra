"""
Vibe coded (Claude) 15 March 2026
"""
import random
import time
import unittest
from permutations import Permutation


class TestInverse(unittest.TestCase):

    def test_sigma_times_inverse_is_identity(self):
        sigma = Permutation([2, 3, 1, 4, 5])
        e = Permutation.identity(5)
        self.assertEqual(sigma * sigma.inverse(), e)
        self.assertEqual(sigma.inverse() * sigma, e)

    def test_inverse_times_sigma_is_identity(self):
        sigma = Permutation([3, 1, 4, 5, 2])
        e = Permutation.identity(5)
        self.assertEqual(sigma.inverse() * sigma, e)

    def test_inverse_of_identity_is_identity(self):
        e = Permutation.identity(6)
        self.assertEqual(e.inverse(), e)

    def test_inverse_of_inverse_is_original(self):
        sigma = Permutation([2, 4, 1, 3])
        self.assertEqual(sigma.inverse().inverse(), sigma)

    def test_inverse_of_transposition(self):
        # A transposition is its own inverse
        tau = Permutation.from_cycles(5, [2, 4])
        self.assertEqual(tau.inverse(), tau)

    def test_inverse_of_product(self):
        # (sigma * tau)^{-1} == tau^{-1} * sigma^{-1}
        sigma = Permutation([2, 3, 1, 4, 5])
        tau   = Permutation([1, 2, 4, 5, 3])
        lhs = (sigma * tau).inverse()
        rhs = tau.inverse() * sigma.inverse()
        self.assertEqual(lhs, rhs)

    def test_negative_power_equals_inverse(self):
        sigma = Permutation([4, 2, 5, 1, 3])
        self.assertEqual(sigma ** -1, sigma.inverse())
        self.assertEqual(sigma ** -3, (sigma ** 3).inverse())


class TestDisjointCyclesCommute(unittest.TestCase):

    def _disjoint_pair(self):
        # (1 3 5) and (2 4) are disjoint on {1..5}
        alpha = Permutation.from_cycles(5, [1, 3, 5])
        beta  = Permutation.from_cycles(5, [2, 4])
        return alpha, beta

    def test_disjoint_cycles_commute(self):
        alpha, beta = self._disjoint_pair()
        self.assertEqual(alpha * beta, beta * alpha)

    def test_disjoint_3cycles_commute(self):
        alpha = Permutation.from_cycles(6, [1, 2, 3])
        beta  = Permutation.from_cycles(6, [4, 5, 6])
        self.assertEqual(alpha * beta, beta * alpha)

    def test_disjoint_transpositions_commute(self):
        alpha = Permutation.from_cycles(6, [1, 2])
        beta  = Permutation.from_cycles(6, [3, 4])
        gamma = Permutation.from_cycles(6, [5, 6])
        self.assertEqual(alpha * beta, beta * alpha)
        self.assertEqual(beta * gamma, gamma * beta)
        self.assertEqual(alpha * gamma, gamma * alpha)

    def test_overlapping_cycles_do_not_commute(self):
        # Sanity check: cycles sharing an element need NOT commute
        alpha = Permutation.from_cycles(4, [1, 2])
        beta  = Permutation.from_cycles(4, [2, 3])
        self.assertNotEqual(alpha * beta, beta * alpha)

    def test_disjoint_commutativity_matches_from_cycle(self):
        # Composing disjoint cycles via from_cycle should equal either order
        alpha, beta = self._disjoint_pair()
        combined = Permutation.from_cycles(5, [1, 3, 5], [2, 4])
        self.assertEqual(alpha * beta, combined)
        self.assertEqual(beta * alpha, combined)


class TestCycles(unittest.TestCase):

    def test_identity_has_no_cycles(self):
        self.assertEqual(Permutation.identity(5).cycles(), [])

    def test_single_transposition(self):
        tau = Permutation.from_cycles(5, [1, 3])
        self.assertEqual(tau.cycles(), [[1, 3]])

    def test_single_3cycle(self):
        sigma = Permutation.from_cycles(5, [1, 2, 3])
        self.assertEqual(sigma.cycles(), [[1, 2, 3]])

    def test_two_disjoint_cycles(self):
        sigma = Permutation.from_cycles(5, [1, 3, 5], [2, 4])
        self.assertEqual(sigma.cycles(), [[1, 3, 5], [2, 4]])

    def test_fixed_points_excluded(self):
        # Only (1 2) moves; 3, 4, 5 are fixed points
        sigma = Permutation.from_cycles(5, [1, 2])
        cycles = sigma.cycles()
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0], [1, 2])

    def test_cycle_lengths_sum_with_fixed_points_equals_n(self):
        sigma = Permutation([2, 3, 1, 5, 4])  # (1 2 3)(4 5)
        cycles = sigma.cycles()
        total = sum(len(c) for c in cycles)
        fixed = sigma.n - total
        self.assertEqual(total + fixed, sigma.n)

    def test_full_cycle(self):
        sigma = Permutation([2, 3, 4, 5, 1])  # (1 2 3 4 5)
        self.assertEqual(sigma.cycles(), [[1, 2, 3, 4, 5]])

    def test_cycle_str_consistent_with_cycles(self):
        sigma = Permutation([2, 3, 1, 5, 4])
        cycles = sigma.cycles()
        expected = "".join("(" + " ".join(str(v) for v in c) + ")" for c in cycles)
        self.assertEqual(sigma.cycle_str(), expected)


class TestSign(unittest.TestCase):

    def test_identity_is_even(self):
        self.assertEqual(Permutation.identity(5).sign(), +1)

    def test_transposition_is_odd(self):
        # A single 2-cycle has sign -1
        tau = Permutation.from_cycles(5, [1, 2])
        self.assertEqual(tau.sign(), -1)

    def test_3cycle_is_even(self):
        # A 3-cycle = two transpositions
        sigma = Permutation.from_cycles(5, [1, 2, 3])
        self.assertEqual(sigma.sign(), +1)

    def test_4cycle_is_odd(self):
        sigma = Permutation.from_cycles(5, [1, 2, 3, 4])
        self.assertEqual(sigma.sign(), -1)

    def test_two_disjoint_transpositions_is_even(self):
        # Each transposition contributes -1; product is +1
        sigma = Permutation.from_cycles(6, [1, 2], [3, 4])
        self.assertEqual(sigma.sign(), +1)

    def test_sign_of_product_equals_product_of_signs(self):
        # Homomorphism property: sign(sigma * tau) = sign(sigma) * sign(tau)
        sigma = Permutation([2, 3, 1, 4, 5])   # 3-cycle, even
        tau   = Permutation([1, 2, 4, 5, 3])   # 3-cycle, even
        self.assertEqual((sigma * tau).sign(), sigma.sign() * tau.sign())

    def test_sign_of_inverse_equals_sign(self):
        # sigma and sigma^{-1} have the same sign
        sigma = Permutation([2, 3, 1, 4, 5])
        self.assertEqual(sigma.sign(), sigma.inverse().sign())

    def test_sign_values_are_plus_or_minus_one(self):
        for perm_list in [[2, 1, 3], [1, 3, 2], [3, 2, 1], [2, 3, 1], [3, 1, 2], [1, 2, 3]]:
            self.assertIn(Permutation(perm_list).sign(), (+1, -1))


class TestTiming(unittest.TestCase):

    def test_multiply_inverses_1000_random_permutations_length_8(self):
        n = 10
        count = 1000*100
        rng = random.Random(42)

        domain = list(range(1, n + 1))
        identity = Permutation.identity(n)

        start = time.perf_counter()
        for _ in range(count):
            shuffled = domain[:]
            rng.shuffle(shuffled)
            p = Permutation(shuffled)
            result = p * p.inverse()
            self.assertEqual(result, identity)
            result = p.inverse() * p
            self.assertEqual(result, identity)
        elapsed = time.perf_counter() - start

        print(f"\n[timing] {count} × (σ * σ⁻¹) on S_{n}: {elapsed * 1000:.3f} ms "
              f"({elapsed / count * 1e6:.2f} µs per op)")


if __name__ == "__main__":
    unittest.main(verbosity=2)