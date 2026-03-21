"""
Vibe coded (Claude)
Tests for coset_fft.py: S_n / S_{n-k} Fourier transform.

Test strategy has three tiers:

  Tier 1 — Helpers
    1. to_coset_function / from_coset_function round-trip
    2. to_coset_function raises on non-invariant input

  Tier 2 — Agreement with DFT (random f_tilde, every n, every k)
    The main correctness criterion: sn_mod_sk_fft(f_tilde, n, k) must equal
    sn_dft(from_coset_function(f_tilde, n, k), n) for every partition.

  Tier 3 — Independent algebraic properties
    3.  k = 0 (constant):     only trivial Fourier component nonzero
    4.  k = n (full FFT):     output equals sn_fft on the same data
    5.  Uniform coset func:   f̂(trivial) = n!, all others 0
    6.  Linearity:            (af + bh)^ = a f̂ + b ĥ
    7.  Inversion:            sn_idft(sn_mod_sk_fft(f_tilde)) recovers f
    8.  Invalid k raises ValueError
"""

import math
import random
import unittest
from itertools import permutations as _perms

import numpy as np

from symmetric_group import SymmetricGroup
from irreps import partitions_of
from fourier import sn_dft, sn_idft
from fft import sn_fft
from coset_fft import sn_mod_sk_fft, to_coset_function, from_coset_function


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _all_k_tuples(n: int, k: int) -> list:
    """All ordered k-tuples of distinct elements from {1,...,n}."""
    if k == 0:
        return [()]
    return list(_perms(range(1, n + 1), k))


def _rand_coset(n: int, k: int, rng: random.Random, real: bool = True) -> dict:
    """Random f_tilde on the coset space S_n / S_{n-k}."""
    if real:
        return {tup: rng.uniform(-1.0, 1.0) for tup in _all_k_tuples(n, k)}
    return {tup: complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
            for tup in _all_k_tuples(n, k)}


# ---------------------------------------------------------------------------
# Tier 1: Helpers
# ---------------------------------------------------------------------------

class TestCosetHelpers(unittest.TestCase):
    """to_coset_function and from_coset_function."""

    def test_round_trip_coset_to_full_to_coset(self):
        """to_coset_function(from_coset_function(f_tilde)) == f_tilde."""
        rng = random.Random(1)
        for n in range(2, 5):
            for k in range(0, n + 1):
                f_tilde = _rand_coset(n, k, rng)
                f_full = from_coset_function(f_tilde, n, k)
                f_tilde2 = to_coset_function(f_full, n, k)
                for tup in _all_k_tuples(n, k):
                    self.assertAlmostEqual(
                        f_tilde[tup], f_tilde2.get(tup, 0.0), places=12,
                        msg=f"S_{n}/S_{n-k}: round-trip mismatch at {tup}"
                    )

    def test_from_coset_is_constant_on_each_coset(self):
        """
        f = from_coset_function(f_tilde) must be right-S_{n-k}-invariant:
        f(σ) depends only on (σ(n-k+1),...,σ(n)).
        """
        rng = random.Random(2)
        for n in range(2, 5):
            for k in range(0, n + 1):
                f_tilde = _rand_coset(n, k, rng)
                f_full = from_coset_function(f_tilde, n, k)
                Sn = SymmetricGroup(n)
                # Group σ by their coset key; all must share the same value
                groups: dict = {}
                for sigma in Sn:
                    key = tuple(sigma(i) for i in range(n - k + 1, n + 1))
                    groups.setdefault(key, []).append(f_full[sigma])
                for key, vals in groups.items():
                    self.assertTrue(
                        all(np.isclose(v, vals[0]) for v in vals),
                        f"S_{n}/S_{n-k}: f not constant on coset {key}"
                    )

    def test_to_coset_raises_on_non_invariant(self):
        """to_coset_function raises ValueError when f violates invariance."""
        # For n=3, k=1: f(σ) must depend only on σ(3).
        # Construct f that differs on two σ with the same σ(3).
        Sn = SymmetricGroup(3)
        elements = list(Sn)
        f = {g: 1.0 for g in elements}
        # Find two perms with the same σ(3) and assign different values
        for g in elements:
            if g(3) == 1:
                f[g] = 999.0
                break
        with self.assertRaises(ValueError):
            to_coset_function(f, 3, 1)

    def test_k0_coset_is_constant_function(self):
        """k=0: f_tilde = {(): val}, from_coset gives f ≡ val."""
        val = 3.14
        Sn = SymmetricGroup(4)
        f = from_coset_function({(): val}, 4, 0)
        for sigma in Sn:
            self.assertAlmostEqual(f[sigma], val)


# ---------------------------------------------------------------------------
# Tier 2: Agreement with DFT
# ---------------------------------------------------------------------------

class TestCosetFFTMatchesDFT(unittest.TestCase):
    """
    sn_mod_sk_fft(f_tilde, n, k) must equal
    sn_dft(from_coset_function(f_tilde, n, k), n)
    for every partition, every valid (n, k), and multiple random seeds.
    """

    ATOL = 1e-11

    def _check(self, n: int, k: int, seed: int, real: bool = True):
        rng = random.Random(seed)
        f_tilde = _rand_coset(n, k, rng, real=real)
        f_full = from_coset_function(f_tilde, n, k)

        got = sn_mod_sk_fft(f_tilde, n, k)
        ref = sn_dft(f_full, n)

        for lam in partitions_of(n):
            self.assertTrue(
                np.allclose(got[lam], ref[lam], atol=self.ATOL),
                f"S_{n}/S_{n-k} k={k} lam={lam} seed={seed}: "
                f"coset FFT disagrees with DFT\n  got:{got[lam]}\n  ref:{ref[lam]}"
            )

    # k = 0 (constant function)
    def test_k0_S2(self):  self._check(2, 0, 10)
    def test_k0_S3(self):  self._check(3, 0, 11)
    def test_k0_S4(self):  self._check(4, 0, 12)
    def test_k0_S5(self):  self._check(5, 0, 13)

    # k = 1
    def test_k1_S2_s1(self): self._check(2, 1, 20)
    def test_k1_S3_s1(self): self._check(3, 1, 21)
    def test_k1_S3_s2(self): self._check(3, 1, 22)
    def test_k1_S4_s1(self): self._check(4, 1, 23)
    def test_k1_S4_s2(self): self._check(4, 1, 24)
    def test_k1_S5_s1(self): self._check(5, 1, 25)

    # k = 2
    def test_k2_S3_s1(self): self._check(3, 2, 30)
    def test_k2_S4_s1(self): self._check(4, 2, 31)
    def test_k2_S4_s2(self): self._check(4, 2, 32)
    def test_k2_S5_s1(self): self._check(5, 2, 33)

    # k = 3
    def test_k3_S4_s1(self): self._check(4, 3, 40)
    def test_k3_S5_s1(self): self._check(5, 3, 41)

    # k = n (full FFT, no invariance)
    def test_kn_S2(self):  self._check(2, 2, 50)
    def test_kn_S3(self):  self._check(3, 3, 51)
    def test_kn_S4(self):  self._check(4, 4, 52)
    def test_kn_S5(self):  self._check(5, 5, 53)

    # complex-valued
    def test_complex_k1_S4(self): self._check(4, 1, 60, real=False)
    def test_complex_k2_S4(self): self._check(4, 2, 61, real=False)
    def test_complex_kn_S3(self): self._check(3, 3, 62, real=False)

    # sparse (most values zero)
    def test_sparse_k2_S4(self):
        rng = random.Random(70)
        n, k = 4, 2
        f_tilde = {tup: (rng.uniform(-1, 1) if rng.random() < 0.3 else 0.0)
                   for tup in _all_k_tuples(n, k)}
        f_full = from_coset_function(f_tilde, n, k)
        got = sn_mod_sk_fft(f_tilde, n, k)
        ref = sn_dft(f_full, n)
        for lam in partitions_of(n):
            self.assertTrue(np.allclose(got[lam], ref[lam], atol=self.ATOL))

    # zero function
    def test_zero_function_all_k(self):
        for n in range(2, 5):
            for k in range(0, n + 1):
                f_tilde = {tup: 0.0 for tup in _all_k_tuples(n, k)}
                for lam, F in sn_mod_sk_fft(f_tilde, n, k).items():
                    self.assertTrue(np.allclose(F, 0),
                                    f"S_{n}/S_{n-k}: FFT(0) ≠ 0 at λ={lam}")


# ---------------------------------------------------------------------------
# Tier 3: Algebraic properties
# ---------------------------------------------------------------------------

class TestKEqualsZero(unittest.TestCase):
    """k=0: f is constant on S_n → only trivial Fourier coefficient nonzero."""

    def test_only_trivial_nonzero(self):
        for n in range(2, 6):
            val = 2.5
            f_hat = sn_mod_sk_fft({(): val}, n, 0)
            for lam, F in f_hat.items():
                if lam == (n,):
                    expected = np.array([[val * math.factorial(n)]])
                    self.assertTrue(np.allclose(F, expected),
                                    f"S_{n} k=0: f̂(trivial) wrong")
                else:
                    self.assertTrue(np.allclose(F, 0),
                                    f"S_{n} k=0: f̂({lam}) should be 0")

    def test_zero_constant_gives_all_zeros(self):
        for n in range(2, 5):
            f_hat = sn_mod_sk_fft({(): 0.0}, n, 0)
            for lam, F in f_hat.items():
                self.assertTrue(np.allclose(F, 0))


class TestKEqualsN(unittest.TestCase):
    """k=n: no invariance — output must equal sn_fft on the same data."""

    def _check(self, n: int, seed: int):
        rng = random.Random(seed)
        # f_tilde with k=n: keys are n-tuples = one-line notation of permutations
        Sn = SymmetricGroup(n)
        f_perm = {g: rng.uniform(-1, 1) for g in Sn}

        # Build f_tilde from the permutation dict
        f_tilde = {tuple(g(i) for i in range(1, n + 1)): v
                   for g, v in f_perm.items()}

        got = sn_mod_sk_fft(f_tilde, n, n)
        ref = sn_fft(f_perm, n)

        for lam in partitions_of(n):
            self.assertTrue(
                np.allclose(got[lam], ref[lam], atol=1e-11),
                f"S_{n} k=n={n}: coset FFT disagrees with sn_fft at λ={lam}"
            )

    def test_S2(self): self._check(2, 80)
    def test_S3(self): self._check(3, 81)
    def test_S4(self): self._check(4, 82)
    def test_S5(self): self._check(5, 83)


class TestUniformCosetFunction(unittest.TestCase):
    """
    f_tilde ≡ 1 on all k-tuples.
    The expanded f(σ) = 1 for all σ, so f̂(trivial) = n! and f̂(other) = 0.
    """

    def _check(self, n: int, k: int):
        f_tilde = {tup: 1.0 for tup in _all_k_tuples(n, k)}
        f_hat = sn_mod_sk_fft(f_tilde, n, k)
        for lam, F in f_hat.items():
            if lam == (n,):
                self.assertTrue(
                    np.allclose(F, [[float(math.factorial(n))]]),
                    f"S_{n}/S_{n-k}: f̂(trivial) should be [[{math.factorial(n)}]]"
                )
            else:
                self.assertTrue(
                    np.allclose(F, 0),
                    f"S_{n}/S_{n-k}: f̂({lam}) for uniform f should be 0"
                )

    def test_S3_k0(self): self._check(3, 0)
    def test_S3_k1(self): self._check(3, 1)
    def test_S3_k2(self): self._check(3, 2)
    def test_S4_k1(self): self._check(4, 1)
    def test_S4_k2(self): self._check(4, 2)
    def test_S4_k3(self): self._check(4, 3)
    def test_S5_k2(self): self._check(5, 2)


class TestLinearity(unittest.TestCase):
    """(a·f + b·h)^ = a·f̂ + b·ĥ  for random scalars and coset functions."""

    def _check(self, n: int, k: int, seed: int):
        rng = random.Random(seed)
        f = _rand_coset(n, k, rng)
        h = _rand_coset(n, k, rng)
        a = rng.uniform(-2, 2)
        b = rng.uniform(-2, 2)
        combo = {tup: a * f[tup] + b * h[tup] for tup in f}

        fhat = sn_mod_sk_fft(f, n, k)
        hhat = sn_mod_sk_fft(h, n, k)
        chat = sn_mod_sk_fft(combo, n, k)

        for lam in partitions_of(n):
            expected = a * fhat[lam] + b * hhat[lam]
            self.assertTrue(
                np.allclose(chat[lam], expected, atol=1e-11),
                f"S_{n}/S_{n-k} lam={lam}: linearity failed (seed={seed})"
            )

    def test_S3_k0(self): self._check(3, 0, 90)
    def test_S3_k1(self): self._check(3, 1, 91)
    def test_S4_k1(self): self._check(4, 1, 92)
    def test_S4_k2(self): self._check(4, 2, 93)
    def test_S5_k2(self): self._check(5, 2, 94)
    def test_S5_k3(self): self._check(5, 3, 95)


class TestInversion(unittest.TestCase):
    """sn_idft(sn_mod_sk_fft(f_tilde)) recovers from_coset_function(f_tilde)."""

    def _check(self, n: int, k: int, seed: int, real: bool = True):
        rng = random.Random(seed)
        f_tilde = _rand_coset(n, k, rng, real=real)
        f_full = from_coset_function(f_tilde, n, k)
        Sn = SymmetricGroup(n)

        f_rec = sn_idft(sn_mod_sk_fft(f_tilde, n, k), n)

        for sigma in Sn:
            self.assertAlmostEqual(
                f_full[sigma], f_rec[sigma], places=10,
                msg=f"S_{n}/S_{n-k} inversion failed at σ={sigma} (seed={seed})"
            )

    def test_S3_k0(self):    self._check(3, 0, 100)
    def test_S3_k1(self):    self._check(3, 1, 101)
    def test_S4_k1(self):    self._check(4, 1, 102)
    def test_S4_k2(self):    self._check(4, 2, 103)
    def test_S5_k2(self):    self._check(5, 2, 104)
    def test_S5_k3(self):    self._check(5, 3, 105)
    def test_S4_kn_complex(self): self._check(4, 4, 106, real=False)


class TestInvalidInputs(unittest.TestCase):
    """sn_mod_sk_fft raises ValueError for k < 0 or k > n."""

    def test_k_negative(self):
        with self.assertRaises(ValueError):
            sn_mod_sk_fft({}, 3, -1)

    def test_k_greater_than_n(self):
        with self.assertRaises(ValueError):
            sn_mod_sk_fft({}, 3, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
