
"""
Vibe coded (Claude)
Tests for fourier.py: Discrete Fourier Transform on S_n.

All tests use random functions drawn from a seeded RNG so results are
reproducible.  The following properties are verified:

  1. Delta at identity   : f̂(ρ_λ) = I_{d_λ}  for all λ
  2. Uniform function    : f̂(ρ_trivial) = |S_n|, f̂(ρ_λ) = 0 for λ ≠ (n,)
  3. Linearity           : (af + bh)^ = a f̂ + b ĥ
  4. Plancherel identity : Σ|f|² = (1/|G|) Σ d_λ ‖f̂(ρ_λ)‖²_F
  5. Inversion           : sn_idft(sn_dft(f)) = f
  6. Convolution theorem : (f * h)^ = f̂ · ĥ  (matrix product per irrep)
  7. Sign function       : f̂(sign irrep) = sum of signs, others follow
  8. Right-shift         : f_σ(g)=f(gσ) ⟹ f̂_σ(ρ) = f̂(ρ) · ρ(σ)
"""

import unittest
import random
import math
import numpy as np

from symmetric_group import SymmetricGroup
from irreps import partitions_of, irrep
from fourier import sn_dft, sn_idft, plancherel_norm_sq, sn_convolve


def _random_function(Sn, rng, real=True):
    """Return a dict {Permutation: float} with i.i.d. Uniform[-1,1] values."""
    if real:
        return {g: rng.uniform(-1.0, 1.0) for g in Sn}
    else:
        return {g: complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for g in Sn}


class TestKnownTransforms(unittest.TestCase):
    """Closed-form DFTs whose results can be computed by hand."""

    def test_delta_at_identity(self):
        """f = δ_e  ⟹  f̂(ρ_λ) = I_{d_λ} for all λ."""
        for n in range(2, 5):
            Sn = SymmetricGroup(n)
            e = Sn.identity()
            f = {g: (1.0 if g == e else 0.0) for g in Sn}
            f_hat = sn_dft(f, n)
            for lam, F in f_hat.items():
                d = F.shape[0]
                self.assertTrue(
                    np.allclose(F, np.eye(d)),
                    f"S_{n} λ={lam}: f̂(δ_e) should be I_{d}, got\n{F}"
                )

    def test_uniform_function(self):
        """f ≡ 1  ⟹  f̂(trivial) = n!,  f̂(all other λ) = 0."""
        for n in range(2, 5):
            Sn = SymmetricGroup(n)
            f = {g: 1.0 for g in Sn}
            f_hat = sn_dft(f, n)
            trivial = (n,)
            for lam, F in f_hat.items():
                if lam == trivial:
                    self.assertTrue(
                        np.allclose(F, [[math.factorial(n)]]),
                        f"S_{n}: f̂(trivial) should be [[{math.factorial(n)}]], got {F}"
                    )
                else:
                    self.assertTrue(
                        np.allclose(F, np.zeros_like(F)),
                        f"S_{n} λ={lam}: f̂ of uniform should be 0, got\n{F}"
                    )

    def test_sign_function_trivial_component_is_zero(self):
        """f = sign  ⟹  f̂(trivial) = 0  (sign is odd, trivial is even)."""
        for n in range(2, 5):
            Sn = SymmetricGroup(n)
            f = {g: float(g.sign()) for g in Sn}
            f_hat = sn_dft(f, n)
            self.assertTrue(
                np.allclose(f_hat[(n,)], 0),
                f"S_{n}: sign function should have zero trivial component"
            )

    def test_sign_function_sign_component(self):
        """f = sign  ⟹  f̂(sign irrep) = sum_{g} sign(g) · sign(g) = |S_n|."""
        for n in range(2, 5):
            Sn = SymmetricGroup(n)
            sign_partition = (1,) * n
            f = {g: float(g.sign()) for g in Sn}
            f_hat = sn_dft(f, n)
            expected = float(math.factorial(n))
            self.assertTrue(
                np.allclose(f_hat[sign_partition], [[expected]]),
                f"S_{n}: f̂(sign, sign irrep) should be [[{expected}]]"
            )


class TestLinearity(unittest.TestCase):
    """DFT is linear: (af + bh)^ = a f̂ + b ĥ."""

    def _check(self, n, seed):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _random_function(Sn, rng)
        h = _random_function(Sn, rng)
        a = rng.uniform(-2, 2)
        b = rng.uniform(-2, 2)

        combo = {g: a * f[g] + b * h[g] for g in Sn}
        f_hat = sn_dft(f, n)
        h_hat = sn_dft(h, n)
        combo_hat = sn_dft(combo, n)

        for lam in partitions_of(n):
            expected = a * f_hat[lam] + b * h_hat[lam]
            self.assertTrue(
                np.allclose(combo_hat[lam], expected),
                f"S_{n} λ={lam}: linearity failed (seed={seed})"
            )

    def test_S2(self): self._check(2, 1)
    def test_S3(self): self._check(3, 2)
    def test_S4(self): self._check(4, 3)


class TestPlancherel(unittest.TestCase):
    """
    Plancherel / Parseval identity:
        Σ_{g} |f(g)|² = (1/|S_n|) Σ_λ d_λ · ‖f̂(ρ_λ)‖²_F
    """

    def _check(self, n, seed):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _random_function(Sn, rng)
        f_hat = sn_dft(f, n)

        l2_sq = sum(v ** 2 for v in f.values())
        planch = plancherel_norm_sq(f_hat, n)

        self.assertAlmostEqual(
            l2_sq, planch, places=10,
            msg=f"S_{n} Plancherel failed: Σ|f|²={l2_sq}, Parseval={planch} (seed={seed})"
        )

    def test_S2_seed_10(self): self._check(2, 10)
    def test_S2_seed_11(self): self._check(2, 11)
    def test_S3_seed_20(self): self._check(3, 20)
    def test_S3_seed_21(self): self._check(3, 21)
    def test_S4_seed_30(self): self._check(4, 30)
    def test_S4_seed_31(self): self._check(4, 31)


class TestInversion(unittest.TestCase):
    """sn_idft(sn_dft(f)) == f  for random real and complex functions."""

    def _check(self, n, seed, complex_valued=False):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _random_function(Sn, rng, real=not complex_valued)
        f_hat = sn_dft(f, n)
        f_rec = sn_idft(f_hat, n)

        for g in Sn:
            self.assertAlmostEqual(
                f[g], f_rec[g], places=10,
                msg=f"S_{n} inversion failed at g={g} (seed={seed})"
            )

    def test_S2_real(self):    self._check(2, 100)
    def test_S3_real(self):    self._check(3, 101)
    def test_S4_real(self):    self._check(4, 102)
    def test_S2_complex(self): self._check(2, 200, complex_valued=True)
    def test_S3_complex(self): self._check(3, 201, complex_valued=True)
    def test_S4_complex(self): self._check(4, 202, complex_valued=True)


class TestConvolutionTheorem(unittest.TestCase):
    """
    (f * h)^(ρ_λ) = f̂(ρ_λ) · ĥ(ρ_λ)  (matrix product)

    and the result matches direct pointwise convolution:
        (f * h)(g) = Σ_{x ∈ S_n} f(x) h(x⁻¹ g)
    """

    def _direct_convolve(self, f, h, Sn):
        return {
            g: sum(f.get(x, 0) * h.get(x.inverse() * g, 0) for x in Sn)
            for g in Sn
        }

    def _check(self, n, seed):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _random_function(Sn, rng)
        h = _random_function(Sn, rng)

        # Verify the transform is pointwise matrix product per irrep
        f_hat = sn_dft(f, n)
        h_hat = sn_dft(h, n)
        conv_hat_direct = sn_dft(self._direct_convolve(f, h, Sn), n)
        for lam in partitions_of(n):
            expected = f_hat[lam] @ h_hat[lam]
            self.assertTrue(
                np.allclose(conv_hat_direct[lam], expected),
                f"S_{n} λ={lam}: convolution theorem (transform side) failed (seed={seed})"
            )

        # Verify sn_convolve matches direct pointwise convolution
        conv_fft = sn_convolve(f, h, n)
        conv_naive = self._direct_convolve(f, h, Sn)
        for g in Sn:
            self.assertAlmostEqual(
                conv_fft[g], conv_naive[g], places=10,
                msg=f"S_{n} sn_convolve mismatch at g={g} (seed={seed})"
            )

    def test_S2_seed_40(self): self._check(2, 40)
    def test_S3_seed_50(self): self._check(3, 50)
    def test_S3_seed_51(self): self._check(3, 51)
    def test_S4_seed_60(self): self._check(4, 60)


class TestRightShift(unittest.TestCase):
    """
    Right-shift property:  let f_σ(g) = f(g·σ).

    Proof: f̂_σ(ρ) = Σ_g f(gσ) ρ(g);  set h = gσ ⟹ g = hσ⁻¹
           = Σ_h f(h) ρ(hσ⁻¹) = (Σ_h f(h) ρ(h)) ρ(σ⁻¹) = f̂(ρ) · ρ(σ⁻¹).
    """

    def _check(self, n, seed):
        Sn = SymmetricGroup(n)
        elements = Sn.elements()
        rng = random.Random(seed)
        f = _random_function(Sn, rng)

        # Pick a random shift σ
        sigma = rng.choice(elements)

        f_sigma = {g: f[g * sigma] for g in Sn}
        f_hat       = sn_dft(f, n)
        f_sigma_hat = sn_dft(f_sigma, n)

        for lam in partitions_of(n):
            rho_sigma_inv = irrep(lam)(sigma.inverse())
            expected = f_hat[lam] @ rho_sigma_inv
            self.assertTrue(
                np.allclose(f_sigma_hat[lam], expected),
                f"S_{n} λ={lam}: right-shift property failed (seed={seed})"
            )

    def test_S2(self): self._check(2, 70)
    def test_S3(self): self._check(3, 71)
    def test_S4(self): self._check(4, 72)


class TestFourierCoefficientsAreZeroForZeroFunction(unittest.TestCase):
    """f ≡ 0  ⟹  f̂(ρ_λ) = 0 for all λ."""

    def test_zero_function(self):
        for n in range(2, 5):
            Sn = SymmetricGroup(n)
            f = {g: 0.0 for g in Sn}
            f_hat = sn_dft(f, n)
            for lam, F in f_hat.items():
                self.assertTrue(
                    np.allclose(F, np.zeros_like(F)),
                    f"S_{n} λ={lam}: f̂(0) should be zero matrix"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
