"""
Vibe coded (Claude)
Tests for fft.py: Clausen's FFT on S_n.

The FFT and DFT produce identical output (up to floating-point error), so the
test strategy has two tiers:

  Tier 1 — Agreement with DFT (random functions):
    The FFT is correct iff its output matches the naive DFT on every irrep.
    This is tested with many random functions across multiple seeds and degrees.

  Tier 2 — Independent algebraic properties (random functions):
    Even without the DFT as a reference, the output must satisfy:
      1. Known transforms   : δ_e, f≡1, sign function
      2. Linearity          : (af + bh)^ = a f̂ + b ĥ
      3. Plancherel identity: Σ|f|² = (1/|G|) Σ d_λ ‖f̂(ρ_λ)‖²_F
      4. Inversion          : sn_idft(sn_fft(f)) = f
      5. Convolution theorem: (f*h)^ = f̂ · ĥ  per irrep
      6. Right-shift        : f_σ(g)=f(gσ) ⟹ f̂_σ(ρ) = f̂(ρ) · ρ(σ⁻¹)
"""

import unittest
import random
import math
import numpy as np

from symmetric_group import SymmetricGroup
from irreps import partitions_of, irrep
from fourier import sn_idft, plancherel_norm_sq, sn_dft
from fft import sn_fft


def _rand(Sn, rng, real=True):
    if real:
        return {g: rng.uniform(-1.0, 1.0) for g in Sn}
    return {g: complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for g in Sn}


# ===========================================================================
# Tier 1: FFT agrees with DFT
# ===========================================================================

class TestFFTMatchesDFT(unittest.TestCase):
    """
    sn_fft(f, n) == sn_dft(f, n)  entry-wise up to floating-point error,
    for random real and complex f, across S_2 through S_5.
    """

    ATOL = 1e-12

    def _check(self, n, seed, real=True):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _rand(Sn, rng, real=real)

        fft_hat = sn_fft(f, n)
        dft_hat = sn_dft(f, n)

        for lam in partitions_of(n):
            self.assertTrue(
                np.allclose(fft_hat[lam], dft_hat[lam], atol=self.ATOL),
                f"S_{n} λ={lam} seed={seed}: FFT and DFT disagree\n"
                f"  FFT:\n{fft_hat[lam]}\n  DFT:\n{dft_hat[lam]}"
            )

    # Real-valued functions
    def test_S2_real_s1(self):  self._check(2,  1)
    def test_S2_real_s2(self):  self._check(2,  2)
    def test_S3_real_s1(self):  self._check(3, 10)
    def test_S3_real_s2(self):  self._check(3, 11)
    def test_S3_real_s3(self):  self._check(3, 12)
    def test_S4_real_s1(self):  self._check(4, 20)
    def test_S4_real_s2(self):  self._check(4, 21)
    def test_S4_real_s3(self):  self._check(4, 22)
    def test_S5_real_s1(self):  self._check(5, 30)
    def test_S5_real_s2(self):  self._check(5, 31)

    # Complex-valued functions
    def test_S2_complex(self):  self._check(2, 100, real=False)
    def test_S3_complex(self):  self._check(3, 101, real=False)
    def test_S4_complex(self):  self._check(4, 102, real=False)
    def test_S5_complex(self):  self._check(5, 103, real=False)

    # Sparse function (most values zero)
    def test_S4_sparse(self):
        n = 4
        Sn = SymmetricGroup(n)
        elements = Sn.elements()
        rng = random.Random(200)
        f = {g: (rng.uniform(-1, 1) if rng.random() < 0.2 else 0.0) for g in elements}
        for lam in partitions_of(n):
            fft_hat = sn_fft(f, n)
            dft_hat = sn_dft(f, n)
            self.assertTrue(np.allclose(fft_hat[lam], dft_hat[lam], atol=self.ATOL))

    # Zero function
    def test_zero_function(self):
        for n in range(2, 5):
            Sn = SymmetricGroup(n)
            f = {g: 0.0 for g in Sn}
            for lam, F in sn_fft(f, n).items():
                self.assertTrue(np.allclose(F, 0), f"S_{n} λ={lam}: FFT(0) ≠ 0")


# ===========================================================================
# Tier 2: algebraic properties verified directly from the FFT output
# ===========================================================================

class TestFFTKnownTransforms(unittest.TestCase):
    """Closed-form results that can be checked without the DFT."""

    def test_delta_at_identity(self):
        """f = δ_e  ⟹  f̂(ρ_λ) = I_{d_λ}."""
        for n in range(2, 6):
            Sn = SymmetricGroup(n)
            e = Sn.identity()
            f = {g: (1.0 if g == e else 0.0) for g in Sn}
            for lam, F in sn_fft(f, n).items():
                d = F.shape[0]
                self.assertTrue(
                    np.allclose(F, np.eye(d)),
                    f"S_{n} λ={lam}: FFT(δ_e) should be I_{d}"
                )

    def test_uniform_function(self):
        """f ≡ 1  ⟹  f̂(trivial) = n!, f̂(other) = 0."""
        for n in range(2, 6):
            Sn = SymmetricGroup(n)
            f = {g: 1.0 for g in Sn}
            f_hat = sn_fft(f, n)
            trivial = (n,)
            for lam, F in f_hat.items():
                if lam == trivial:
                    self.assertTrue(
                        np.allclose(F, [[math.factorial(n)]]),
                        f"S_{n}: FFT(1)[trivial] should be [[{math.factorial(n)}]]"
                    )
                else:
                    self.assertTrue(
                        np.allclose(F, 0),
                        f"S_{n} λ={lam}: FFT(1)[non-trivial] should be 0"
                    )

    def test_sign_function_sign_component(self):
        """f = sign  ⟹  f̂(sign irrep) = n!."""
        for n in range(2, 6):
            Sn = SymmetricGroup(n)
            f = {g: float(g.sign()) for g in Sn}
            f_hat = sn_fft(f, n)
            sign_lam = (1,) * n
            self.assertTrue(
                np.allclose(f_hat[sign_lam], [[float(math.factorial(n))]]),
                f"S_{n}: FFT(sign)[sign irrep] should be [[{math.factorial(n)}]]"
            )

    def test_sign_function_trivial_component_is_zero(self):
        """f = sign  ⟹  f̂(trivial) = 0."""
        for n in range(2, 6):
            Sn = SymmetricGroup(n)
            f = {g: float(g.sign()) for g in Sn}
            self.assertTrue(
                np.allclose(sn_fft(f, n)[(n,)], 0),
                f"S_{n}: FFT(sign)[trivial] should be 0"
            )


class TestFFTLinearity(unittest.TestCase):
    """(af + bh)^ = a f̂ + b ĥ  for random a, b, f, h."""

    def _check(self, n, seed):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _rand(Sn, rng)
        h = _rand(Sn, rng)
        a, b = rng.uniform(-2, 2), rng.uniform(-2, 2)
        combo = {g: a * f[g] + b * h[g] for g in Sn}

        fft_f = sn_fft(f, n)
        fft_h = sn_fft(h, n)
        fft_combo = sn_fft(combo, n)

        for lam in partitions_of(n):
            expected = a * fft_f[lam] + b * fft_h[lam]
            self.assertTrue(
                np.allclose(fft_combo[lam], expected, atol=1e-12),
                f"S_{n} λ={lam}: linearity failed (seed={seed})"
            )

    def test_S2(self): self._check(2, 300)
    def test_S3(self): self._check(3, 301)
    def test_S4(self): self._check(4, 302)
    def test_S5(self): self._check(5, 303)


class TestFFTPlancherel(unittest.TestCase):
    """
    Plancherel identity:
        Σ_{g} |f(g)|² = (1/|S_n|) Σ_λ d_λ · ‖f̂(ρ_λ)‖²_F
    """

    def _check(self, n, seed):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _rand(Sn, rng)
        f_hat = sn_fft(f, n)
        l2_sq  = sum(v ** 2 for v in f.values())
        planch = plancherel_norm_sq(f_hat, n)
        self.assertAlmostEqual(l2_sq, planch, places=10,
            msg=f"S_{n} Plancherel failed (seed={seed}): Σ|f|²={l2_sq}, Parseval={planch}")

    def test_S2_s1(self): self._check(2, 400)
    def test_S2_s2(self): self._check(2, 401)
    def test_S3_s1(self): self._check(3, 410)
    def test_S3_s2(self): self._check(3, 411)
    def test_S4_s1(self): self._check(4, 420)
    def test_S4_s2(self): self._check(4, 421)
    def test_S5_s1(self): self._check(5, 430)


class TestFFTInversion(unittest.TestCase):
    """sn_idft(sn_fft(f)) == f  for random real and complex f."""

    def _check(self, n, seed, real=True):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _rand(Sn, rng, real=real)
        f_rec = sn_idft(sn_fft(f, n), n)
        for g in Sn:
            self.assertAlmostEqual(f[g], f_rec[g], places=10,
                msg=f"S_{n} inversion failed at g={g} (seed={seed})")

    def test_S2_real(self):    self._check(2, 500)
    def test_S3_real(self):    self._check(3, 501)
    def test_S4_real(self):    self._check(4, 502)
    def test_S5_real(self):    self._check(5, 503)
    def test_S2_complex(self): self._check(2, 600, real=False)
    def test_S3_complex(self): self._check(3, 601, real=False)
    def test_S4_complex(self): self._check(4, 602, real=False)


class TestFFTConvolutionTheorem(unittest.TestCase):
    """
    The FFT satisfies the convolution theorem:
        (f * h)^(ρ_λ) = f̂(ρ_λ) · ĥ(ρ_λ)

    verified against direct pointwise convolution
        (f * h)(g) = Σ_{x ∈ S_n} f(x) h(x⁻¹ g).
    """

    def _direct_convolve(self, f, h, Sn):
        return {
            g: sum(f.get(x, 0) * h.get(x.inverse() * g, 0) for x in Sn)
            for g in Sn
        }

    def _check(self, n, seed):
        Sn = SymmetricGroup(n)
        rng = random.Random(seed)
        f = _rand(Sn, rng)
        h = _rand(Sn, rng)

        f_hat = sn_fft(f, n)
        h_hat = sn_fft(h, n)
        conv_hat_direct = sn_fft(self._direct_convolve(f, h, Sn), n)

        for lam in partitions_of(n):
            expected = f_hat[lam] @ h_hat[lam]
            self.assertTrue(
                np.allclose(conv_hat_direct[lam], expected, atol=1e-10),
                f"S_{n} λ={lam}: convolution theorem failed (seed={seed})"
            )

    def test_S2(self): self._check(2, 700)
    def test_S3_s1(self): self._check(3, 701)
    def test_S3_s2(self): self._check(3, 702)
    def test_S4(self):   self._check(4, 703)


class TestFFTRightShift(unittest.TestCase):
    """
    Right-shift property: f_σ(g) = f(g·σ)  ⟹  f̂_σ(ρ) = f̂(ρ) · ρ(σ⁻¹).

    Proof: f̂_σ(ρ) = Σ_g f(gσ)ρ(g); set h=gσ ⟹ g=hσ⁻¹
           = Σ_h f(h) ρ(hσ⁻¹) = f̂(ρ) · ρ(σ⁻¹).
    """

    def _check(self, n, seed):
        Sn = SymmetricGroup(n)
        elements = Sn.elements()
        rng = random.Random(seed)
        f = _rand(Sn, rng)
        sigma = rng.choice(elements)
        f_sigma = {g: f[g * sigma] for g in Sn}

        fft_f = sn_fft(f, n)
        fft_f_sigma = sn_fft(f_sigma, n)

        for lam in partitions_of(n):
            expected = fft_f[lam] @ irrep(lam)(sigma.inverse())
            self.assertTrue(
                np.allclose(fft_f_sigma[lam], expected, atol=1e-12),
                f"S_{n} λ={lam}: right-shift failed (seed={seed})"
            )

    def test_S2(self): self._check(2, 800)
    def test_S3(self): self._check(3, 801)
    def test_S4(self): self._check(4, 802)
    def test_S5(self): self._check(5, 803)


if __name__ == "__main__":
    unittest.main(verbosity=2)
