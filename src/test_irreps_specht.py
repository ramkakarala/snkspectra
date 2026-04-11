"""
Tests for irreps_specht.py: irreducible representations of S_n via Gram-Schmidt
orthogonalization of the Specht module.

Mirrors test_irreps.py with two additional test classes specific to the Specht
construction:

  TestHomomorphism           rho(a*b) = rho(a) @ rho(b)
  TestOrthogonalMatrices     rho(g)^T rho(g) = I  (orthogonal representation)
  TestDimensionSumRule       sum_{λ ⊢ n} (dim λ)^2 = n!
  TestCharacterOrthogonality <chi_λ, chi_μ> = δ_{λμ}
  TestMatrixCoefficientOrthogonality  Schur orthogonality of matrix coefficients
  TestAverageOverGroup       (1/|G|) Σ rho(g) = 0 for non-trivial irreps
  TestEquivalenceWithYOF     same characters as Young's orthogonal form
  TestSpechtBasisOrthonormality  Gram-Schmidt basis rows are orthonormal
"""

import unittest
import math
import numpy as np
from symmetric_group import SymmetricGroup
from irreps import partitions_of, irrep as irrep_yof
from irreps_specht import irrep_specht, specht_basis


def chi(lam, g):
    """Character of Specht irrep λ at group element g."""
    return np.trace(irrep_specht(lam)(g)).real


def inner_product(lam, mu, Sn):
    """Normalised inner product of characters chi_λ and chi_μ over S_n."""
    return sum(chi(lam, g) * chi(mu, g) for g in Sn) / Sn.order()


class TestHomomorphism(unittest.TestCase):
    """rho(a * b) = rho(a) @ rho(b) for all a, b in S_n."""

    def _check(self, n):
        Sn = SymmetricGroup(n)
        elements = Sn.elements()
        for lam in partitions_of(n):
            rho = irrep_specht(lam)
            for a in elements[:8]:
                for b in elements[:8]:
                    self.assertTrue(
                        np.allclose(rho(a * b), rho(a) @ rho(b), atol=1e-10),
                        f"Homomorphism failed for λ={lam} in S_{n}"
                    )

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


class TestOrthogonalMatrices(unittest.TestCase):
    """rho(g)^T @ rho(g) = I for all g in S_n (orthogonal representation)."""

    def _check(self, n):
        Sn = SymmetricGroup(n)
        for lam in partitions_of(n):
            rho = irrep_specht(lam)
            dim = rho(Sn.identity()).shape[0]
            for g in Sn.elements():
                M = rho(g)
                self.assertTrue(
                    np.allclose(M.T @ M, np.eye(dim), atol=1e-10),
                    f"rho(g) not orthogonal for λ={lam}, g={g} in S_{n}"
                )

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


class TestDimensionSumRule(unittest.TestCase):
    """sum_{λ ⊢ n} (dim λ)^2 = n!"""

    def _check(self, n):
        Sn = SymmetricGroup(n)
        e = Sn.identity()
        total = sum(
            round(chi(lam, e)) ** 2
            for lam in partitions_of(n)
        )
        self.assertEqual(total, math.factorial(n),
                         f"Dimension sum rule failed for S_{n}")

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


class TestCharacterOrthogonality(unittest.TestCase):
    """
    <chi_λ, chi_μ> = 1 if λ == μ  (irreducible)
                   = 0 if λ != μ  (distinct irreps are orthogonal)
    """

    def _check_all_pairs(self, n):
        Sn = SymmetricGroup(n)
        parts = partitions_of(n)
        for i, lam in enumerate(parts):
            for j, mu in enumerate(parts):
                ip = inner_product(lam, mu, Sn)
                expected = 1 if i == j else 0
                self.assertAlmostEqual(
                    ip, expected, places=10,
                    msg=f"<chi_{lam}, chi_{mu}> = {ip:.6f}, expected {expected}"
                )

    def test_S2(self): self._check_all_pairs(2)
    def test_S3(self): self._check_all_pairs(3)
    def test_S4(self): self._check_all_pairs(4)
    def test_S5(self): self._check_all_pairs(5)


class TestMatrixCoefficientOrthogonality(unittest.TestCase):
    """
    Schur orthogonality of matrix coefficients (Peter-Weyl):

        sum_{g in G} rho_λ(g)_{ij} * rho_μ(g)_{kl}
            = (|G| / dim_λ) * δ_{λμ} * δ_{ik} * δ_{jl}

    Distinct (λ, i, j) coefficient functions are pairwise orthogonal on G.
    """

    def _check(self, n):
        Sn = SymmetricGroup(n)
        elements = Sn.elements()

        basis_fns = []
        for lam in partitions_of(n):
            rho = irrep_specht(lam)
            matrices = [rho(g) for g in elements]
            dim = matrices[0].shape[0]
            for i in range(dim):
                for j in range(dim):
                    vec = np.array([M[i, j] for M in matrices])
                    basis_fns.append(((lam, i, j), vec))

        for a, (label_a, v_a) in enumerate(basis_fns):
            for b, (label_b, v_b) in enumerate(basis_fns):
                if a >= b:
                    continue
                ip = v_a @ v_b
                self.assertAlmostEqual(
                    ip, 0.0, places=10,
                    msg=f"Matrix coefficients {label_a} and {label_b} not orthogonal: {ip:.6e}"
                )

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


class TestAverageOverGroup(unittest.TestCase):
    """
    (1/|G|) * sum_{g in S_n} rho_λ(g) = 0  for all non-trivial irreps λ.
    """

    def _check(self, n):
        Sn = SymmetricGroup(n)
        elements = Sn.elements()
        trivial = (n,)
        for lam in partitions_of(n):
            if lam == trivial:
                continue
            rho = irrep_specht(lam)
            avg = sum(rho(g) for g in elements) / Sn.order()
            self.assertTrue(
                np.allclose(avg, 0, atol=1e-10),
                f"Average of irrep {lam} over S_{n} is not zero:\n{avg}"
            )

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


class TestEquivalenceWithYOF(unittest.TestCase):
    """
    The Specht and Young's orthogonal form representations have identical
    characters (traces), confirming they are the same irrep up to a change
    of orthonormal basis.
    """

    def _check(self, n):
        Sn = SymmetricGroup(n)
        for lam in partitions_of(n):
            rho_s = irrep_specht(lam)
            rho_y = irrep_yof(lam)
            for g in Sn.elements():
                chi_s = np.trace(rho_s(g)).real
                chi_y = np.trace(rho_y(g)).real
                self.assertAlmostEqual(
                    chi_s, chi_y, places=10,
                    msg=f"Character mismatch for λ={lam}, g={g}: Specht={chi_s:.6f}, YOF={chi_y:.6f}"
                )

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


class TestSpechtBasisOrthonormality(unittest.TestCase):
    """
    The Gram-Schmidt basis matrix B satisfies B @ B.T = I_dim,
    i.e. the rows are orthonormal vectors in the tabloid inner product.
    """

    def _check(self, n):
        from irreps import partitions_of
        for lam in partitions_of(n):
            B, _, _ = specht_basis(lam)
            dim = B.shape[0]
            self.assertTrue(
                np.allclose(B @ B.T, np.eye(dim), atol=1e-10),
                f"Specht basis not orthonormal for λ={lam} in S_{n}"
            )

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
