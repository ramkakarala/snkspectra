"""
Vibe coded (Claude)
Tests for irreps.py: irreducible representations of S_n.

Two fundamental facts from representation theory are tested:

1. Sum of squares of dimensions equals the group order:
       sum_{λ ⊢ n} (dim λ)^2 = n!

2. Character orthogonality:
       <chi_λ, chi_μ> = (1/n!) * sum_{g in S_n} chi_λ(g) * chi_μ(g) = δ_{λμ}
"""

import unittest
import math
import numpy as np
from symmetric_group import SymmetricGroup
from irreps import partitions_of, irrep


def chi(lam, g):
    """Character of irrep λ at group element g (real-valued)."""
    return np.trace(irrep(lam)(g)).real


def inner_product(lam, mu, Sn):
    """Normalised inner product of characters chi_λ and chi_μ over S_n."""
    return sum(chi(lam, g) * chi(mu, g) for g in Sn) / Sn.order()


class TestDimensionSumRule(unittest.TestCase):
    """sum_{λ ⊢ n} (dim λ)^2 = n!"""

    def _check(self, n):
        Sn = SymmetricGroup(n)
        e = Sn.identity()
        total = sum(
            round(chi(lam, e)) ** 2          # chi(λ, e) = dim λ
            for lam in partitions_of(n)
        )
        self.assertEqual(total, math.factorial(n),
                         f"dimension sum rule failed for S_{n}")

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


class TestCharacterOrthogonality(unittest.TestCase):
    """
    <chi_λ, chi_μ> = 1 if λ == μ (self inner product = 1, i.e. irreducible)
                   = 0 if λ != μ (distinct irreps are orthogonal)
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

        sum_{g in G} rho_lambda(g)_{ij} * rho_mu(g)_{kl}
            = (|G| / dim_lambda) * delta_{lambda,mu} * delta_{ik} * delta_{jl}

    In particular, matrix coefficient functions for distinct (lambda, i, j) triples
    are pairwise orthogonal on the group.
    """

    def _check(self, n):
        Sn = SymmetricGroup(n)
        elements = Sn.elements()
        order = Sn.order()

        # Collect all (lambda, i, j) matrix coefficient functions as vectors over G
        basis_fns = []   # list of (label, vector)
        for lam in partitions_of(n):
            rho = irrep(lam)
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
                    msg=f"Matrix coefficients {label_a} and {label_b} not orthogonal: inner product = {ip:.6e}"
                )

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


class TestAverageOverGroup(unittest.TestCase):
    """
    (1/|G|) * sum_{g in S_n} rho_lambda(g) = 0  for all non-trivial irreps lambda.

    By Schur's lemma the projection onto the trivial isotypic component is zero
    whenever the irrep is non-trivial.  The trivial irrep is (n,).
    """

    def _check(self, n):
        Sn = SymmetricGroup(n)
        elements = Sn.elements()
        order = Sn.order()
        trivial = (n,)
        for lam in partitions_of(n):
            if lam == trivial:
                continue
            rho = irrep(lam)
            avg = sum(rho(g) for g in elements) / order
            self.assertTrue(
                np.allclose(avg, 0, atol=1e-10),
                f"Average of irrep {lam} over S_{n} is not zero:\n{avg}"
            )

    def test_S2(self): self._check(2)
    def test_S3(self): self._check(3)
    def test_S4(self): self._check(4)
    def test_S5(self): self._check(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
