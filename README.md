# snkspectra

A vibe-coded Python library for working with permutations and their representations in group theory.  Even this README is written by AI.

## Overview

snkspectra provides a clean implementation of the symmetric group Sₙ — the group of all permutations on n elements. It supports multiple standard notations, algebraic operations, cycle decomposition, sign computation, matrix representations, and the full symmetric group as a first-class object.

## Features

- **Multiple input formats**: dict mapping, one-line notation (list/tuple), cycle notation, two-line notation
- **Notation conversions**: one-line, two-line, and disjoint cycle notation
- **Algebraic operations**: composition, inversion, exponentiation (including negative powers)
- **Cycle decomposition**: `cycles()` returns the disjoint cycle structure as a list of lists
- **Sign (signature)**: `sign()` returns +1 (even) or -1 (odd)
- **Symmetric group**: `SymmetricGroup(n)` enumerates all n! elements with group-theoretic queries
- **Matrix representation**: converts permutations to binary permutation matrices
- **Homomorphism verification**: M(σ · τ) = M(σ) @ M(τ)

## Installation

No package installation required. Clone the repo and ensure NumPy is available:

```bash
pip install numpy
```

## Usage

### Creating Permutations

```python
from src import Permutation

# From one-line notation (σ(1), σ(2), ..., σ(n))
sigma = Permutation([3, 1, 5, 2, 4])

# From a dict mapping
sigma = Permutation({1: 3, 2: 1, 3: 5, 4: 2, 5: 4})

# From cycle notation
sigma = Permutation.from_cycles(5, [1, 3, 5], [2, 4])

# From two-line notation
sigma = Permutation.from_two_line([1, 2, 3, 4, 5], [3, 1, 5, 2, 4])

# Identity permutation
e = Permutation.identity(5)
```

### Viewing Representations

```python
sigma.one_line_str()   # (3 1 5 2 4)
sigma.two_line_str()   # ( 1  2  3  4  5 )
                       # ( 3  1  5  2  4 )
sigma.cycle_str()      # (1 3 5)(2 4)
sigma.print_all()      # prints all three representations
```

### Algebraic Operations

```python
tau = Permutation([2, 3, 1, 5, 4])

product   = sigma * tau          # composition: (sigma * tau)(x) = sigma(tau(x))
sigma_inv = sigma.inverse()      # inverse
sigma_sq  = sigma ** 2           # exponentiation
sigma(3)                         # evaluate: returns 5
sigma * sigma.inverse() == Permutation.identity(5)   # True
```

### Cycle Decomposition and Sign

```python
sigma = Permutation([2, 3, 1, 5, 4])

sigma.cycles()    # [[1, 2, 3], [4, 5]]
sigma.cycle_str() # (1 2 3)(4 5)
sigma.sign()      # +1  (even: one 3-cycle + one 2-cycle = 2+1 = 3 transpositions... wait, even)
```

### Symmetric Group

```python
from src import SymmetricGroup

S4 = SymmetricGroup(4)

S4.order()        # 24
S4.identity()     # Permutation([1, 2, 3, 4])
list(S4)          # all 24 Permutation objects

# Alternating group A_n (even permutations)
S4.alternating()  # list of 12 even permutations

# Conjugacy class
tau = Permutation.from_cycles(4, [1, 2])
S4.conjugacy_class(tau)   # all 6 transpositions in S_4

# Cycle index (maps cycle-type tuple -> count)
S4.cycle_index()  # {(1,1,1,1): 1, (1,1,2): 6, (1,3): 8, (2,2): 3, (4,): 6}
```

### Matrix Representation

```python
from src import perm_mat_rep
import numpy as np

M_sigma = perm_mat_rep(sigma)
M_tau   = perm_mat_rep(tau)

np.array_equal(perm_mat_rep(sigma * tau), M_sigma @ M_tau)   # True  (homomorphism)
np.array_equal(M_sigma @ M_sigma.T, np.eye(5))               # True  (orthogonal)
```

## Project Structure

```
snkspectra/
├── src/
│   ├── __init__.py                      # Exports Permutation, SymmetricGroup, perm_mat_rep
│   ├── permutations.py                  # Permutation class
│   ├── symmetric_group.py               # SymmetricGroup class
│   ├── permutation_matrix_rep.py        # Matrix representation
│   ├── test_permutations.py             # Permutation tests
│   ├── test_symmetric_group.py          # SymmetricGroup tests
│   └── test_permutation_matrix_rep.py   # Matrix representation tests
├── LICENSE
└── README.md
```

## Running Tests

```bash
python -m pytest src/
# or
python -m unittest discover src/
```

## Dependencies

- Python 3.x
- NumPy

## License

This project is released into the public domain under the [Unlicense](LICENSE).

## Mathematical Background

This library implements the **symmetric group Sₙ**, the set of all bijections from {1, …, n} to itself under function composition. Key concepts:

- **Cycle notation**: every permutation factors uniquely into disjoint cycles
- **Sign (signature)**: each cycle of length k contributes k−1 transpositions; the sign is (−1) raised to the total count
- **Alternating group Aₙ**: the index-2 subgroup of even permutations
- **Conjugacy classes**: two permutations are conjugate iff they have the same cycle type
- **Permutation matrices**: n×n binary matrices with exactly one 1 per row and column; M⁻¹ = Mᵀ
- **Group homomorphism**: the map σ ↦ M(σ) satisfies M(σ · τ) = M(σ) @ M(τ)
