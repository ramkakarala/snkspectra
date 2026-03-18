# snkspectra

A WORK IN PROGRESS, vibe-coded, Python library for working with permutations and their representations in group theory.  

## Overview

Python code for noodling around with Sn mod Sk, or "Snk" for short. Snk models k choices out of n, e.g., the top 5 NBA players of all time. The library supports multiple standard notations, algebraic operations, cycle decomposition, sign computation, matrix representations, and the full symmetric group as a first-class object.

## Features

- **Multiple input formats**: dict mapping, one-line notation (list/tuple), cycle notation, two-line notation
- **Notation conversions**: one-line, two-line, and disjoint cycle notation
- **Algebraic operations**: composition, inversion, exponentiation (including negative powers)
- **Cycle decomposition**: `cycles()` returns the disjoint cycle structure as a list of lists
- **Sign (signature)**: `sign()` returns +1 (even) or -1 (odd)
- **Symmetric group**: `SymmetricGroup(n)` enumerates all n! elements with group-theoretic queries
- **Matrix representation**: converts permutations to binary permutation matrices
- **Homomorphism verification**: M(σ · τ) = M(σ) @ M(τ)
- **Trivial representation**: the 1×1 identity representation
- **Sign representation**: the 1×1 representation mapping even/odd permutations to ±1
- **Standard representation**: the irreducible (n−1)-dimensional subrepresentation
- **Characters**: trace of any representation matrix; satisfies χ_std = χ_perm − χ_trivial
- **Distance functions**: Cayley distance (minimum transpositions) and Hamming distance (disagreeing positions)

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

### Representations

```python
from src import perm_mat_rep, trivial_rep, sign_rep, standard_rep, character

sigma = Permutation([2, 3, 1, 4, 5])   # (1 2 3)
tau   = Permutation([2, 1, 3, 4, 5])   # (1 2)

# Permutation matrix representation (n×n)
M = perm_mat_rep(sigma)
np.array_equal(perm_mat_rep(sigma * tau), perm_mat_rep(sigma) @ perm_mat_rep(tau))  # True

# Trivial representation (1×1, always [[1]])
trivial_rep(sigma)   # array([[1]])

# Sign representation (1×1, [[+1]] or [[-1]])
sign_rep(sigma)      # array([[1]])   — (1 2 3) is even
sign_rep(tau)        # array([[-1]])  — (1 2) is odd

# Standard representation ((n-1)×(n-1), irreducible)
standard_rep(sigma)  # 4×4 orthogonal matrix

# Characters (trace of the representation matrix)
character(sigma, trivial_rep)   # 1
character(sigma, sign_rep)      # 1  (even permutation)
character(sigma, perm_mat_rep)  # 2  (number of fixed points)
character(sigma, standard_rep)  # 1  (fixed points − 1)
```

### Distance Functions

```python
from src.cayley_distance import cayley_distance, hamming_distance

sigma = Permutation([2, 3, 1, 4, 5])
tau   = Permutation([2, 1, 3, 4, 5])

cayley_distance(sigma, tau)   # minimum transpositions to transform sigma into tau
hamming_distance(sigma, tau)  # number of positions where sigma and tau disagree
```

## Project Structure

```
snkspectra/
├── src/
│   ├── __init__.py                  # Exports Permutation, SymmetricGroup, all representations
│   ├── permutations.py              # Permutation class
│   ├── symmetric_group.py           # SymmetricGroup class
│   ├── representations.py           # perm_mat_rep, trivial_rep, sign_rep, standard_rep, character
│   ├── cayley_distance.py           # cayley_distance, hamming_distance
│   ├── test_permutations.py         # Permutation tests
│   ├── test_symmetric_group.py      # SymmetricGroup tests
│   └── test_representations.py      # Representation and distance tests
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
- **Representations**: group homomorphisms ρ: Sₙ → GL(V); the trivial, sign, and standard representations are all irreducible
- **Characters**: χ_ρ(σ) = tr(ρ(σ)); class functions that determine the representation up to isomorphism
- **Character orthogonality**: ⟨χ_ρ, χ_ρ'⟩ = (1/n!) Σ_σ χ_ρ(σ)χ_ρ'(σ) = δ_{ρ,ρ'} for irreducible ρ, ρ'
- **Standard representation**: the (n−1)-dimensional complement of the trivial subspace in the permutation representation; satisfies χ_std = χ_perm − χ_trivial
- **Cayley distance**: d(σ, τ) = n − c(σ⁻¹τ), where c(π) is the number of cycles of π
- **Hamming distance**: d_H(σ, τ) = #{j : σ(j) ≠ τ(j)}; equals ½‖M(σ) − M(τ)‖²_F

## References

* Horace Pan, https://github.com/horacepan/snpy
* Risi Kondor, https://github.com/risi-kondor/Snob2
* Risi Kondor, https://people.cs.uchicago.edu/~risi/SnOB/index.html

