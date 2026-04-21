# snkspectra

A WORK IN PROGRESS, vibe-coded, Python library for working with permutations and their representations in group theory.

## Overview

Python code for noodling around with Sn mod Sk, or "Snk" for short. Snk models k choices out of n, e.g., the top 5 NBA players of all time. The library supports multiple standard notations, algebraic operations, cycle decomposition, sign computation, matrix representations, the full symmetric group as a first-class object, all irreducible representations via Young's orthogonal form, and Fourier analysis on S_n.

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
- **All irreducible representations**: `irrep(λ)` constructs every irrep of S_n via Young's orthogonal form
- **Discrete Fourier transform**: `sn_dft` computes f̂(ρ_λ) = Σ f(σ) ρ_λ(σ) for all partitions λ ⊢ n
- **Fast Fourier transform**: `sn_fft` implements Clausen's O(n² · n!) algorithm (vs O((n!)²) naive)
- **Coset FFT**: `sn_mod_sk_fft` computes the FFT of right-S_{n-k}-invariant functions in O(n!/(n-k)!) domain sweeps
- **Inverse DFT**: `sn_idft` recovers f from its Fourier coefficients
- **Plancherel identity**: `plancherel_norm_sq` verifies Σ|f|² = (1/n!) Σ d_λ ‖f̂(λ)‖²_F
- **Convolution theorem**: `sn_convolve` computes group convolution via pointwise matrix product of transforms
- **Coset projection matrices**: `snk_projection` computes the orthogonal projection P_λ onto the S_{n-k}-fixed subspace of each irrep
- **Specht module irreps**: `irrep_specht` constructs irreps via Gram-Schmidt orthogonalization of polytabloids; equivalent to Young's orthogonal form
- **DFT-basis representation**: `irrep_dft` constructs the n-dimensional unitary representation ρ(σ) = F_n · P_σ · F_n†
- **Sign-normalized irreps**: `irrep_sign` builds sign-pattern variants of Young's orthogonal form (exploratory/non-homomorphic)
- **Character table**: `character_table(n)` returns the full character table of S_n as an integer NumPy matrix via the Murnaghan-Nakayama rule

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
sigma.sign()      # +1  (one 3-cycle + one 2-cycle = 2+1 transpositions → even)
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

### Matrix Representations

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

### All Irreducible Representations (Young's Orthogonal Form)

```python
from src import irrep, partitions_of

# All partitions of n index the irreps of S_n
partitions_of(4)   # [(4,), (3,1), (2,2), (2,1,1), (1,1,1,1)]

# Get the irrep for a partition
rho = irrep((3, 1))          # returns a function Permutation → np.ndarray
sigma = Permutation([2, 3, 1, 4])
rho(sigma)                   # 3×3 orthogonal matrix (Young's orthogonal form)

# Verify it's a homomorphism
rho(sigma * tau) ≈ rho(sigma) @ rho(tau)   # True
```

### Fourier Transform on S_n

```python
from src import sn_dft, sn_idft, sn_fft, plancherel_norm_sq, sn_convolve
from src import SymmetricGroup

S4 = SymmetricGroup(4)
f = {g: float(g.sign()) for g in S4}   # sign function

# Naive DFT: O((n!)²)
f_hat = sn_dft(f, 4)    # {partition: d_λ × d_λ matrix}

# Fast FFT (Clausen): O(n² · n!) — same output, much faster for large n
f_hat = sn_fft(f, 4)

# Inverse: recover f from its Fourier coefficients
f_rec = sn_idft(f_hat, 4)   # {Permutation: value}

# Plancherel identity: Σ|f(g)|² = (1/n!) Σ d_λ ‖f̂(λ)‖²_F
plancherel_norm_sq(f_hat, 4)   # equals sum(v**2 for v in f.values())

# Group convolution via FFT
h = {g: 1.0 for g in S4}
conv = sn_convolve(f, h, 4)   # (f * h)(g) = Σ_x f(x) h(x⁻¹g)
```

### Coset FFT (S_n / S_{n-k}-invariant functions)

```python
from src import sn_mod_sk_fft, to_coset_function, from_coset_function
from src import SymmetricGroup

# A right-S_{n-k}-invariant function depends only on (σ(n-k+1),...,σ(n)).
# Represent it as a dict keyed by k-tuples of distinct elements from {1,...,n}.

n, k = 5, 2
# f_tilde[(a1, a2)] = value where (a1, a2) = (σ(4), σ(5))
f_tilde = {(a1, a2): float(a1 - a2)
           for a1 in range(1, n+1) for a2 in range(1, n+1) if a1 != a2}

# FFT: sweeps only n!/(n-k)! = 20 coset representatives (vs 120 group elements)
f_hat = sn_mod_sk_fft(f_tilde, n, k)   # {partition λ ⊢ 5: matrix}

# Convert to/from full S_n functions
f_full = from_coset_function(f_tilde, n, k)   # {Permutation: value}
f_tilde2 = to_coset_function(f_full, n, k)    # back to k-tuple dict
```

### Coset Projection Matrices

```python
from src.snk_projection import snk_projection, projection_summary, apply_projection

# Projection matrices P_λ onto the S_{n-k}-fixed subspace of each irrep V_λ
# P_λ = (1/|S_{n-k}|) Σ_{τ ∈ S_{n-k}} ρ_λ(τ)
projections = snk_projection(5, 2)   # {partition λ ⊢ 5: d_λ × d_λ matrix}

# Summary: projection + rank + dim for each λ
summary = projection_summary(5, 2)
# summary[λ] = {'P': matrix, 'rank': int, 'dim': int}
# rank = dim of S_{n-k}-fixed subspace; Σ d_λ · rank(P_λ) = n!/(n-k)!

# Project existing DFT coefficients onto the invariant subspace
f_hat_proj = apply_projection(f_hat, 5, 2)   # {λ: f̂(λ) · P_λ}
```

### Specht Module Irreps

```python
from src.irreps_specht import irrep_specht, specht_basis

# Irrep via Gram-Schmidt orthogonalization of Specht module polytabloids
# Equivalent to Young's orthogonal form (same characters, orthogonal change of basis)
rho = irrep_specht((3, 2))          # returns Permutation → np.ndarray
sigma = Permutation([2, 3, 1, 4, 5])
rho(sigma)                          # 5×5 orthogonal matrix

# Inspect the Specht basis (orthonormal rows in the tabloid inner product)
B, tabloids, tab_index = specht_basis((3, 2))
# B.shape = (dim, num_tabloids);  B @ B.T = I_dim
```

### DFT-Basis Representation

```python
from src.irreps_dft import irrep_dft, dft_matrix, perm_matrix

# n-dimensional unitary representation: ρ(σ) = F_n · P_σ · F_n†
rho = irrep_dft(5)                  # returns Permutation → np.ndarray (5×5 complex)
sigma = Permutation([2, 3, 1, 4, 5])
rho(sigma)                          # 5×5 unitary matrix

# Cyclic shift diagonalizes to DFT eigenvalues: ρ(c) = diag(ω^0, ω^{-1}, …, ω^{-(n-1)})
# Access precomputed s_i matrices and the DFT matrix
rho._dft_s_matrix(1)    # DFT-basis matrix for s_1 = (1 2)
rho.F                   # normalized n×n DFT matrix F_n

# Low-level helpers
F = dft_matrix(5)       # 5×5 DFT matrix (unitary)
P = perm_matrix(sigma)  # 5×5 permutation matrix
```

### Character Table (Murnaghan-Nakayama Rule)

```python
from src.character_table_Sn import character_table

# Returns an integer NumPy matrix for S_n
# Rows = irreps, columns = conjugacy classes (both indexed by partitions of n
# in decreasing lex order, with columns reversed so identity class comes first)
X = character_table(4)
# array([[ 1,  1,  1,  1,  1],
#        [ 3,  1, -1,  0, -1],
#        [ 2,  0,  2, -1,  0],
#        [ 3, -1, -1,  0,  1],
#        [ 1, -1,  1,  1, -1]])

# Column orthogonality: X.T @ X is diagonal with centralizer orders on the diagonal
D = X.T @ X   # diag = [24, 4, 8, 3, 4] for S_4 (centralizer |C(μ)| per class)
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
│   ├── __init__.py                  # Exports all public symbols
│   ├── permutations.py              # Permutation class
│   ├── symmetric_group.py           # SymmetricGroup class
│   ├── representations.py           # perm_mat_rep, trivial_rep, sign_rep, standard_rep, character
│   ├── cayley_distance.py           # cayley_distance, hamming_distance
│   ├── irreps.py                    # irrep(), partitions_of() — all irreps via Young's orthogonal form
│   ├── irreps_specht.py             # irrep_specht() — irreps via Gram-Schmidt on Specht module
│   ├── irreps_dft.py                # irrep_dft() — DFT-basis unitary representation ρ(σ)=F·P_σ·F†
│   ├── irreps_sign.py               # irrep_sign() — sign-normalized variant of Young's orthogonal form
│   ├── fourier.py                   # sn_dft, sn_idft, plancherel_norm_sq, sn_convolve
│   ├── fft.py                       # sn_fft — Clausen's O(n² · n!) FFT
│   ├── coset_fft.py                 # sn_mod_sk_fft — S_n/S_{n-k} coset FFT
│   ├── snk_projection.py            # snk_projection, projection_summary, apply_projection
│   ├── character_table_Sn.py        # character_table(n) — full character table as integer NumPy matrix
│   ├── demo_sn_irreps.py            # Demo: irreps of a random permutation
│   ├── print_s_matrices.py          # Demo: print S matrices for a given partition
│   ├── test_permutations.py         # Permutation tests
│   ├── test_symmetric_group.py      # SymmetricGroup tests
│   ├── test_representations.py      # Representation and distance tests
│   ├── test_irreps.py               # Irrep tests (dimension sum rule, character orthogonality)
│   ├── test_irreps_specht.py        # Specht irrep tests (orthogonality, equivalence with YOF)
│   ├── test_fourier.py              # DFT tests (Plancherel, inversion, convolution, shift)
│   ├── test_fft.py                  # FFT tests (agreement with DFT + algebraic properties)
│   └── test_coset_fft.py            # Coset FFT tests
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
- **Irreducible representations (irreps)**: indexed by partitions λ ⊢ n; constructed in Young's orthogonal form using standard Young tableaux (SYT)
- **Young's orthogonal form**: orthogonal matrices where adjacent transpositions act by axial-distance formulas; any permutation is a product of s_i matrices
- **Branching rule**: ρ_λ|_{S_{n-1}} = ⊕_{μ ≺ λ} ρ_μ (restriction decomposes by removing one box from the Young diagram)
- **Characters**: χ_ρ(σ) = tr(ρ(σ)); class functions that determine the representation up to isomorphism
- **Character orthogonality**: ⟨χ_λ, χ_μ⟩ = (1/n!) Σ_σ χ_λ(σ)χ_μ(σ) = δ_{λμ} for irreducible λ, μ
- **Dimension sum rule**: Σ_{λ ⊢ n} (dim λ)² = n!
- **Standard representation**: the (n−1)-dimensional complement of the trivial subspace in the permutation representation; satisfies χ_std = χ_perm − χ_trivial
- **Non-abelian Fourier transform**: f̂(ρ_λ) = Σ_{σ ∈ Sₙ} f(σ) ρ_λ(σ); a d_λ × d_λ matrix per partition
- **Plancherel / Parseval**: Σ|f(σ)|² = (1/n!) Σ_λ d_λ ‖f̂(λ)‖²_F
- **Inverse DFT**: f(σ) = (1/n!) Σ_λ d_λ tr(f̂(λ) · ρ_λ(σ⁻¹))
- **Clausen FFT**: O(n² · n!) algorithm exploiting the subgroup chain S₁ ⊂ … ⊂ Sₙ and the branching rule; speedup n!/n² over naive
- **Coset FFT**: right-S_{n-k}-invariant functions depend only on (σ(n-k+1),…,σ(n)); FFT runs in O(n!/(n-k)!) domain sweeps using k levels of Clausen decomposition
- **Cayley distance**: d(σ, τ) = n − c(σ⁻¹τ), where c(π) is the number of cycles of π
- **Hamming distance**: d_H(σ, τ) = #{j : σ(j) ≠ τ(j)}; equals ½‖M(σ) − M(τ)‖²_F
- **Permutation module M^λ**: basis = tabloids of shape λ (row-unordered fillings); dimension = n! / (λ₁! · λ₂! · … · λ_k!)
- **Polytabloid**: e_T = Σ_{π ∈ C_T} sgn(π) · {π·T}, where C_T is the column stabilizer of T
- **Specht module S^λ**: submodule of M^λ spanned by polytabloids {e_T : T standard}; Gram-Schmidt yields an orthonormal basis equivalent to Young's orthogonal form
- **Projection onto S_{n-k}-fixed subspace**: P_λ = (1/|S_{n-k}|) Σ_{τ ∈ S_{n-k}} ρ_λ(τ); orthogonal projection with rank equal to multiplicity of trivial S_{n-k}-rep in ρ_λ|_{S_{n-k}}; satisfies Σ_λ d_λ · rank(P_λ) = n!/(n-k)!
- **Murnaghan-Nakayama rule**: χ^λ(μ) = Σ (−1)^{height(S)} χ^{λ∖S}(μ′), summing over border strips S of size μ₁; a border strip is a connected skew shape with no 2×2 block, and height = rows spanned − 1
- **Column orthogonality**: Σ_λ χ^λ(μ) χ^λ(ν) = |C(μ)| δ_{μν}, i.e. X^T X = diag of centralizer orders
- **DFT-basis representation**: ρ(σ) = F_n · P_σ · F_n†; unitary n-dimensional representation where the cyclic shift diagonalizes to diag(ω⁰, ω⁻¹, …, ω^{-(n-1)}); decomposes as trivial ⊕ standard

## References

* Horace Pan, https://github.com/horacepan/snpy
* Horace Pan, https://github.com/horacepan/SnFFT
* Risi Kondor, https://github.com/risi-kondor/Snob2
* Risi Kondor, https://people.cs.uchicago.edu/~risi/SnOB/index.html
* Clausen, "Fast generalized Fourier transforms" (1989)
* Maslen & Rockmore, "Generalized FFTs — a survey of applications" (1997)
* Diaconis, "Group Representations in Probability and Statistics" (1988)
* Sagan, "The Symmetric Group" (2001)
