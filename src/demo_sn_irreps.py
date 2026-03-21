"""
Vibe coded (Claude)
Pick a random permutation in S(n) and print its representation matrix
for every irreducible representation (one per partition of n).
"""

import random
import numpy as np
from symmetric_group import SymmetricGroup
from irreps import partitions_of, irrep

N = 5   # <-- change this to use a different symmetric group

rng = random.Random()
Sn = SymmetricGroup(N)
sigma = rng.choice(Sn.elements())

print(f"S({N})")
print(f"Permutation : {sigma.one_line_str()}")
print(f"Cycle form  : {sigma.cycle_str() or 'e'}")
print(f"Sign        : {sigma.sign():+d}")
print()

for lam in partitions_of(N):
    if lam != (4,1):
        continue
    rho = irrep(lam)
    M = rho(sigma)
    dim = M.shape[0]
    chi = np.trace(M).real
    print(f"λ = {lam}   dim = {dim}   χ(σ) = {chi:+.4g}")
    for row in M:
        print("  " + "  ".join(f"{v:8.4f}" for v in row))
    print()
