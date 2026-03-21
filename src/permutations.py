"""
Vibe coded (Claude)
Permutation class supporting one-line and two-line notation,
multiplication (composition), and inversion.

Permutations act on {1, 2, ..., n} and are stored internally
as a dict mapping each element to its image.
"""


class Permutation:
    def __init__(self, mapping):
        """
        Create a permutation from a dict {i: sigma(i)} or a list/tuple
        representing one-line notation [sigma(1), sigma(2), ..., sigma(n)].
        """
        if isinstance(mapping, (list, tuple)):
            self._map = {i + 1: v for i, v in enumerate(mapping)}
        elif isinstance(mapping, dict):
            self._map = dict(mapping)
        else:
            raise TypeError("mapping must be a dict, list, or tuple")

        self._validate()

    def _validate(self):
        n = len(self._map)
        domain = set(self._map.keys())
        codomain = set(self._map.values())
        expected = set(range(1, n + 1))
        if domain != expected or codomain != expected:
            raise ValueError(
                f"Not a valid permutation on {{1..{n}}}: "
                f"got domain={sorted(domain)}, image={sorted(codomain)}"
            )

    @property
    def n(self):
        return len(self._map)

    # ------------------------------------------------------------------
    # Notation conversions
    # ------------------------------------------------------------------

    def to_one_line(self):
        """Return the one-line notation as a list [sigma(1), ..., sigma(n)]."""
        return [self._map[i] for i in range(1, self.n + 1)]

    def to_two_line(self):
        """
        Return the two-line notation as a pair of lists:
            (top_row, bottom_row)
        where top_row = [1, 2, ..., n] and bottom_row = [sigma(1), ..., sigma(n)].
        """
        top = list(range(1, self.n + 1))
        bottom = self.to_one_line()
        return top, bottom

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def __mul__(self, other):
        """
        Compose two permutations: (self * other)(x) = self(other(x)).
        Both permutations must have the same degree n.
        """
        if not isinstance(other, Permutation):
            return NotImplemented
        if self.n != other.n:
            raise ValueError(
                f"Cannot compose permutations of different degrees ({self.n} vs {other.n})"
            )
        composed = {i: self._map[other._map[i]] for i in range(1, self.n + 1)}
        return Permutation(composed)

    def inverse(self):
        """Return the inverse permutation sigma^{-1}."""
        inv = {v: k for k, v in self._map.items()}
        return Permutation(inv)

    def __pow__(self, exponent):
        """Return self ** exponent (supports negative integers via inverse)."""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")
        if exponent == 0:
            return Permutation.identity(self.n)
        base = self if exponent > 0 else self.inverse()
        result = Permutation.identity(self.n)
        for _ in range(abs(exponent)):
            result = result * base
        return result

    def cycles(self):
        """
        Return the disjoint cycle decomposition as a list of lists.
        Fixed points are excluded. Each cycle starts at its smallest element.

        Example: Permutation([2, 3, 1, 5, 4]).cycles() == [[1, 2, 3], [4, 5]]
        """
        visited = set()
        result = []
        for start in range(1, self.n + 1):
            if start in visited:
                continue
            cycle = []
            x = start
            while x not in visited:
                visited.add(x)
                cycle.append(x)
                x = self._map[x]
            if len(cycle) > 1:
                result.append(cycle)
        return result

    def sign(self):
        """
        Return the sign (signature) of the permutation: +1 if even, -1 if odd.
        Each cycle of length k contributes k-1 transpositions.
        """
        parity = 1
        for cycle in self.cycles():
            parity *= (-1) ** (len(cycle) - 1)
        return parity

    def __eq__(self, other):
        if not isinstance(other, Permutation):
            return NotImplemented
        return self._map == other._map

    def __hash__(self):
        return hash(tuple(sorted(self._map.items())))

    def __call__(self, x):
        """Evaluate the permutation at a point: sigma(x)."""
        if x not in self._map:
            raise ValueError(f"{x} is not in the domain {{1..{self.n}}}")
        return self._map[x]

    # ------------------------------------------------------------------
    # Class-level constructors
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, n):
        """Return the identity permutation on {1, ..., n}."""
        return cls({i: i for i in range(1, n + 1)})

    @classmethod
    def from_two_line(cls, top, bottom):
        """
        Construct a permutation from two-line notation.
            top    = [1, 2, ..., n]  (can be any ordering)
            bottom = [sigma(top[0]), sigma(top[1]), ...]
        """
        if len(top) != len(bottom):
            raise ValueError("Top and bottom rows must have the same length")
        return cls(dict(zip(top, bottom)))

    @classmethod
    def from_cycles(cls, n, *cycles):
        """
        Construct a permutation on {1..n} from disjoint cycle notation.
        Example: Permutation.from_cycle(5, [1, 3, 5], [2, 4])
        """
        mapping = {i: i for i in range(1, n + 1)}
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            for i, elem in enumerate(cycle):
                mapping[elem] = cycle[(i + 1) % len(cycle)]
        return cls(mapping)

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"Permutation({self.to_one_line()})"

    def __str__(self):
        return self.one_line_str()

    def one_line_str(self):
        """Pretty one-line notation: (2 3 1 4)"""
        return "(" + " ".join(str(v) for v in self.to_one_line()) + ")"

    def two_line_str(self):
        """Pretty two-line notation aligned by column width."""
        top, bottom = self.to_two_line()
        col_widths = [
            max(len(str(t)), len(str(b))) for t, b in zip(top, bottom)
        ]
        top_str = "  ".join(str(t).rjust(w) for t, w in zip(top, col_widths))
        bot_str = "  ".join(str(b).rjust(w) for b, w in zip(bottom, col_widths))
        bar = "-" * len(top_str)
        return f"( {top_str} )\n( {bot_str} )"

    
    def cycle_str(self):
        """Express the permutation in disjoint cycle notation."""
        cycles = self.cycles()
        return "".join("(" + " ".join(str(v) for v in c) + ")" for c in cycles) if cycles else "e"

    def print_all(self):
        """Print all representations of the permutation."""
        print(f"One-line  : {self.one_line_str()}")
        print(f"Two-line  :\n{self.two_line_str()}")
        print(f"Cycle     : {self.cycle_str()}")


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sigma = Permutation([2, 3, 1, 4, 5])
    tau   = Permutation([1, 2, 4, 5, 3])

    print("=== sigma ===")
    sigma.print_all()

    print("\n=== tau ===")
    tau.print_all()

    print("\n=== sigma * tau ===")
    (sigma * tau).print_all()

    print("\n=== sigma^{-1} ===")
    sigma.inverse().print_all()

    print("\n=== sigma^3 ===")
    (sigma ** 3).print_all()

    print("\n=== from_cycle: (1 3 5)(2 4) on n=5 ===")
    Permutation.from_cycles(5, [1, 3, 5], [2, 4]).print_all()

    print("\n=== from_two_line ===")
    p = Permutation.from_two_line([1, 2, 3, 4, 5], [3, 1, 4, 5, 2])
    p.print_all()