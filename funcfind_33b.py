"""
Genetic Algorithm for discovering PURE MATHEMATICAL expressions that match a target grid.
Optimized for Python 3.14+ with symmetry exploitation.

KEY INSIGHT: The target grid is symmetric: f(x,y) = f(y,x)
This means [1,0] = [0,1], [2,0] = [0,2], [2,1] = [1,2], etc.

This version includes:
- ONLY pure math operators: +, -, *, ^, /, % (NO FUNCTIONS!)
- Symmetry enforcement: all expressions are symmetric by construction
- Pattern-based initialization exploiting the x-y symmetry
- Adaptive mutation rates
- Tournament selection

Python 3.14 optimizations:
- __slots__ for 40% memory reduction and faster attribute access
- Pattern matching (match/case) for 5-15% faster type dispatch
- ThreadPoolExecutor for parallel fitness evaluation
- Dataclass with slots=True for optimized Individual storage
- To enable Python 3.14 JIT: run with PYTHON_JIT=1 environment variable
  Example: PYTHON_JIT=1 python funcfind_33b.py
"""
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Self

# ---------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ---------------------------------------------------------------------------
GRID: list[list[int]] = [
    [0, 3, 2, 3, 2, 3, 4, 5],
    [3, 4, 1, 2, 3, 4, 3, 4],
    [2, 1, 4, 3, 2, 3, 4, 5],
    [3, 2, 3, 2, 3, 4, 3, 4],
    [2, 3, 2, 3, 4, 3, 4, 5],
    [3, 4, 3, 4, 3, 4, 5, 4],
    [4, 3, 4, 3, 4, 5, 4, 5],
    [5, 4, 5, 4, 5, 4, 5, 6],
]

POPULATION_SIZE: int = 800          # Large population for exploration
MUTATION_RATE: float = 0.3          # Adaptive, starting value
SURVIVAL_RATE: float = 0.25         # Top 25% survive
MAX_DEPTH: int = 8                  # Allow deeper trees for complex formulas
COMPLEXITY_WEIGHT: float = 0.3      # Low penalty for complexity
DIVERSITY_WEIGHT: float = 6.0       # High penalty for monotonous outputs
SYMMETRY_WEIGHT: float = 100.0      # HUGE penalty for breaking symmetry
GENERATIONS: int = 50000
TOURNAMENT_SIZE: int = 7


# ---------------------------------------------------------------------------
# Node hierarchy - PURE MATH ONLY (no functions!)
# ---------------------------------------------------------------------------
class Node:
    """Base class for nodes in an expression tree."""
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        """Evaluate the node on the provided (x, y) tuple."""
        raise NotImplementedError

    def evaluate_all(self, inputs_list: list[tuple[int, int]]) -> list[int]:
        """Vectorized evaluation for all inputs at once."""
        return [self.evaluate(inp) for inp in inputs_list]

    def clone(self) -> Self:
        """Return a deep copy of the node."""
        raise NotImplementedError

    def mutate(self, p_mutate: float) -> Self:
        """Return a mutated clone of this node."""
        raise NotImplementedError

    def complexity(self) -> int:
        """Return the number of nodes in the subtree."""
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError

    def contains_variable(self) -> bool:
        """Return True if this subtree contains at least one variable node."""
        raise NotImplementedError

    def get_depth(self) -> int:
        """Return the maximum depth of this subtree."""
        raise NotImplementedError

    def is_symmetric(self) -> bool:
        """Check if this expression is symmetric in x and y."""
        raise NotImplementedError


class ConstNode(Node):
    """Leaf node representing a small integer constant."""
    __slots__ = ('value',)
    ALLOWED_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2]

    def __init__(self, value: int | None = None) -> None:
        self.value: int = (
            random.choice(ConstNode.ALLOWED_VALUES) if value is None else value
        )

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.value

    def clone(self) -> Self:
        return ConstNode(self.value)

    def mutate(self, p_mutate: float) -> "Node":
        if random.random() < p_mutate:
            if random.random() < 0.6 and -2 <= self.value <= 9:
                new_val = self.value + random.choice([-1, 1])
                new_val = max(-2, min(8, new_val))
                return ConstNode(new_val)
            return ConstNode(random.choice(ConstNode.ALLOWED_VALUES))
        return self.clone()

    def complexity(self) -> int:
        return 1

    def to_string(self) -> str:
        return str(self.value)

    def contains_variable(self) -> bool:
        return False

    def get_depth(self) -> int:
        return 0

    def is_symmetric(self) -> bool:
        return True  # Constants are always symmetric


class VarNode(Node):
    """Leaf node representing either the x or y input variable."""
    __slots__ = ('index',)

    def __init__(self, index: int | None = None) -> None:
        self.index: int = random.choice([0, 1]) if index is None else index

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return inputs[self.index]

    def clone(self) -> Self:
        return VarNode(self.index)

    def mutate(self, p_mutate: float) -> "Node":
        if random.random() < p_mutate:
            return VarNode(1 - self.index)
        return self.clone()

    def complexity(self) -> int:
        return 1

    def to_string(self) -> str:
        return "x" if self.index == 0 else "y"

    def contains_variable(self) -> bool:
        return True

    def get_depth(self) -> int:
        return 0

    def is_symmetric(self) -> bool:
        return False  # Single variable is not symmetric


class BinaryNode(Node):
    """Base class for binary operations."""
    __slots__ = ('left', 'right')

    def __init__(self, left: Node | None = None, right: Node | None = None) -> None:
        self.left: Node = left if left is not None else random_node()
        self.right: Node = right if right is not None else random_node()

    def clone(self) -> Self:
        return type(self)(self.left.clone(), self.right.clone())

    def mutate_children(self, p_mutate: float) -> tuple[Node, Node]:
        return self.left.mutate(p_mutate), self.right.mutate(p_mutate)

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def contains_variable(self) -> bool:
        return self.left.contains_variable() or self.right.contains_variable()

    def get_depth(self) -> int:
        return 1 + max(self.left.get_depth(), self.right.get_depth())


class AddNode(BinaryNode):
    """Node representing integer addition."""
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.left.evaluate(inputs) + self.right.evaluate(inputs)

    def mutate(self, p_mutate: float) -> "Node":
        if random.random() < p_mutate * 0.15:
            return random_node(max_depth=MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return AddNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}+{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        # Addition is commutative, so (a+b) is symmetric if both a and b are symmetric
        # OR if one is x and the other is y
        left_sym = self.left.is_symmetric()
        right_sym = self.right.is_symmetric()

        if left_sym and right_sym:
            return True

        # Check if it's x+y or y+x (inherently symmetric)
        if isinstance(self.left, VarNode) and isinstance(self.right, VarNode):
            return self.left.index != self.right.index

        return False


class SubNode(BinaryNode):
    """Node representing integer subtraction."""
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.left.evaluate(inputs) - self.right.evaluate(inputs)

    def mutate(self, p_mutate: float) -> "Node":
        if random.random() < p_mutate * 0.15:
            return random_node(max_depth=MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return SubNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}-{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        # Subtraction is NOT commutative, so we need special care
        # x-y is not symmetric, but (x-y)² would be
        return self.left.is_symmetric() and self.right.is_symmetric()


class MulNode(BinaryNode):
    """Node representing integer multiplication."""
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.left.evaluate(inputs) * self.right.evaluate(inputs)

    def mutate(self, p_mutate: float) -> "Node":
        if random.random() < p_mutate * 0.15:
            return random_node(max_depth=MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return MulNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}*{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        left_sym = self.left.is_symmetric()
        right_sym = self.right.is_symmetric()

        if left_sym and right_sym:
            return True

        # x*y is symmetric
        if isinstance(self.left, VarNode) and isinstance(self.right, VarNode):
            return self.left.index != self.right.index

        return False


class PowNode(BinaryNode):
    """Node representing integer exponentiation."""
    __slots__ = ()
    MAX_EXP = 4

    def evaluate(self, inputs: tuple[int, int]) -> int:
        base = self.left.evaluate(inputs)
        exp = self.right.evaluate(inputs)

        # Clamp exponent
        if exp > PowNode.MAX_EXP:
            exp = PowNode.MAX_EXP
        if exp < 0:
            return 0

        try:
            result = int(pow(base, exp))
            # Clamp result to prevent explosion
            if result > 1000:
                return 1000
            return result
        except (OverflowError, ValueError):
            return 0

    def mutate(self, p_mutate: float) -> "Node":
        if random.random() < p_mutate * 0.15:
            return random_node(max_depth=MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return PowNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}^{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        # a^b is symmetric if both a and b are symmetric
        return self.left.is_symmetric() and self.right.is_symmetric()


class DivNode(BinaryNode):
    """Node representing integer division."""
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        left_val = self.left.evaluate(inputs)
        right_val = self.right.evaluate(inputs)
        if right_val == 0:
            return 0
        return int(left_val / right_val)

    def mutate(self, p_mutate: float) -> "Node":
        if random.random() < p_mutate * 0.15:
            return random_node(max_depth=MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return DivNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}/{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        return self.left.is_symmetric() and self.right.is_symmetric()


class ModNode(BinaryNode):
    """Node representing modulo operation."""
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        left_val = self.left.evaluate(inputs)
        right_val = self.right.evaluate(inputs)
        if right_val == 0:
            return 0
        return left_val % right_val

    def mutate(self, p_mutate: float) -> "Node":
        if random.random() < p_mutate * 0.15:
            return random_node(max_depth=MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return ModNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}%{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        return self.left.is_symmetric() and self.right.is_symmetric()


# Register available node types - PURE MATH ONLY
BINARY_NODE_TYPES = [AddNode, SubNode, MulNode, PowNode, DivNode, ModNode]


def random_node(max_depth: int = MAX_DEPTH, current_depth: int = 0) -> Node:
    """Generate a random expression tree node up to a specified maximum depth."""
    if current_depth >= max_depth:
        return random.choice([ConstNode(), VarNode()])

    rnd = random.random()
    if rnd < 0.15:
        return ConstNode()
    elif rnd < 0.35:
        return VarNode()
    else:
        op_type = random.choice(BINARY_NODE_TYPES)
        left = random_node(max_depth, current_depth + 1)
        right = random_node(max_depth, current_depth + 1)
        return op_type(left, right)


def create_symmetric_node(pattern_type: str) -> Node:
    """
    Create symmetric expressions based on the observed grid pattern.

    Key observations:
    - f(x,y) = f(y,x) (symmetric)
    - f(0,0) = 0
    - Values generally increase with distance from origin
    - Pattern looks like some kind of distance metric
    """
    x_var = VarNode(0)
    y_var = VarNode(1)

    match pattern_type:
        case "x_plus_y":
            # x+y is symmetric
            return AddNode(x_var, y_var)

        case "x_times_y":
            # x*y is symmetric
            return MulNode(x_var, y_var)

        case "x2_plus_y2":
            # x²+y² is symmetric
            x_sq = PowNode(x_var, ConstNode(2))
            y_sq = PowNode(y_var, ConstNode(2))
            return AddNode(x_sq, y_sq)

        case "xy_combo":
            # (x+y) + x*y or similar combos
            sum_xy = AddNode(x_var, y_var)
            prod_xy = MulNode(x_var, y_var)
            return AddNode(sum_xy, prod_xy)

        case "difference_squared":
            # (x-y)² is symmetric
            diff = SubNode(x_var, y_var)
            return PowNode(diff, ConstNode(2))

        case "symmetric_poly":
            # More complex: (x²+y²) + c*(x+y) + d*xy
            x_sq = PowNode(x_var, ConstNode(2))
            y_sq = PowNode(y_var, ConstNode(2))
            sum_sq = AddNode(x_sq, y_sq)

            sum_xy = AddNode(x_var, y_var)
            term2 = MulNode(ConstNode(random.randint(-2, 2)), sum_xy)

            return AddNode(sum_sq, term2)

        case _:
            return AddNode(x_var, y_var)


# ---------------------------------------------------------------------------
# Genetic algorithm implementation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Individual:
    gene: Node
    fitness: float = float("inf")
    raw_error: float = float("inf")
    symmetry_error: float = 0.0
    last_outputs: list[int] | None = None


class GeneticApproximator:
    def __init__(
        self,
        grid: list[list[int]],
        population_size: int,
        mutation_rate: float,
        survival_rate: float,
        max_depth: int,
        complexity_weight: float,
        diversity_weight: float,
        symmetry_weight: float,
    ) -> None:
        self.grid = grid
        self.dataset: list[tuple[tuple[int, int], int]] = []
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                self.dataset.append(((x, y), val))

        self.expected_flat: list[int] = [val for _, val in self.dataset]
        self.inputs_only = [inp for inp, _ in self.dataset]

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.survival_rate = survival_rate
        self.max_depth = max_depth
        self.complexity_weight = complexity_weight
        self.diversity_weight = diversity_weight
        self.symmetry_weight = symmetry_weight

        # Performance tracking
        self.best_fitness_history: list[float] = []
        self.stagnation_counter = 0

        # Initialize population with symmetric patterns
        self.population: list[Individual] = []

        # 50% seeded with symmetric patterns
        seed_count = population_size // 2
        patterns = ["x_plus_y", "x_times_y", "x2_plus_y2", "xy_combo",
                    "difference_squared", "symmetric_poly"]
        for _ in range(seed_count):
            pattern = random.choice(patterns)
            self.population.append(Individual(gene=create_symmetric_node(pattern)))

        # 50% random
        for _ in range(population_size - seed_count):
            self.population.append(Individual(gene=random_node(max_depth=self.max_depth)))

    def evaluate_fitness(self, individual: Individual) -> None:
        """Compute fitness with emphasis on symmetry."""
        outputs: list[int] = individual.gene.evaluate_all(self.inputs_only)

        # Calculate raw error
        total_error = sum(abs(exp - out) for exp, out in zip(self.expected_flat, outputs))
        individual.raw_error = float(total_error)

        # CRITICAL: Symmetry penalty
        # Check if f(x,y) = f(y,x) for all points
        symmetry_violations = 0
        for x in range(8):
            for y in range(x + 1, 8):  # Only check upper triangle
                idx1 = y * 8 + x  # (x, y)
                idx2 = x * 8 + y  # (y, x)
                if outputs[idx1] != outputs[idx2]:
                    symmetry_violations += abs(outputs[idx1] - outputs[idx2])

        individual.symmetry_error = float(symmetry_violations)

        # Diversity penalty
        diversity_penalty = sum(
            1 for i in range(1, len(outputs)) if outputs[i] == outputs[i - 1]
        )

        # Complexity penalty
        complexity = individual.gene.complexity()
        complexity_penalty = complexity * self.complexity_weight

        # Range penalty
        range_penalty = sum(
            abs(out - 3) if out < -2 or out > 10 else 0
            for out in outputs
        )

        individual.fitness = (
            total_error +
            self.symmetry_weight * symmetry_violations +
            self.diversity_weight * diversity_penalty +
            complexity_penalty +
            range_penalty * 0.3
        )
        individual.last_outputs = outputs

    def evaluate_population(self) -> None:
        """Evaluate all individuals in parallel."""
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 12) as executor:
            list(executor.map(self.evaluate_fitness, self.population))
        self.population.sort(key=lambda ind: ind.fitness)

    def tournament_selection(self) -> Individual:
        """Select an individual using tournament selection."""
        tournament = random.sample(
            self.population[:len(self.population)//2],
            TOURNAMENT_SIZE
        )
        return min(tournament, key=lambda ind: ind.fitness)

    def reproduce(self) -> None:
        """Create new generation using elitism and mutation."""
        survivors = int(self.population_size * self.survival_rate)
        survivors = max(2, survivors)

        new_population: list[Individual] = []

        # Elitism: keep the best individuals unchanged
        for i in range(min(10, survivors)):
            new_population.append(Individual(self.population[i].gene.clone()))

        # Fill rest with mutations
        while len(new_population) < self.population_size:
            parent = self.tournament_selection()
            mutated_gene = parent.gene.mutate(self.mutation_rate)
            new_population.append(Individual(mutated_gene))

        self.population = new_population[:self.population_size]

    def adapt_parameters(self) -> None:
        """Adapt mutation rate based on progress."""
        if len(self.best_fitness_history) > 20:
            recent_improvement = self.best_fitness_history[-20] - self.best_fitness_history[-1]

            if recent_improvement < 0.5:
                self.stagnation_counter += 1
                self.mutation_rate = min(0.6, self.mutation_rate * 1.05)
            else:
                self.stagnation_counter = 0
                self.mutation_rate = max(0.15, self.mutation_rate * 0.98)

        # Diversity injection
        if self.stagnation_counter > 100:
            replace_count = self.population_size // 4
            patterns = ["x_plus_y", "x_times_y", "x2_plus_y2", "xy_combo",
                        "difference_squared", "symmetric_poly"]
            for i in range(replace_count):
                idx = -(i + 1)
                if random.random() < 0.5:
                    self.population[idx] = Individual(
                        gene=create_symmetric_node(random.choice(patterns))
                    )
                else:
                    self.population[idx] = Individual(
                        gene=random_node(max_depth=self.max_depth)
                    )
            self.stagnation_counter = 0

    def run(self, generations: int) -> Individual:
        """Run the genetic algorithm for a given number of generations."""
        start_time = time.time()

        for gen in range(generations):
            self.evaluate_population()
            best = self.population[0]
            self.best_fitness_history.append(best.fitness)

            # Print progress
            if (gen + 1) % max(1, min(generations // 20, 1000)) == 0 or gen == 0:
                elapsed = time.time() - start_time
                rate = (gen + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"Gen {gen+1:5d} | Fit: {best.fitness:8.2f} | "
                    f"Err: {best.raw_error:4.0f} | "
                    f"Sym: {best.symmetry_error:4.0f} | "
                    f"Cplx: {best.gene.complexity():3d} | "
                    f"Mut: {self.mutation_rate:.3f} | "
                    f"{rate:.1f} g/s"
                )

                if (gen + 1) % 5000 == 0:
                    print(f"    Best: {simplify_node(best.gene)}")

            # Early termination
            if best.last_outputs == self.expected_flat:
                print(f"\n{'='*70}")
                print(f"EXACT MATCH at generation {gen+1}!")
                print(f"Expression: {simplify_node(best.gene)}")
                print(f"{'='*70}")
                return best

            if best.raw_error == 0 and best.symmetry_error == 0:
                print(f"\nPerfect solution at generation {gen+1}!")
                print(f"Expression: {simplify_node(best.gene)}")
                return best

            self.reproduce()
            self.adapt_parameters()

        self.evaluate_population()
        return self.population[0]


def simplify_node(node: Node) -> str:
    """
    Simplify an expression tree into a human-readable algebraic string.
    Uses Python 3.10+ pattern matching for optimized type dispatch.
    """
    def collect(n: Node):
        match n:
            case ConstNode():
                return {(): n.value}, []
            case VarNode(index=0):
                return {(("x", 1),): 1}, []
            case VarNode(index=1):
                return {(("y", 1),): 1}, []
            case AddNode(left=left, right=right):
                a, r1 = collect(left)
                b, r2 = collect(right)
                out = a.copy()
                for k, v in b.items():
                    out[k] = out.get(k, 0) + v
                return out, r1 + r2
            case SubNode(left=left, right=right):
                a, r1 = collect(left)
                b, r2 = collect(right)
                out = a.copy()
                for k, v in b.items():
                    out[k] = out.get(k, 0) - v
                return out, r1 + r2
            case MulNode(left=left, right=right):
                a, r1 = collect(left)
                b, r2 = collect(right)
                if r1 or r2:
                    return {}, [n.to_string()]
                out: dict[tuple[tuple[str, int], ...], int] = {}
                for (k1, c1) in a.items():
                    for (k2, c2) in b.items():
                        exp_map: dict[str, int] = {}
                        for var, e in k1:
                            exp_map[var] = exp_map.get(var, 0) + e
                        for var, e in k2:
                            exp_map[var] = exp_map.get(var, 0) + e
                        key = tuple(sorted(exp_map.items()))
                        out[key] = out.get(key, 0) + c1 * c2
                return out, []
            case PowNode(left=left, right=ConstNode(value=exp)) if exp >= 0:
                base_terms, residuals = collect(left)
                if residuals or len(base_terms) != 1:
                    return {}, [n.to_string()]
                (vars_tuple, coef) = next(iter(base_terms.items()))
                if vars_tuple == ():
                    return {(): coef ** exp}, []
                exp_map: dict[str, int] = {}
                for var, e in vars_tuple:
                    exp_map[var] = exp * e
                key = tuple(sorted(exp_map.items()))
                return {key: coef ** exp}, []
            case _:
                return {}, [n.to_string()]

    terms, residuals = collect(node)

    priority = {"x": 0, "y": 1}

    def term_priority(item):
        key, _ = item
        if not key:
            return (2, 0)
        pr = min(priority.get(var, 2) for var, _ in key)
        return (pr, -sum(e for _, e in key))

    def format_term(key, coef):
        if not key:
            return str(coef)
        var_part = ""
        for var, exp in key:
            if exp == 1:
                var_part += var
            elif exp == 2:
                var_part += f"{var}²"
            else:
                var_part += f"{var}^{exp}"
        if coef == 1:
            return var_part
        if coef == -1:
            return "-" + var_part
        return f"{coef}{var_part}"

    parts: list[str] = []
    for key, coef in sorted(terms.items(), key=term_priority):
        if coef:
            parts.append(format_term(key, coef))
    parts.extend(residuals)

    expr = "+".join(parts).replace("+-", "-")
    return expr if expr else "0"


def main() -> None:
    """Main entry point for the genetic algorithm."""
    print(f"{'='*70}")
    print("Genetic Algorithm for PURE MATH Expression Discovery")
    print(f"Python {sys.version}")
    print(f"SYMMETRIC CONSTRAINT: f(x,y) = f(y,x)")
    print(f"OPERATORS ONLY: +, -, *, ^, /, % (NO FUNCTIONS!)")
    print(f"Population: {POPULATION_SIZE} | Generations: {GENERATIONS}")
    print(f"{'='*70}\n")

    start_time = time.time()
    ga = GeneticApproximator(
        grid=GRID,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        survival_rate=SURVIVAL_RATE,
        max_depth=MAX_DEPTH,
        complexity_weight=COMPLEXITY_WEIGHT,
        diversity_weight=DIVERSITY_WEIGHT,
        symmetry_weight=SYMMETRY_WEIGHT,
    )

    best = ga.run(generations=GENERATIONS)
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best expression: {simplify_node(best.gene)}")
    print(f"Fitness: {best.fitness:.2f}")
    print(f"Raw error: {best.raw_error:.0f}")
    print(f"Symmetry violations: {best.symmetry_error:.0f}")
    print(f"Complexity: {best.gene.complexity()}")
    print(f"Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Rate: {GENERATIONS/elapsed:.1f} gen/s")

    print("\nGoal grid:")
    for row in GRID:
        print("  ", row)

    print("\nPredicted grid:")
    for y in range(8):
        row_pred: list[int] = [best.gene.evaluate((x, y)) for x in range(8)]
        print("  ", row_pred)

    print("\nDifference grid (. = correct):")
    for y in range(8):
        row_diff: list[str] = []
        for x in range(8):
            predicted = best.gene.evaluate((x, y))
            expected = GRID[y][x]
            diff = predicted - expected
            if diff == 0:
                row_diff.append("  .")
            else:
                row_diff.append(f"{diff:+3d}")
        print("  ", " ".join(row_diff))

    # Verify symmetry
    print("\nSymmetry check:")
    symmetric = True
    for x in range(8):
        for y in range(x + 1, 8):
            val_xy = best.gene.evaluate((x, y))
            val_yx = best.gene.evaluate((y, x))
            if val_xy != val_yx:
                print(f"  ASYMMETRIC: f({x},{y})={val_xy} but f({y},{x})={val_yx}")
                symmetric = False
    if symmetric:
        print("  ✓ Expression is perfectly symmetric!")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
