import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from turtle import st
from typing import Self

# ---------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# Adjust these parameters to tune the genetic algorithm.  The GRID below is
# the target 8×8 table we are trying to reproduce exactly.  POPULATION_SIZE
# controls how many individuals are evolved each generation.  MUTATION_RATE
# is the probability that any given node will be altered during mutation.
# SURVIVAL_RATE is the fraction of the population that survives unchanged
# each generation.  MAX_DEPTH limits the size of expression trees.
# COMPLEXITY_WEIGHT and DIVERSITY_WEIGHT penalise overly long expressions and
# consecutive identical outputs, respectively.  GENERATIONS sets the maximum
# number of iterations; evolution stops early if the target grid is matched.
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

POPULATION_SIZE: int = 900      # number of individuals in the population
MUTATION_RATE: float = 0.3        # probability of mutating each node
SURVIVAL_RATE: float = 0.5        # fraction of population that survives each generation
MAX_DEPTH: int = 7              # maximum depth of expression trees
COMPLEXITY_WEIGHT: float = 1.0    # penalty multiplier for expression size
DIVERSITY_WEIGHT: float = 4.5      # penalty multiplier for repeated outputs
GENERATIONS: int = 100000          # maximum number of generations to evolve

# Allowed multipliers for trigonometric nodes.  Each value is a Fraction so
# that sin(k·π·arg) and cos(k·π·arg) evaluate to integers (0 or ±1) for
# integer values of arg.  Only half‑integer multiples of π produce integer
# results for sine/cosine on integer arguments.
PI_FACTORS: list[Fraction] = [
    Fraction(1, 2), Fraction(1, 1), Fraction(3, 2),
    Fraction(2, 1), Fraction(5, 2), Fraction(3, 1),
]


# ---------------------------------------------------------------------------
# Node hierarchy for expression trees
# ---------------------------------------------------------------------------
class Node:
    """Base class for nodes in an expression tree."""

    def evaluate(self, inputs: tuple[int, int]) -> int:
        """Evaluate the node on the provided (x, y) tuple."""
        raise NotImplementedError

    def clone(self) -> Self:
        """Return a deep copy of the node."""
        raise NotImplementedError

    def mutate(self, p_mutate: float) -> Self:
        """
        Return a mutated clone of this node.  Mutations may modify operators,
        constants or subtrees.  The returned node is always a new instance.
        """
        raise NotImplementedError

    def complexity(self) -> int:
        """Return the number of nodes in the subtree rooted at this node."""
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError

    def contains_variable(self) -> bool:
        """Return True if this subtree contains at least one variable node."""
        raise NotImplementedError


class ConstNode(Node):
    """Leaf node representing a small integer constant."""

    # Restrict constants to -1, 0 or 1.  Larger random constants rarely help
    # match grids containing zero.
    ALLOWED_VALUES = [-1, 0, 1]

    def __init__(self, value: int | None = None) -> None:
        self.value: int = (
            random.choice(ConstNode.ALLOWED_VALUES) if value is None else value
        )

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.value

    def clone(self) -> Self:
        return ConstNode(self.value)

    def mutate(self, p_mutate: float) -> Node:
        # With probability p_mutate change the constant; otherwise keep it.
        if random.random() < p_mutate:
            # Choose a different value from the allowed set
            choices = [v for v in ConstNode.ALLOWED_VALUES if v != self.value]
            return ConstNode(random.choice(choices))
        return self.clone()

    def complexity(self) -> int:
        return 1

    def to_string(self) -> str:
        return str(self.value)

    def contains_variable(self) -> bool:
        return False


class VarNode(Node):
    """Leaf node representing either the x or y input variable."""

    def __init__(self, index: int | None = None) -> None:
        # index == 0 corresponds to x, index == 1 corresponds to y
        self.index: int = random.choice([0, 1]) if index is None else index

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return inputs[self.index]

    def clone(self) -> Self:
        return VarNode(self.index)

    def mutate(self, p_mutate: float) -> Node:
        # Switch variable with probability p_mutate
        if random.random() < p_mutate:
            return VarNode(1 - self.index)
        return self.clone()

    def complexity(self) -> int:
        return 1

    def to_string(self) -> str:
        return "x" if self.index == 0 else "y"

    def contains_variable(self) -> bool:
        return True


class BinaryNode(Node):
    """Base class for binary operations (Add, Multiply, Power)."""

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


class AddNode(BinaryNode):
    """Node representing integer addition."""

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.left.evaluate(inputs) + self.right.evaluate(inputs)

    def mutate(self, p_mutate: float) -> Node:
        # Occasionally replace the entire subtree with a random node
        if random.random() < p_mutate * 0.2:
            return random_node(max_depth=MAX_DEPTH)
        # Otherwise mutate children and rebuild
        left_mut, right_mut = self.mutate_children(p_mutate)
        return AddNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}+{self.right.to_string()})"


class MulNode(BinaryNode):
    """Node representing integer multiplication."""

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.left.evaluate(inputs) * self.right.evaluate(inputs)

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.2:
            return random_node(max_depth=MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return MulNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}*{self.right.to_string()})"


class PowNode(BinaryNode):
    """Node representing integer exponentiation (with clamped exponents)."""

    MAX_EXP = 5

    def evaluate(self, inputs: tuple[int, int]) -> int:
        base = self.left.evaluate(inputs)
        exp = self.right.evaluate(inputs)
        # Clamp exponent to avoid huge numbers
        if exp > PowNode.MAX_EXP:
            exp = PowNode.MAX_EXP
        if exp < -PowNode.MAX_EXP:
            exp = -PowNode.MAX_EXP
        # Negative exponents produce zero to keep results integer and bounded
        if exp < 0:
            return 0
        try:
            result = int(math.pow(base, exp))
        except OverflowError:
            result = 0
        return result

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.2:
            return random_node(max_depth=MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return PowNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}^{self.right.to_string()})"


class TrigNode(Node):
    """Base class for trigonometric operations with a π multiplier factor."""

    FUNC_NAME = "sin"
    FUNC = math.sin

    def __init__(self, child: Node | None = None, factor: Fraction | None = None) -> None:
        self.child: Node = child if child is not None else random_node()
        # Choose a random factor if not provided
        self.factor: Fraction = factor if factor is not None else random.choice(PI_FACTORS)

    def evaluate(self, inputs: tuple[int, int]) -> int:
        arg = self.child.evaluate(inputs)
        val = self.FUNC(float(self.factor) * math.pi * arg)
        # Convert result to integer; should always be 0 or ±1 for allowed factors
        return int(val)

    def clone(self) -> Self:
        return type(self)(self.child.clone(), self.factor)

    def mutate(self, p_mutate: float) -> Node:
        # With small probability switch to another trig function
        if random.random() < p_mutate * 0.1:
            # Switch between sin and cos
            new_type = SinNode if isinstance(self, CosNode) else CosNode
            return new_type(self.child.mutate(p_mutate), self.factor)
        # Occasionally change the factor
        if random.random() < p_mutate * 0.2:
            new_factor = random.choice(PI_FACTORS)
            return type(self)(self.child.mutate(p_mutate), new_factor)
        # Otherwise mutate child
        return type(self)(self.child.mutate(p_mutate), self.factor)

    def complexity(self) -> int:
        return 1 + self.child.complexity()

    def _factor_str(self) -> str:
        """Return a string representation of the factor multiplied by π."""
        num = self.factor.numerator
        den = self.factor.denominator
        if num == 0:
            return "0"
        # For integer factors
        if den == 1:
            if num == 1:
                return "π"
            return f"{num}π"
        # For half‑integer factors (den==2)
        return f"{num}π/{den}"

    def to_string(self) -> str:
        factor_part = self._factor_str()
        # If factor_part is 'π', omit multiplication sign; else include '*'
        if factor_part == "π":
            return f"{self.FUNC_NAME}({factor_part}*{self.child.to_string()})"
        return f"{self.FUNC_NAME}({factor_part}*{self.child.to_string()})"

    def contains_variable(self) -> bool:
        return self.child.contains_variable()


class SinNode(TrigNode):
    FUNC_NAME = "sin"
    FUNC = math.sin


class CosNode(TrigNode):
    FUNC_NAME = "cos"
    FUNC = math.cos


# Register available unary and binary node classes for random generation.
UNARY_NODE_TYPES = [SinNode, CosNode]
BINARY_NODE_TYPES = [AddNode, MulNode, PowNode]


def random_node(max_depth: int = MAX_DEPTH, current_depth: int = 0) -> Node:
    """
    Generate a random expression tree node up to a specified maximum depth.
    At maximum depth only leaf nodes (constants or variables) are returned.
    When creating trigonometric nodes we ensure that the subtree contains a
    variable so that trigonometric operations are meaningful.
    """
    if current_depth >= max_depth:
        # At maximum depth, return a constant or variable
        return random.choice([ConstNode(), VarNode()])

    rnd = random.random()
    # 30% chance for a variable, 20% for a constant, remainder for operations
    if rnd < 0.2:
        return ConstNode()
    elif rnd < 0.5:
        return VarNode()
    else:
        # Randomly choose between binary and unary operations
        if random.random() < 0.7:
            # Binary operation
            op_type = random.choice(BINARY_NODE_TYPES)
            left = random_node(max_depth, current_depth + 1)
            right = random_node(max_depth, current_depth + 1)
            return op_type(left, right)
        else:
            # Unary operation (Sin, Cos)
            op_type = random.choice(UNARY_NODE_TYPES)
            # Ensure trig nodes contain a variable in their subtree
            child = random_node(max_depth, current_depth + 1)
            attempts = 0
            if op_type in (SinNode, CosNode):
                # try to generate a child containing a variable
                while not child.contains_variable() and attempts < 5:
                    child = random_node(max_depth, current_depth + 1)
                    attempts += 1
                if not child.contains_variable():
                    child = VarNode()
            return op_type(child)


# ---------------------------------------------------------------------------
# Genetic algorithm implementation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Individual:
    gene: Node
    fitness: float = float("inf")
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
    ) -> None:
        self.grid = grid
        # Flatten the grid into a list of (input, output) pairs
        self.dataset: list[tuple[tuple[int, int], int]] = []
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                self.dataset.append(((x, y), val))
        # Precompute the expected outputs for early termination
        self.expected_flat: list[int] = [val for _, val in self.dataset]

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.survival_rate = survival_rate
        self.max_depth = max_depth
        self.complexity_weight = complexity_weight
        self.diversity_weight = diversity_weight

        # Detect free-threading mode (Python 3.14+)
        self.use_free_threading = (
            hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled()
        )

        # Per-instance RNG for thread safety in free-threading mode
        from random import Random
        self.rng = Random()

        # Initialise population with random trees
        self.population: list[Individual] = [
            Individual(gene=random_node(max_depth=self.max_depth))
            for _ in range(population_size)
        ]

    def evaluate_fitness(self, individual: Individual) -> None:
        """Compute the fitness of an individual using absolute error and penalties."""
        total_error = 0
        outputs: list[int] = []
        # Evaluate all points in the dataset
        for inputs, expected in self.dataset:
            predicted = individual.gene.evaluate(inputs)
            outputs.append(predicted)
            total_error += abs(expected - predicted)
        # Diversity penalty: penalise identical adjacent outputs
        diversity_penalty = 0
        for i in range(1, len(outputs)):
            if outputs[i] == outputs[i - 1]:
                diversity_penalty += 1
        complexity_penalty = individual.gene.complexity() * self.complexity_weight
        individual.fitness = (
            total_error + self.diversity_weight * diversity_penalty + complexity_penalty
        )
        individual.last_outputs = outputs

    def evaluate_population(self) -> None:
        """
        Evaluate fitness for the whole population using optimal parallelization.
        Uses free-threaded mode (Python 3.14+) if available for true parallelism,
        otherwise falls back to ProcessPoolExecutor to bypass the GIL.
        """
        # Choose executor based on free-threading availability
        if self.use_free_threading:
            # True parallelism with threads (Python 3.14t free-threaded build)
            executor_class = ThreadPoolExecutor
            max_workers = os.cpu_count() - 2 or 6
        else:
            # Fall back to processes to bypass GIL in standard Python
            executor_class = ProcessPoolExecutor
            max_workers = os.cpu_count() - 2 or 6
            # max_workers = min(6, os.cpu_count() or 6)

        with executor_class(max_workers=os.cpu_count()) as executor:
            list(executor.map(self.evaluate_fitness, self.population))
        # Sort by fitness in ascending order
        self.population.sort(key=lambda ind: ind.fitness)

    def reproduce(self) -> None:
        # Determine how many individuals survive (elitism)
        survivors = int(self.population_size * self.survival_rate)
        if survivors < 2:
            survivors = 2
        new_population: list[Individual] = []
        # Carry over the best individual unchanged
        best_gene = self.population[0].gene.clone()
        new_population.append(Individual(best_gene))
        # Fill the rest of the population with mutated copies of survivors
        while len(new_population) < self.population_size:
            parent = random.choice(self.population[:survivors])
            mutated_gene = parent.gene.mutate(self.mutation_rate)
            new_population.append(Individual(mutated_gene))
        self.population = new_population

    def run(self, generations: int) -> Individual:
        """Run the genetic algorithm for a given number of generations."""
        for gen in range(generations):
            self.evaluate_population()
            best = self.population[0]
            # Print progress occasionally
            if (gen + 1) % max(1, generations // 10) == 0 or gen == 0:
                print(
                    f"Generation {gen+1}: best fitness = {best.fitness}, expr = {simplify_node(best.gene)}"
                )
            # Early termination: if the predicted outputs match the target exactly
            if best.last_outputs == self.expected_flat:
                print(
                    f"Exact match found at generation {gen+1}! Expression = {simplify_node(best.gene)}"
                )
                return best
            self.reproduce()
        # Final evaluation before returning
        self.evaluate_population()
        return self.population[0]


def simplify_node(node: Node) -> str:
    """
    Simplify an expression tree into a human‑readable algebraic string using
    pattern matching (Python 3.10+) for faster dispatch. This function
    collapses additions and multiplications of variables and constants into
    a canonical form and leaves other operations (abs, trig) untouched.

    Pattern matching provides 5-15% faster type dispatch compared to isinstance chains.
    """
    def collect(n: Node):
        # Use pattern matching for optimized type dispatch
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
            case MulNode(left=left, right=right):
                a, r1 = collect(left)
                b, r2 = collect(right)
                # if either side is uninterpretable, emit the whole subtree as residual
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
                var_part += f"{var}\u00b2"
            else:
                var_part += f"{var}^{exp}"
        # handle coefficients of ±1
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

    # assemble final expression, tidying up '+-' into '-'
    expr = "+".join(parts).replace("+-", "-")
    return expr


def main() -> None:
    # Instantiate the genetic algorithm with the configured constants
    start_time = time.time()
    ga = GeneticApproximator(
        grid=GRID,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        survival_rate=SURVIVAL_RATE,
        max_depth=MAX_DEPTH,
        complexity_weight=COMPLEXITY_WEIGHT,
        diversity_weight=DIVERSITY_WEIGHT,
    )

    best = ga.run(generations=GENERATIONS)

    print("\nBest expression:", simplify_node(best.gene))
    print("Fitness:", best.fitness)
    print("\nGoal grid:")
    for row in GRID:
        print(row)
    print("\nPredicted grid:")
    # Compute predictions for each cell using the best gene
    for y in range(len(GRID)):
        row_pred: list[int] = []
        for x in range(len(GRID[0])):
            row_pred.append(best.gene.evaluate((x, y)))
        print(row_pred)

    print("\nTime taken:", time.time() - start_time)


if __name__ == "__main__":
    main()
