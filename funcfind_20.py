import math
import random
from dataclasses import dataclass


GRID: list[list[int]] = [
    [0, 3, 2, 3, 2, 3, 4, 5],
    [3, 4, 1, 2, 3, 4, 3, 4],
    [2, 1, 4, 3, 2, 3, 4, 5],
    [3, 2, 3, 2, 3, 4, 3, 4],
    [2, 3, 2, 3, 4, 3, 4, 5],
    [3, 4, 3, 4, 3, 4, 5, 4],
    [4, 3, 4, 3, 4, 5, 4, 5],
    [5, 4, 5, 4, 5, 4, 5, 6]
]

POPULATION_SIZE: int = 8999      # number of individuals in the population
MUTATION_RATE: float = 0.3      # probability of mutating each node
SURVIVAL_RATE: float = 0.5      # fraction of population that survives each generation
MAX_DEPTH: int = 8             # maximum depth of expression trees
COMPLEXITY_WEIGHT: float = 1.1  # penalty multiplier for expression size
DIVERSITY_WEIGHT: float = 5.0    # penalty multiplier for repeated outputs
GENERATIONS: int = 9001          # number of generations to evolve


class Node:
    """Base class for nodes in an expression tree."""

    def evaluate(self, inputs: tuple[int, int]) -> int:
        """Evaluate the node on the provided (x, y) tuple."""
        raise NotImplementedError

    def clone(self) -> "Node":
        """Return a deep copy of the node."""
        raise NotImplementedError

    def mutate(self, p_mutate: float) -> "Node":
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
    """Leaf node representing an integer constant."""

    def __init__(self, value: int | None = None) -> None:
        self.value: int = random.randint(-10, 10) if value is None else value

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.value

    def clone(self) -> "ConstNode":
        return ConstNode(self.value)

    def mutate(self, p_mutate: float) -> "Node":
        # With probability p_mutate change the constant; otherwise keep it.
        if random.random() < p_mutate:
            # Small jitter or completely new constant
            if random.random() < 0.7:
                # Adjust by a small random step
                step = random.randint(-3, 3)
                return ConstNode(self.value + step)
            else:
                # Replace with a fresh random constant
                return ConstNode()
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

    def clone(self) -> "VarNode":
        return VarNode(self.index)

    def mutate(self, p_mutate: float) -> "Node":
        # Switch variable with probability p_mutate
        if random.random() < p_mutate:
            return VarNode(1 - self.index)
        return self.clone()

    def complexity(self) -> int:
        return 1

    def to_string(self) -> str:
        return 'x' if self.index == 0 else 'y'

    def contains_variable(self) -> bool:
        return True


class BinaryNode(Node):
    """Base class for binary operations (Add, Multiply, Power)."""

    def __init__(self, left: Node | None = None, right: Node | None = None) -> None:
        self.left: Node = left if left is not None else random_node()
        self.right: Node = right if right is not None else random_node()

    def clone(self) -> "BinaryNode":
        return type(self)(self.left.clone(), self.right.clone())

    def mutate_children(self, p_mutate: float) -> tuple[Node, Node]:
        return self.left.mutate(p_mutate), self.right.mutate(p_mutate)

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def contains_variable(self) -> bool:
        return self.left.contains_variable() or self.right.contains_variable()


class AddNode(BinaryNode):
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


class SinNode(Node):
    """Unary sine function.  Returns sin(k·π) where k is an integer, yielding 0."""

    def __init__(self, child: Node | None = None) -> None:
        self.child: Node = child if child is not None else random_node()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        # Multiply argument by π to ensure sin() yields an integer value.
        arg = self.child.evaluate(inputs)
        return int(math.sin(arg * math.pi))

    def clone(self) -> "SinNode":
        return SinNode(self.child.clone())

    def mutate(self, p_mutate: float) -> Node:
        # Occasionally switch to cosine
        if random.random() < p_mutate * 0.2:
            return CosNode(self.child.mutate(p_mutate))
        # Otherwise mutate child and rebuild
        return SinNode(self.child.mutate(p_mutate))

    def complexity(self) -> int:
        return 1 + self.child.complexity()

    def to_string(self) -> str:
        return f"sin({self.child.to_string()})"

    def contains_variable(self) -> bool:
        return self.child.contains_variable()


class CosNode(Node):
    """Unary cosine function.  Returns cos(k·π) yielding ±1 depending on k."""

    def __init__(self, child: Node | None = None) -> None:
        self.child: Node = child if child is not None else random_node()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        arg = self.child.evaluate(inputs)
        return int(math.cos(arg * math.pi))

    def clone(self) -> "CosNode":
        return CosNode(self.child.clone())

    def mutate(self, p_mutate: float) -> Node:
        # Occasionally switch to sine
        if random.random() < p_mutate * 0.2:
            return SinNode(self.child.mutate(p_mutate))
        return CosNode(self.child.mutate(p_mutate))

    def complexity(self) -> int:
        return 1 + self.child.complexity()

    def to_string(self) -> str:
        return f"cos({self.child.to_string()})"

    def contains_variable(self) -> bool:
        return self.child.contains_variable()



## ---- NODE GENERATION FUNCTION ---- ##
def random_node(max_depth: int = MAX_DEPTH, current_depth: int = 0) -> Node:
    """
    Generate a random expression tree node up to a specified maximum depth.
    At maximum depth only leaf nodes (constants or variables) are returned.
    When creating sin or cos nodes we ensure that the subtree contains a
    variable so that trigonometric operations are meaningful.
    """
    if current_depth >= max_depth:
        # At maximum depth, return a constant or variable
        return random.choice([ConstNode(), VarNode()])

    rnd = random.random()
    # 25% chance for a constant, 25% for a variable, 50% for an operation
    if rnd < 0.25:
        return ConstNode()
    elif rnd < 0.5:
        return VarNode()
    else:
        op_type = random.choice([AddNode, MulNode, PowNode, SinNode, CosNode])
        # Sin and Cos must contain a variable in their subtree to avoid useless calls
        if op_type in (SinNode, CosNode):
            # Try to generate a child that contains a variable
            attempts = 0
            child = random_node(max_depth, current_depth + 1)
            while not child.contains_variable() and attempts < 5:
                child = random_node(max_depth, current_depth + 1)
                attempts += 1
            # If after several attempts there is still no variable, force a variable node
            if not child.contains_variable():
                child = VarNode()
            return op_type(child)
        # Binary operations require two children
        if issubclass(op_type, BinaryNode):
            left = random_node(max_depth, current_depth + 1)
            right = random_node(max_depth, current_depth + 1)
            return op_type(left, right)
        else:
            # For completeness; unary ops handled above
            child = random_node(max_depth, current_depth + 1)
            return op_type(child)


# ---------------------------------------------------------------------------
# Genetic algorithm implementation
# ---------------------------------------------------------------------------
@dataclass
class Individual:
    gene: Node
    fitness: float = float('inf')
    last_outputs: list[int] | None = None


class GeneticApproximator:
    def __init__(self, grid: list[list[int]], population_size: int,
                 mutation_rate: float, survival_rate: float,
                 max_depth: int, complexity_weight: float,
                 diversity_weight: float) -> None:
        self.grid = grid
        # Flatten the grid into a list of (input, output) pairs
        self.dataset: list[tuple[tuple[int, int], int]] = []
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                self.dataset.append(((x, y), val))
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.survival_rate = survival_rate
        self.max_depth = max_depth
        self.complexity_weight = complexity_weight
        self.diversity_weight = diversity_weight
        # Initialise population with random trees
        self.population: list[Individual] = [Individual(gene=random_node(max_depth=self.max_depth))
                                             for _ in range(population_size)]

    def evaluate_fitness(self, individual: Individual) -> None:
        """Compute the fitness of an individual using absolute error and penalties."""
        total_error = 0
        outputs: list[int] = []
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
        individual.fitness = total_error + self.diversity_weight * diversity_penalty + complexity_penalty
        individual.last_outputs = outputs

    def evaluate_population(self) -> None:
        for ind in self.population:
            self.evaluate_fitness(ind)
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
        for gen in range(generations):
            self.evaluate_population()
            best = self.population[0]
            # Print progress occasionally
            if (gen + 1) % max(1, generations // 10) == 0 or gen == 0:
                print(f"Generation {gen+1}: best fitness = {best.fitness}, expr = {simplify_node(best.gene)}")
            self.reproduce()
        # Final evaluation before returning
        self.evaluate_population()
        return self.population[0]

def simplify_node(node):
    def collect(n):
        if isinstance(n, ConstNode):
            return {(): n.value}, []
        if isinstance(n, VarNode):
            name = 'x' if n.index == 0 else 'y'
            return {((name, 1),): 1}, []
        if isinstance(n, AddNode):
            a, r1 = collect(n.left)
            b, r2 = collect(n.right)
            out = a.copy()
            for k, v in b.items():
                out[k] = out.get(k, 0) + v
            return out, r1 + r2
        if isinstance(n, MulNode):
            a, r1 = collect(n.left)
            b, r2 = collect(n.right)
            # if either side is uninterpretable, emit the whole subtree as residual
            if r1 or r2:
                return {}, [n.to_string()]
            out = {}
            for (k1, c1) in a.items():
                for (k2, c2) in b.items():
                    # combine variable exponents
                    exp_map = {}
                    for var, e in k1:
                        exp_map[var] = exp_map.get(var, 0) + e
                    for var, e in k2:
                        exp_map[var] = exp_map.get(var, 0) + e
                    key = tuple(sorted(exp_map.items()))
                    out[key] = out.get(key, 0) + c1 * c2
            return out, []
        if isinstance(n, PowNode):
            if isinstance(n.right, ConstNode) and n.right.value >= 0:
                exp = n.right.value
                base_terms, residuals = collect(n.left)
                if residuals or len(base_terms) != 1:
                    return {}, [n.to_string()]
                (vars_tuple, coef) = next(iter(base_terms.items()))
                if vars_tuple == ():
                    return {(): coef ** exp}, []
                exp_map = {}
                for var, e in vars_tuple:
                    exp_map[var] = exp * e
                key = tuple(sorted(exp_map.items()))
                return {key: coef ** exp}, []
            return {}, [n.to_string()]
        # sin, cos or anything else -> uninterpretable
        return {}, [n.to_string()]

    terms, residuals = collect(node)

    priority = {'x': 0, 'y': 1}
    def term_priority(item):
        key, _ = item
        if not key:
            return (2, 0)
        pr = min(priority.get(var, 2) for var, _ in key)
        return (pr, -sum(e for _, e in key))

    def format_term(key, coef):
        if not key:
            return str(coef)
        var_part = ''
        for var, exp in key:
            if exp == 1:
                var_part += var
            elif exp == 2:
                var_part += f"{var}\u00b2"
            else:
                var_part += f"{var}^{exp}"
        # handle coefficients of �1
        if coef == 1:
            return var_part
        if coef == -1:
            return '-' + var_part
        return f"{coef}{var_part}"

    parts = []
    for key, coef in sorted(terms.items(), key=term_priority):
        if coef:
            parts.append(format_term(key, coef))
    parts.extend(residuals)

    # assemble final expression, tidying up '+-' into '-'
    expr = '+'.join(parts).replace('+-', '-')
    return expr

def main() -> None:
    # Instantiate the genetic algorithm with the configured constants
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


if __name__ == "__main__":
    main()
