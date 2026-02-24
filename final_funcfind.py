#!/usr/bin/env python3
"""Canonical high-performance function finder with Numba-first acceleration."""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

_FORCE_DISABLE_NUMBA = os.environ.get("FUNCFIND_DISABLE_NUMBA", "").strip() == "1"

try:
    if _FORCE_DISABLE_NUMBA:
        raise ImportError("FUNCFIND_DISABLE_NUMBA=1")
    from numba import njit
    HAS_NUMBA = True
    NUMBA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - runtime fallback guard
    HAS_NUMBA = False
    NUMBA_IMPORT_ERROR = exc

    def njit(*_args: object, **_kwargs: object):
        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Target and defaults
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

DEFAULT_POPULATION = 640
DEFAULT_GENERATIONS = 35_000
FAST_POPULATION = 280
FAST_GENERATIONS = 8_000

MUTATION_RATE = 0.30
SURVIVAL_RATE = 0.25
MAX_DEPTH = 8
COMPLEXITY_WEIGHT = 0.35
DIVERSITY_WEIGHT = 5.5
SYMMETRY_WEIGHT = 80.0
TOURNAMENT_SIZE = 7

OUTPUT_ABS_CLIP = 20_000
POW_EXP_CLIP = 5
OUT_OF_RANGE_LOW = -2
OUT_OF_RANGE_HIGH = 10
RANGE_TARGET = 3
RANGE_WEIGHT = 0.25

PATTERN_TYPES = (
    "x_plus_y",
    "x_times_y",
    "x2_plus_y2",
    "xy_combo",
    "difference_squared",
    "symmetric_poly",
)


# ---------------------------------------------------------------------------
# Postfix VM opcodes (for Numba scoring)
# ---------------------------------------------------------------------------
OP_CONST = np.int64(0)
OP_X = np.int64(1)
OP_Y = np.int64(2)
OP_ADD = np.int64(3)
OP_SUB = np.int64(4)
OP_MUL = np.int64(5)
OP_POW = np.int64(6)
OP_DIV = np.int64(7)
OP_MOD = np.int64(8)


@njit(cache=True)
def _clip_int(value: np.int64, abs_limit: np.int64) -> np.int64:
    if value > abs_limit:
        return abs_limit
    if value < -abs_limit:
        return -abs_limit
    return value


@njit(cache=True)
def _safe_pow(base: np.int64, exp: np.int64, max_exp: np.int64, abs_limit: np.int64) -> np.int64:
    if exp < 0:
        return np.int64(0)
    if exp > max_exp:
        exp = max_exp

    result = np.int64(1)
    for _ in range(exp):
        result = _clip_int(result * base, abs_limit)
    return result


@njit(cache=True)
def _trunc_div(a: np.int64, b: np.int64) -> np.int64:
    if b == 0:
        return np.int64(0)
    q = abs(a) // abs(b)
    if (a < 0 and b > 0) or (a > 0 and b < 0):
        q = -q
    return q


@njit(cache=True)
def _eval_program_point(
    opcodes: np.ndarray,
    operands: np.ndarray,
    x_val: np.int64,
    y_val: np.int64,
    stack: np.ndarray,
    abs_limit: np.int64,
    max_exp: np.int64,
) -> np.int64:
    sp = 0
    for i in range(opcodes.shape[0]):
        op = opcodes[i]
        arg = operands[i]

        if op == OP_CONST:
            stack[sp] = arg
            sp += 1
            continue

        if op == OP_X:
            stack[sp] = x_val
            sp += 1
            continue

        if op == OP_Y:
            stack[sp] = y_val
            sp += 1
            continue

        if sp < 2:
            return np.int64(0)

        right = stack[sp - 1]
        left = stack[sp - 2]

        if op == OP_ADD:
            stack[sp - 2] = _clip_int(left + right, abs_limit)
        elif op == OP_SUB:
            stack[sp - 2] = _clip_int(left - right, abs_limit)
        elif op == OP_MUL:
            stack[sp - 2] = _clip_int(left * right, abs_limit)
        elif op == OP_POW:
            stack[sp - 2] = _safe_pow(left, right, max_exp, abs_limit)
        elif op == OP_DIV:
            stack[sp - 2] = _clip_int(_trunc_div(left, right), abs_limit)
        elif op == OP_MOD:
            if right == 0:
                stack[sp - 2] = np.int64(0)
            else:
                stack[sp - 2] = _clip_int(left % right, abs_limit)
        else:
            stack[sp - 2] = np.int64(0)

        sp -= 1

    if sp != 1:
        return np.int64(0)
    return stack[0]


@njit(cache=True)
def _score_program(
    opcodes: np.ndarray,
    operands: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    expected_flat: np.ndarray,
    symmetry_left: np.ndarray,
    symmetry_right: np.ndarray,
    complexity_weight: np.float64,
    diversity_weight: np.float64,
    symmetry_weight: np.float64,
    range_weight: np.float64,
    out_low: np.int64,
    out_high: np.int64,
    range_target: np.int64,
    abs_limit: np.int64,
    max_exp: np.int64,
) -> tuple[np.float64, np.int64, np.int64, np.int64, np.int64, np.ndarray]:
    n = expected_flat.shape[0]
    outputs = np.empty(n, dtype=np.int64)
    stack = np.empty(opcodes.shape[0] + 2, dtype=np.int64)

    raw_error = np.int64(0)
    diversity_penalty = np.int64(0)
    range_penalty = np.int64(0)

    prev = np.int64(0)
    for i in range(n):
        out = _eval_program_point(
            opcodes,
            operands,
            xs[i],
            ys[i],
            stack,
            abs_limit,
            max_exp,
        )
        outputs[i] = out

        diff = expected_flat[i] - out
        if diff < 0:
            diff = -diff
        raw_error += diff

        if i > 0 and out == prev:
            diversity_penalty += 1
        prev = out

        if out < out_low or out > out_high:
            delta = out - range_target
            if delta < 0:
                delta = -delta
            range_penalty += delta

    symmetry_error = np.int64(0)
    for i in range(symmetry_left.shape[0]):
        l_idx = symmetry_left[i]
        r_idx = symmetry_right[i]
        sdiff = outputs[l_idx] - outputs[r_idx]
        if sdiff < 0:
            sdiff = -sdiff
        symmetry_error += sdiff

    complexity = opcodes.shape[0]
    fitness = (
        np.float64(raw_error)
        + symmetry_weight * np.float64(symmetry_error)
        + diversity_weight * np.float64(diversity_penalty)
        + complexity_weight * np.float64(complexity)
        + range_weight * np.float64(range_penalty)
    )

    return fitness, raw_error, symmetry_error, diversity_penalty, range_penalty, outputs


def _ensure_numba_ready() -> None:
    """Force JIT compile at startup so failures happen early and clearly."""
    if not HAS_NUMBA:
        raise RuntimeError("Numba is not available in this environment.")

    xs = np.array([0, 1, 2], dtype=np.int64)
    ys = np.array([0, 1, 2], dtype=np.int64)
    expected = np.array([0, 2, 4], dtype=np.int64)

    # Program: x + y
    opcodes = np.array([OP_X, OP_Y, OP_ADD], dtype=np.int64)
    operands = np.array([0, 0, 0], dtype=np.int64)

    left = np.array([1], dtype=np.int64)
    right = np.array([1], dtype=np.int64)

    try:
        _score_program(
            opcodes,
            operands,
            xs,
            ys,
            expected,
            left,
            right,
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.int64(-100),
            np.int64(100),
            np.int64(0),
            np.int64(10_000),
            np.int64(5),
        )
    except Exception as exc:  # pragma: no cover - startup guard
        raise SystemExit(f"Numba JIT failed during warmup: {exc}") from exc


def _enforce_acceleration_policy() -> None:
    """
    Runtime policy:
    - Python 3.14.x: Numba is required.
    - Python 3.15+: allow pure-Python fallback while Numba/llvmlite catches up.
    """
    major, minor = sys.version_info[:2]

    if HAS_NUMBA:
        _ensure_numba_ready()
        return

    if (major, minor) <= (3, 14):
        detail = f" Original import error: {NUMBA_IMPORT_ERROR}" if NUMBA_IMPORT_ERROR else ""
        raise SystemExit(
            "Numba is required on Python 3.14. Install project dependencies with `uv sync`."
            + detail
        )


# ---------------------------------------------------------------------------
# Expression tree model
# ---------------------------------------------------------------------------
class Node:
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        raise NotImplementedError

    def clone(self) -> Self:
        raise NotImplementedError

    def mutate(self, p_mutate: float) -> Self:
        raise NotImplementedError

    def complexity(self) -> int:
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError

    def contains_variable(self) -> bool:
        raise NotImplementedError

    def get_depth(self) -> int:
        raise NotImplementedError

    def is_symmetric(self) -> bool:
        raise NotImplementedError


class ConstNode(Node):
    __slots__ = ("value",)
    ALLOWED_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2]

    def __init__(self, value: int | None = None) -> None:
        self.value = random.choice(ConstNode.ALLOWED_VALUES) if value is None else int(value)

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.value

    def clone(self) -> Self:
        return ConstNode(self.value)

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate:
            if random.random() < 0.6 and -4 <= self.value <= 10:
                jitter = random.choice((-1, 1))
                return ConstNode(max(-4, min(10, self.value + jitter)))
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
        return True


class VarNode(Node):
    __slots__ = ("index",)

    def __init__(self, index: int | None = None) -> None:
        self.index = random.choice([0, 1]) if index is None else int(index)

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return inputs[self.index]

    def clone(self) -> Self:
        return VarNode(self.index)

    def mutate(self, p_mutate: float) -> Node:
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
        return False


class BinaryNode(Node):
    __slots__ = ("left", "right")

    def __init__(self, left: Node | None = None, right: Node | None = None) -> None:
        self.left = left if left is not None else random_node()
        self.right = right if right is not None else random_node()

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
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.left.evaluate(inputs) + self.right.evaluate(inputs)

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.15:
            return random_node(MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return AddNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}+{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        if self.left.is_symmetric() and self.right.is_symmetric():
            return True
        if isinstance(self.left, VarNode) and isinstance(self.right, VarNode):
            return self.left.index != self.right.index
        return False


class SubNode(BinaryNode):
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.left.evaluate(inputs) - self.right.evaluate(inputs)

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.15:
            return random_node(MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return SubNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}-{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        return self.left.is_symmetric() and self.right.is_symmetric()


class MulNode(BinaryNode):
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.left.evaluate(inputs) * self.right.evaluate(inputs)

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.15:
            return random_node(MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return MulNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}*{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        if self.left.is_symmetric() and self.right.is_symmetric():
            return True
        if isinstance(self.left, VarNode) and isinstance(self.right, VarNode):
            return self.left.index != self.right.index
        return False


class PowNode(BinaryNode):
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        base = self.left.evaluate(inputs)
        exp = self.right.evaluate(inputs)
        if exp < 0:
            return 0
        if exp > POW_EXP_CLIP:
            exp = POW_EXP_CLIP
        out = 1
        for _ in range(exp):
            out *= base
            if out > OUTPUT_ABS_CLIP:
                return OUTPUT_ABS_CLIP
            if out < -OUTPUT_ABS_CLIP:
                return -OUTPUT_ABS_CLIP
        return out

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.15:
            return random_node(MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return PowNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}^{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        return self.left.is_symmetric() and self.right.is_symmetric()


class DivNode(BinaryNode):
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        left = self.left.evaluate(inputs)
        right = self.right.evaluate(inputs)
        if right == 0:
            return 0
        q = abs(left) // abs(right)
        if (left < 0 and right > 0) or (left > 0 and right < 0):
            q = -q
        return q

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.15:
            return random_node(MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return DivNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}/{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        return self.left.is_symmetric() and self.right.is_symmetric()


class ModNode(BinaryNode):
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        left = self.left.evaluate(inputs)
        right = self.right.evaluate(inputs)
        if right == 0:
            return 0
        return left % right

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.15:
            return random_node(MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return ModNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}%{self.right.to_string()})"

    def is_symmetric(self) -> bool:
        return self.left.is_symmetric() and self.right.is_symmetric()


BINARY_NODE_TYPES = (AddNode, SubNode, MulNode, PowNode, DivNode, ModNode)


def random_node(max_depth: int = MAX_DEPTH, current_depth: int = 0) -> Node:
    if current_depth >= max_depth:
        return random.choice((ConstNode(), VarNode()))

    pick = random.random()
    if pick < 0.15:
        return ConstNode()
    if pick < 0.35:
        return VarNode()

    op_cls = random.choice(BINARY_NODE_TYPES)
    left = random_node(max_depth, current_depth + 1)
    right = random_node(max_depth, current_depth + 1)
    return op_cls(left, right)


def create_symmetric_node(pattern_type: str) -> Node:
    x_var = VarNode(0)
    y_var = VarNode(1)

    match pattern_type:
        case "x_plus_y":
            return AddNode(x_var, y_var)
        case "x_times_y":
            return MulNode(x_var, y_var)
        case "x2_plus_y2":
            x_sq = PowNode(x_var, ConstNode(2))
            y_sq = PowNode(y_var, ConstNode(2))
            return AddNode(x_sq, y_sq)
        case "xy_combo":
            sum_xy = AddNode(x_var, y_var)
            prod_xy = MulNode(x_var, y_var)
            return AddNode(sum_xy, prod_xy)
        case "difference_squared":
            diff = SubNode(x_var, y_var)
            return PowNode(diff, ConstNode(2))
        case "symmetric_poly":
            x_sq = PowNode(x_var, ConstNode(2))
            y_sq = PowNode(y_var, ConstNode(2))
            sum_sq = AddNode(x_sq, y_sq)
            sum_xy = AddNode(x_var, y_var)
            weighted_sum = MulNode(ConstNode(random.randint(-2, 3)), sum_xy)
            return AddNode(sum_sq, weighted_sum)
        case _:
            return AddNode(x_var, y_var)


def compile_gene_to_postfix(gene: Node) -> tuple[np.ndarray, np.ndarray]:
    opcodes: list[int] = []
    operands: list[int] = []

    def emit(node: Node) -> None:
        match node:
            case ConstNode(value=value):
                opcodes.append(int(OP_CONST))
                operands.append(int(value))
            case VarNode(index=0):
                opcodes.append(int(OP_X))
                operands.append(0)
            case VarNode(index=1):
                opcodes.append(int(OP_Y))
                operands.append(0)
            case AddNode(left=left, right=right):
                emit(left)
                emit(right)
                opcodes.append(int(OP_ADD))
                operands.append(0)
            case SubNode(left=left, right=right):
                emit(left)
                emit(right)
                opcodes.append(int(OP_SUB))
                operands.append(0)
            case MulNode(left=left, right=right):
                emit(left)
                emit(right)
                opcodes.append(int(OP_MUL))
                operands.append(0)
            case PowNode(left=left, right=right):
                emit(left)
                emit(right)
                opcodes.append(int(OP_POW))
                operands.append(0)
            case DivNode(left=left, right=right):
                emit(left)
                emit(right)
                opcodes.append(int(OP_DIV))
                operands.append(0)
            case ModNode(left=left, right=right):
                emit(left)
                emit(right)
                opcodes.append(int(OP_MOD))
                operands.append(0)
            case _:
                raise TypeError(f"Unsupported node type: {type(node).__name__}")

    emit(gene)

    return (
        np.asarray(opcodes, dtype=np.int64),
        np.asarray(operands, dtype=np.int64),
    )


def _is_free_threaded() -> bool:
    checker = getattr(sys, "_is_gil_enabled", None)
    if checker is None:
        return False
    try:
        return not bool(checker())
    except Exception:
        return False


def _build_symmetry_pairs(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if width != height:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    left: list[int] = []
    right: list[int] = []
    for x in range(width):
        for y in range(x + 1, height):
            left.append(y * width + x)
            right.append(x * width + y)

    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


@dataclass(slots=True)
class Individual:
    gene: Node
    fitness: float = float("inf")
    raw_error: float = float("inf")
    symmetry_error: float = float("inf")
    complexity: int = 0
    last_outputs: np.ndarray | None = None
    opcodes: np.ndarray | None = None
    operands: np.ndarray | None = None


class GeneticApproximator:
    def __init__(
        self,
        grid: list[list[int]],
        population_size: int,
        mutation_rate: float,
        survival_rate: float,
        complexity_weight: float,
        diversity_weight: float,
        symmetry_weight: float,
        workers: int,
        verbose: bool,
    ) -> None:
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

        self.expected_grid = np.asarray(grid, dtype=np.int64)
        self.expected_flat = self.expected_grid.reshape(-1).copy()

        n = self.expected_flat.shape[0]
        self.x_inputs = np.empty(n, dtype=np.int64)
        self.y_inputs = np.empty(n, dtype=np.int64)

        idx = 0
        for y in range(self.height):
            for x in range(self.width):
                self.x_inputs[idx] = x
                self.y_inputs[idx] = y
                idx += 1

        self.symmetry_left, self.symmetry_right = _build_symmetry_pairs(self.width, self.height)

        self.population_size = max(4, population_size)
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.survival_rate = survival_rate
        self.complexity_weight = complexity_weight
        self.diversity_weight = diversity_weight
        self.symmetry_weight = symmetry_weight
        self.workers = max(1, workers)
        self.verbose = verbose

        self.use_free_threading = _is_free_threaded()

        self.best_fitness_history: list[float] = []
        self.stagnation_counter = 0
        self.best_seen: Individual | None = None
        self.generations_completed = 0
        self.exact_match = False

        self.population: list[Individual] = []

        seed_count = self.population_size // 2
        for _ in range(seed_count):
            self.population.append(
                Individual(gene=create_symmetric_node(random.choice(PATTERN_TYPES)))
            )

        for _ in range(self.population_size - seed_count):
            self.population.append(Individual(gene=random_node(max_depth=MAX_DEPTH)))

    def _score_individual(self, individual: Individual) -> None:
        if individual.opcodes is None or individual.operands is None:
            opcodes, operands = compile_gene_to_postfix(individual.gene)
            individual.opcodes = opcodes
            individual.operands = operands

        fitness, raw_error, sym_error, _, _, outputs = _score_program(
            individual.opcodes,
            individual.operands,
            self.x_inputs,
            self.y_inputs,
            self.expected_flat,
            self.symmetry_left,
            self.symmetry_right,
            np.float64(self.complexity_weight),
            np.float64(self.diversity_weight),
            np.float64(self.symmetry_weight),
            np.float64(RANGE_WEIGHT),
            np.int64(OUT_OF_RANGE_LOW),
            np.int64(OUT_OF_RANGE_HIGH),
            np.int64(RANGE_TARGET),
            np.int64(OUTPUT_ABS_CLIP),
            np.int64(POW_EXP_CLIP),
        )

        individual.fitness = float(fitness)
        individual.raw_error = float(raw_error)
        individual.symmetry_error = float(sym_error)
        individual.last_outputs = outputs
        individual.complexity = int(individual.opcodes.shape[0])

    def evaluate_population(self) -> None:
        if self.workers <= 1:
            for individual in self.population:
                self._score_individual(individual)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                list(executor.map(self._score_individual, self.population))

        self.population.sort(key=lambda ind: ind.fitness)

    def tournament_selection(self) -> Individual:
        pool_limit = max(2, len(self.population) // 2)
        pool = self.population[:pool_limit]
        k = min(TOURNAMENT_SIZE, len(pool))
        tournament = random.sample(pool, k)
        return min(tournament, key=lambda ind: ind.fitness)

    def _clone_elite(self, parent: Individual) -> Individual:
        elite = Individual(gene=parent.gene.clone())
        if parent.opcodes is not None and parent.operands is not None:
            elite.opcodes = parent.opcodes.copy()
            elite.operands = parent.operands.copy()
        return elite

    def reproduce(self) -> None:
        survivors = max(2, int(self.population_size * self.survival_rate))
        elite_count = min(12, survivors, self.population_size)

        new_population: list[Individual] = []

        for i in range(elite_count):
            new_population.append(self._clone_elite(self.population[i]))

        while len(new_population) < self.population_size:
            parent = self.tournament_selection()
            mutated_gene = parent.gene.mutate(self.mutation_rate)
            new_population.append(Individual(gene=mutated_gene))

        self.population = new_population

    def adapt_parameters(self) -> None:
        if len(self.best_fitness_history) >= 30:
            recent_improvement = self.best_fitness_history[-30] - self.best_fitness_history[-1]

            if recent_improvement < 1.0:
                self.stagnation_counter += 1
                self.mutation_rate = min(0.65, self.mutation_rate * 1.03)
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
                self.mutation_rate = max(0.12, self.mutation_rate * 0.985)

        if self.stagnation_counter > 120:
            replace_count = max(1, self.population_size // 5)
            for i in range(replace_count):
                idx = self.population_size - 1 - i
                if random.random() < 0.65:
                    self.population[idx] = Individual(
                        gene=create_symmetric_node(random.choice(PATTERN_TYPES))
                    )
                else:
                    self.population[idx] = Individual(gene=random_node(max_depth=MAX_DEPTH))
            self.stagnation_counter = 0

    def run(self, generations: int) -> Individual:
        start_time = time.perf_counter()
        progress_step = max(1, min(generations // 20, 1000))

        try:
            for gen in range(1, generations + 1):
                self.evaluate_population()
                best = self.population[0]
                self.generations_completed = gen

                self.best_fitness_history.append(best.fitness)
                if self.best_seen is None or best.fitness < self.best_seen.fitness:
                    self.best_seen = best

                if self.verbose and (gen == 1 or gen % progress_step == 0):
                    elapsed = time.perf_counter() - start_time
                    rate = gen / elapsed if elapsed > 0 else 0.0
                    print(
                        f"gen={gen:6d} "
                        f"fit={best.fitness:10.3f} "
                        f"raw={best.raw_error:7.0f} "
                        f"sym={best.symmetry_error:7.0f} "
                        f"cx={best.complexity:4d} "
                        f"mut={self.mutation_rate:0.3f} "
                        f"rate={rate:7.2f} g/s"
                    )

                if best.last_outputs is not None and np.array_equal(best.last_outputs, self.expected_flat):
                    self.exact_match = True
                    return best

                if best.raw_error == 0.0 and best.symmetry_error == 0.0:
                    return best

                self.reproduce()
                self.adapt_parameters()

        except KeyboardInterrupt:
            print("\nInterrupted. Returning best-so-far result.")

        self.evaluate_population()
        final_best = self.population[0]

        if self.best_seen is None:
            return final_best
        return self.best_seen if self.best_seen.fitness <= final_best.fitness else final_best


def simplify_node(node: Node) -> str:
    def collect(n: Node) -> tuple[dict[tuple[tuple[str, int], ...], int], list[str]]:
        match n:
            case ConstNode(value=value):
                return {(): value}, []
            case VarNode(index=0):
                return {(('x', 1),): 1}, []
            case VarNode(index=1):
                return {(('y', 1),): 1}, []
            case AddNode(left=left, right=right):
                a, r1 = collect(left)
                b, r2 = collect(right)
                out = a.copy()
                for key, value in b.items():
                    out[key] = out.get(key, 0) + value
                return out, r1 + r2
            case SubNode(left=left, right=right):
                a, r1 = collect(left)
                b, r2 = collect(right)
                out = a.copy()
                for key, value in b.items():
                    out[key] = out.get(key, 0) - value
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
                        for var, exp in k1:
                            exp_map[var] = exp_map.get(var, 0) + exp
                        for var, exp in k2:
                            exp_map[var] = exp_map.get(var, 0) + exp
                        combined_key = tuple(sorted(exp_map.items()))
                        out[combined_key] = out.get(combined_key, 0) + c1 * c2
                return out, []
            case PowNode(left=left, right=ConstNode(value=exp)) if exp >= 0:
                base_terms, residuals = collect(left)
                if residuals or len(base_terms) != 1:
                    return {}, [n.to_string()]

                (vars_tuple, coef) = next(iter(base_terms.items()))
                if vars_tuple == ():
                    return {(): coef ** exp}, []

                exp_map: dict[str, int] = {}
                for var, old_exp in vars_tuple:
                    exp_map[var] = exp * old_exp
                key = tuple(sorted(exp_map.items()))
                return {key: coef ** exp}, []
            case _:
                return {}, [n.to_string()]

    terms, residuals = collect(node)

    var_priority = {"x": 0, "y": 1}

    def sort_key(item: tuple[tuple[tuple[str, int], ...], int]) -> tuple[int, int]:
        key, _ = item
        if not key:
            return (3, 0)
        first = min(var_priority.get(var, 3) for var, _ in key)
        degree = -sum(exp for _, exp in key)
        return (first, degree)

    def format_term(key: tuple[tuple[str, int], ...], coef: int) -> str:
        if not key:
            return str(coef)

        parts: list[str] = []
        for var, exp in key:
            if exp == 1:
                parts.append(var)
            else:
                parts.append(f"{var}^{exp}")
        var_expr = "".join(parts)

        if coef == 1:
            return var_expr
        if coef == -1:
            return f"-{var_expr}"
        return f"{coef}{var_expr}"

    rendered: list[str] = []
    for key, coef in sorted(terms.items(), key=sort_key):
        if coef != 0:
            rendered.append(format_term(key, coef))

    rendered.extend(residuals)

    expr = "+".join(rendered).replace("+-", "-")
    return expr if expr else "0"


def _validate_grid(data: object, source: str) -> list[list[int]]:
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError(f"{source}: grid must be a non-empty list of rows")

    rows: list[list[int]] = []
    width = None

    for row_idx, row in enumerate(data):
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"{source}: row {row_idx} must be a non-empty list")

        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{source}: all rows must have equal length")

        parsed_row: list[int] = []
        for col_idx, value in enumerate(row):
            if isinstance(value, bool):
                raise ValueError(f"{source}: boolean at row {row_idx}, col {col_idx} is invalid")
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{source}: value at row {row_idx}, col {col_idx} must be numeric"
                )
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(
                    f"{source}: value at row {row_idx}, col {col_idx} must be an integer"
                )
            parsed_row.append(int(value))
        rows.append(parsed_row)

    return rows


def _load_grid(path: Path) -> list[list[int]]:
    text = path.read_text(encoding="utf-8").strip()

    parse_errors: list[str] = []
    for parser_name, parser_fn in (("json", json.loads), ("python-literal", ast.literal_eval)):
        try:
            data = parser_fn(text)
            return _validate_grid(data, str(path))
        except Exception as exc:
            parse_errors.append(f"{parser_name}: {exc}")

    # Accept row-per-line format:
    # [1,2,3],\n[4,5,6], ...
    try:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and all(line.startswith("[") for line in lines):
            rows: list[list[int]] = []
            for line in lines:
                cleaned = line[:-1] if line.endswith(",") else line
                row = ast.literal_eval(cleaned)
                rows.append(row)
            return _validate_grid(rows, str(path))
    except Exception as exc:
        parse_errors.append(f"line-grid: {exc}")

    details = "; ".join(parse_errors)
    raise ValueError(f"Unable to parse target file {path}: {details}")


def _auto_workers(population: int) -> int:
    # For this workload (many tiny fitness evaluations), thread orchestration
    # costs more than it saves on CPython builds. Use 1 by default.
    _ = population
    return 1


def _predict_grid(best: Individual, width: int, height: int) -> list[list[int]]:
    if best.last_outputs is None:
        return [[0 for _ in range(width)] for _ in range(height)]

    matrix = best.last_outputs.reshape((height, width))
    return [[int(value) for value in row] for row in matrix]


def _difference_grid(predicted: list[list[int]], target: list[list[int]]) -> list[list[int]]:
    diff: list[list[int]] = []
    for y in range(len(target)):
        row: list[int] = []
        for x in range(len(target[0])):
            row.append(predicted[y][x] - target[y][x])
        diff.append(row)
    return diff


def _print_human_report(
    *,
    best: Individual,
    expression: str,
    target: list[list[int]],
    predicted: list[list[int]],
    diff: list[list[int]],
    elapsed: float,
    generations: int,
    exact_match: bool,
    workers: int,
    fast_mode: bool,
    seed: int | None,
    verbose: bool,
    acceleration_backend: str,
) -> None:
    print("=" * 72)
    print("Final FuncFind Results")
    print("=" * 72)
    print(f"Expression: {expression}")
    print(f"Fitness: {best.fitness:.4f}")
    print(f"Raw error: {best.raw_error:.0f}")
    print(f"Symmetry error: {best.symmetry_error:.0f}")
    print(f"Complexity: {best.complexity}")
    print(f"Exact match: {exact_match}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Generations executed: {generations}")
    print(f"Workers: {workers} | Fast mode: {fast_mode}")
    print(f"Acceleration backend: {acceleration_backend}")
    print(f"Free-threaded runtime: {_is_free_threaded()}")
    if seed is not None:
        print(f"Seed: {seed}")
    if verbose:
        print(f"Python: {sys.version.split()[0]}")
    print()

    print("Target grid:")
    for row in target:
        print("  ", row)

    print("\nPredicted grid:")
    for row in predicted:
        print("  ", row)

    print("\nDifference grid (. means exact):")
    for row in diff:
        rendered = []
        for value in row:
            if value == 0:
                rendered.append("  .")
            else:
                rendered.append(f"{value:+3d}")
        print("  ", " ".join(rendered))

    print("=" * 72)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final Numba-first genetic function finder. "
            "Defaults are balanced for quality and runtime, with a Python fallback on 3.15+."
        )
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help=f"Maximum generations (default: {DEFAULT_GENERATIONS})",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION,
        help=f"Population size (default: {DEFAULT_POPULATION})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic runs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker threads for population scoring. 0 uses auto.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use speed-biased preset values when defaults are in use.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress each generation checkpoint.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit final result as JSON.",
    )
    parser.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help="Optional JSON or Python-literal file containing the target grid.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.population < 4:
        raise SystemExit("--population must be >= 4")
    if args.generations < 1:
        raise SystemExit("--generations must be >= 1")
    if args.workers < 0:
        raise SystemExit("--workers must be >= 0")

    _enforce_acceleration_policy()
    acceleration_backend = "numba-jit" if HAS_NUMBA else "python-fallback"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed & 0xFFFFFFFF)

    target_grid = GRID
    if args.target_file is not None:
        if not args.target_file.exists():
            raise SystemExit(f"Target file does not exist: {args.target_file}")
        target_grid = _load_grid(args.target_file)

    population = args.population
    generations = args.generations
    mutation_rate = MUTATION_RATE

    if args.fast:
        if args.population == DEFAULT_POPULATION:
            population = FAST_POPULATION
        if args.generations == DEFAULT_GENERATIONS:
            generations = FAST_GENERATIONS
        mutation_rate = min(0.55, mutation_rate * 1.15)

    workers = args.workers if args.workers > 0 else _auto_workers(population)

    if not args.json:
        print("=" * 72)
        print("Final FuncFind")
        print("=" * 72)
        print(f"Python: {sys.version.split()[0]}")
        print(f"Numba available: {HAS_NUMBA}")
        print(f"Acceleration backend: {acceleration_backend}")
        print(f"Population: {population} | Generations: {generations}")
        print(f"Workers: {workers} | Fast mode: {args.fast}")
        print(f"Free-threaded runtime: {_is_free_threaded()}")
        if args.seed is not None:
            print(f"Seed: {args.seed}")
        if len(target_grid) == len(target_grid[0]):
            print("Symmetry scoring: enabled")
        else:
            print("Symmetry scoring: disabled (non-square target)")
        print()

    solver = GeneticApproximator(
        grid=target_grid,
        population_size=population,
        mutation_rate=mutation_rate,
        survival_rate=SURVIVAL_RATE,
        complexity_weight=COMPLEXITY_WEIGHT,
        diversity_weight=DIVERSITY_WEIGHT,
        symmetry_weight=SYMMETRY_WEIGHT,
        workers=workers,
        verbose=args.verbose and not args.json,
    )

    started = time.perf_counter()
    best = solver.run(generations=generations)
    elapsed = time.perf_counter() - started

    expression = simplify_node(best.gene)
    predicted = _predict_grid(best, solver.width, solver.height)
    diff = _difference_grid(predicted, target_grid)

    report = {
        "expression": expression,
        "fitness": float(best.fitness),
        "raw_error": float(best.raw_error),
        "symmetry_error": float(best.symmetry_error),
        "complexity": int(best.complexity),
        "exact_match": bool(
            best.last_outputs is not None and np.array_equal(best.last_outputs, solver.expected_flat)
        ),
        "elapsed_seconds": elapsed,
        "generations_executed": int(solver.generations_completed),
        "population": population,
        "workers": workers,
        "fast_mode": bool(args.fast),
        "seed": args.seed,
        "numba_available": HAS_NUMBA,
        "acceleration_backend": acceleration_backend,
        "free_threaded_runtime": _is_free_threaded(),
        "target_shape": [solver.height, solver.width],
        "target_grid": target_grid,
        "predicted_grid": predicted,
        "difference_grid": diff,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human_report(
            best=best,
            expression=expression,
            target=target_grid,
            predicted=predicted,
            diff=diff,
            elapsed=elapsed,
            generations=solver.generations_completed,
            exact_match=report["exact_match"],
            workers=workers,
            fast_mode=args.fast,
            seed=args.seed,
            verbose=args.verbose,
            acceleration_backend=acceleration_backend,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
