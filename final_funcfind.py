#!/usr/bin/env python3
"""Lean, integrated high-performance function finder with always-on JSON run logs."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Self

import numpy as np

_FORCE_DISABLE_NUMBA = (
    os.environ.get("FUNCFIND_DISABLE_NUMBA", "").strip() == "1"
)

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
# Constants
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

RUNS_DIR: Final[Path] = Path("runs")

DEFAULT_POPULATION = 640
DEFAULT_GENERATIONS = 35_000
DEFAULT_MUTATION_RATE = 0.30

FAST_POPULATION = 160
FAST_GENERATIONS = 3_500
FAST_MUTATION_RATE = 0.33

SURVIVAL_RATE = 0.25
MAX_DEPTH = 8
COMPLEXITY_WEIGHT = 0.35
DIVERSITY_WEIGHT = 5.5
SYMMETRY_WEIGHT = 80.0
TOURNAMENT_SIZE = 7

DEFAULT_CROSSOVER_RATE = 0.35
DEFAULT_STAGNATION_WINDOW = 30
DEFAULT_STAGNATION_THRESHOLD = 1.0
DEFAULT_STAGNATION_REFRESH = 0.15

OUTPUT_ABS_CLIP = 20_000
POW_EXP_CLIP = 5
OUT_OF_RANGE_LOW = -2
OUT_OF_RANGE_HIGH = 10
RANGE_TARGET = 3
RANGE_WEIGHT = 0.25
SCORE_CACHE_LIMIT = 20_000

PATTERN_TYPES = (
    "x_plus_y",
    "x_times_y",
    "x2_plus_y2",
    "xy_combo",
    "difference_squared",
    "symmetric_poly",
)

# Postfix VM opcodes
OP_CONST = np.int64(0)
OP_X = np.int64(1)
OP_Y = np.int64(2)
OP_ADD = np.int64(3)
OP_SUB = np.int64(4)
OP_MUL = np.int64(5)
OP_POW = np.int64(6)
OP_DIV = np.int64(7)
OP_MOD = np.int64(8)

OP_CONST_I = int(OP_CONST)
OP_X_I = int(OP_X)
OP_Y_I = int(OP_Y)
OP_ADD_I = int(OP_ADD)
OP_SUB_I = int(OP_SUB)
OP_MUL_I = int(OP_MUL)
OP_POW_I = int(OP_POW)
OP_DIV_I = int(OP_DIV)
OP_MOD_I = int(OP_MOD)


# ---------------------------------------------------------------------------
# VM core
# ---------------------------------------------------------------------------
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
    if (a < 0 < b) or (a > 0 > b):
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
) -> tuple[np.float64, np.int64, np.int64, np.ndarray]:
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

    return fitness, raw_error, symmetry_error, outputs


def _clip_int_py(value: int, abs_limit: int) -> int:
    if value > abs_limit:
        return abs_limit
    if value < -abs_limit:
        return -abs_limit
    return value


def _safe_pow_py(base: int, exp: int, max_exp: int, abs_limit: int) -> int:
    if exp < 0:
        return 0
    if exp > max_exp:
        exp = max_exp

    result = 1
    for _ in range(exp):
        result = _clip_int_py(result * base, abs_limit)
    return result


def _trunc_div_py(a: int, b: int) -> int:
    if b == 0:
        return 0
    q = abs(a) // abs(b)
    if (a < 0 < b) or (a > 0 > b):
        q = -q
    return q


def _eval_program_point_py(
    opcodes: tuple[int, ...],
    operands: tuple[int, ...],
    x_val: int,
    y_val: int,
    stack: list[int],
    abs_limit: int,
    max_exp: int,
) -> int:
    sp = 0
    for op, arg in zip(opcodes, operands):
        if op == OP_CONST_I:
            stack[sp] = arg
            sp += 1
            continue
        if op == OP_X_I:
            stack[sp] = x_val
            sp += 1
            continue
        if op == OP_Y_I:
            stack[sp] = y_val
            sp += 1
            continue

        if sp < 2:
            return 0

        right = stack[sp - 1]
        left = stack[sp - 2]

        if op == OP_ADD_I:
            stack[sp - 2] = _clip_int_py(left + right, abs_limit)
        elif op == OP_SUB_I:
            stack[sp - 2] = _clip_int_py(left - right, abs_limit)
        elif op == OP_MUL_I:
            stack[sp - 2] = _clip_int_py(left * right, abs_limit)
        elif op == OP_POW_I:
            stack[sp - 2] = _safe_pow_py(left, right, max_exp, abs_limit)
        elif op == OP_DIV_I:
            stack[sp - 2] = _clip_int_py(_trunc_div_py(left, right), abs_limit)
        elif op == OP_MOD_I:
            stack[sp - 2] = 0 if right == 0 else _clip_int_py(left % right, abs_limit)
        else:
            stack[sp - 2] = 0

        sp -= 1

    if sp != 1:
        return 0
    return stack[0]


def _score_program_py(
    opcodes: tuple[int, ...],
    operands: tuple[int, ...],
    xs: list[int],
    ys: list[int],
    expected_flat: list[int],
    symmetry_left: list[int],
    symmetry_right: list[int],
    complexity_weight: float,
    diversity_weight: float,
    symmetry_weight: float,
    range_weight: float,
    out_low: int,
    out_high: int,
    range_target: int,
    abs_limit: int,
    max_exp: int,
) -> tuple[float, int, int, np.ndarray]:
    n = len(expected_flat)
    outputs = [0] * n
    stack = [0] * (len(opcodes) + 2)

    raw_error = 0
    diversity_penalty = 0
    range_penalty = 0

    prev = 0
    for i in range(n):
        out = _eval_program_point_py(
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

    symmetry_error = 0
    for l_idx, r_idx in zip(symmetry_left, symmetry_right):
        sdiff = outputs[l_idx] - outputs[r_idx]
        if sdiff < 0:
            sdiff = -sdiff
        symmetry_error += sdiff

    complexity = len(opcodes)
    fitness = (
        float(raw_error)
        + symmetry_weight * float(symmetry_error)
        + diversity_weight * float(diversity_penalty)
        + complexity_weight * float(complexity)
        + range_weight * float(range_penalty)
    )

    return fitness, raw_error, symmetry_error, np.asarray(outputs, dtype=np.int64)


def _ensure_numba_ready() -> None:
    if not HAS_NUMBA:
        raise RuntimeError("Numba unavailable")

    xs = np.array([0, 1, 2], dtype=np.int64)
    ys = np.array([0, 1, 2], dtype=np.int64)
    expected = np.array([0, 2, 4], dtype=np.int64)
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
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Numba JIT warmup failed: {exc}") from exc


def _enforce_acceleration_policy() -> str | None:
    if HAS_NUMBA:
        _ensure_numba_ready()
        return None

    detail = f" Original import error: {NUMBA_IMPORT_ERROR}" if NUMBA_IMPORT_ERROR else ""
    return (
        "Warning: running Python fallback mode (Numba unavailable). "
        "Use a runtime with supported acceleration for maximum speed." + detail
    )


# ---------------------------------------------------------------------------
# GP tree model
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


class ConstNode(Node):
    __slots__ = ("value",)
    ALLOWED_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2]

    def __init__(self, value: int | None = None) -> None:
        self.value = random.choice(self.ALLOWED_VALUES) if value is None else int(value)

    def evaluate(self, inputs: tuple[int, int]) -> int:
        return self.value

    def clone(self) -> Self:
        return ConstNode(self.value)

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate:
            if random.random() < 0.6 and -4 <= self.value <= 10:
                jitter = random.choice((-1, 1))
                return ConstNode(max(-4, min(10, self.value + jitter)))
            return ConstNode(random.choice(self.ALLOWED_VALUES))
        return self.clone()

    def complexity(self) -> int:
        return 1

    def to_string(self) -> str:
        return str(self.value)

    def contains_variable(self) -> bool:
        return False

    def get_depth(self) -> int:
        return 0


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


class DivNode(BinaryNode):
    __slots__ = ()

    def evaluate(self, inputs: tuple[int, int]) -> int:
        left = self.left.evaluate(inputs)
        right = self.right.evaluate(inputs)
        if right == 0:
            return 0
        q = abs(left) // abs(right)
        if (left < 0 < right) or (left > 0 > right):
            q = -q
        return q

    def mutate(self, p_mutate: float) -> Node:
        if random.random() < p_mutate * 0.15:
            return random_node(MAX_DEPTH)
        left_mut, right_mut = self.mutate_children(p_mutate)
        return DivNode(left_mut, right_mut)

    def to_string(self) -> str:
        return f"({self.left.to_string()}/{self.right.to_string()})"


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


def create_seed_pattern(pattern_type: str) -> Node:
    x_var = VarNode(0)
    y_var = VarNode(1)

    match pattern_type:
        case "x_plus_y":
            return AddNode(x_var, y_var)
        case "x_times_y":
            return MulNode(x_var, y_var)
        case "x2_plus_y2":
            return AddNode(PowNode(x_var, ConstNode(2)), PowNode(y_var, ConstNode(2)))
        case "xy_combo":
            return AddNode(AddNode(x_var, y_var), MulNode(x_var, y_var))
        case "difference_squared":
            return PowNode(SubNode(x_var, y_var), ConstNode(2))
        case "symmetric_poly":
            sum_sq = AddNode(PowNode(x_var, ConstNode(2)), PowNode(y_var, ConstNode(2)))
            weighted_sum = MulNode(ConstNode(random.randint(-2, 3)), AddNode(x_var, y_var))
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
    return np.asarray(opcodes, dtype=np.int64), np.asarray(operands, dtype=np.int64)


PathRef = tuple[str, ...]


def _collect_subtree_paths(node: Node, path: PathRef = ()) -> list[PathRef]:
    paths: list[PathRef] = [path]
    if isinstance(node, BinaryNode):
        paths.extend(_collect_subtree_paths(node.left, path + ("left",)))
        paths.extend(_collect_subtree_paths(node.right, path + ("right",)))
    return paths


def _get_subtree(node: Node, path: PathRef) -> Node:
    current = node
    for step in path:
        current = getattr(current, step)
    return current


def _replace_subtree(root: Node, path: PathRef, replacement: Node) -> Node:
    if not path:
        return replacement

    new_root = root.clone()
    parent = new_root
    for step in path[:-1]:
        parent = getattr(parent, step)
    setattr(parent, path[-1], replacement)
    return new_root


# ---------------------------------------------------------------------------
# GA engine
# ---------------------------------------------------------------------------
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
    opcodes_py: tuple[int, ...] | None = None
    operands_py: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class SolverConfig:
    population: int
    generations: int
    mutation_rate: float
    survival_rate: float = SURVIVAL_RATE
    complexity_weight: float = COMPLEXITY_WEIGHT
    diversity_weight: float = DIVERSITY_WEIGHT
    symmetry_weight: float = SYMMETRY_WEIGHT
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    stagnation_window: int = DEFAULT_STAGNATION_WINDOW
    stagnation_threshold: float = DEFAULT_STAGNATION_THRESHOLD
    stagnation_refresh: float = DEFAULT_STAGNATION_REFRESH


class GeneticApproximator:
    def __init__(self, grid: list[list[int]], cfg: SolverConfig) -> None:
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
        self.x_inputs_py: list[int] | None = None
        self.y_inputs_py: list[int] | None = None
        self.expected_flat_py: list[int] | None = None
        self.symmetry_left_py: list[int] | None = None
        self.symmetry_right_py: list[int] | None = None
        if not HAS_NUMBA:
            self.x_inputs_py = self.x_inputs.tolist()
            self.y_inputs_py = self.y_inputs.tolist()
            self.expected_flat_py = self.expected_flat.tolist()
            self.symmetry_left_py = self.symmetry_left.tolist()
            self.symmetry_right_py = self.symmetry_right.tolist()

        self.population_size = max(4, cfg.population)
        self.mutation_rate = cfg.mutation_rate
        self.base_mutation_rate = cfg.mutation_rate
        self.survival_rate = cfg.survival_rate
        self.complexity_weight = cfg.complexity_weight
        self.diversity_weight = cfg.diversity_weight
        self.symmetry_weight = cfg.symmetry_weight
        self.crossover_rate = max(0.0, min(1.0, cfg.crossover_rate))
        self.stagnation_window = max(2, cfg.stagnation_window)
        self.stagnation_threshold = cfg.stagnation_threshold
        self.stagnation_refresh = max(0.02, min(0.50, cfg.stagnation_refresh))
        self.stagnation_trigger_rounds = max(6, self.stagnation_window // 4)
        self._complexity_weight64 = np.float64(self.complexity_weight)
        self._diversity_weight64 = np.float64(self.diversity_weight)
        self._symmetry_weight64 = np.float64(self.symmetry_weight)
        self._range_weight64 = np.float64(RANGE_WEIGHT)
        self._out_low64 = np.int64(OUT_OF_RANGE_LOW)
        self._out_high64 = np.int64(OUT_OF_RANGE_HIGH)
        self._range_target64 = np.int64(RANGE_TARGET)
        self._output_abs_clip64 = np.int64(OUTPUT_ABS_CLIP)
        self._pow_exp_clip64 = np.int64(POW_EXP_CLIP)

        self.best_fitness_history: list[float] = []
        self.stagnation_counter = 0
        self.last_recent_improvement = 0.0
        self.best_seen: Individual | None = None
        self.generations_completed = 0
        self.exact_match = False
        self.crossover_events = 0
        self.escape_events = 0
        self.score_cache_hits = 0
        self.score_cache_misses = 0
        self._score_cache: OrderedDict[
            tuple[bytes, bytes],
            tuple[float, float, float, np.ndarray, int],
        ] = OrderedDict()
        self._score_cache_limit = SCORE_CACHE_LIMIT

        self.population: list[Individual] = []
        seed_count = self.population_size // 2

        for _ in range(seed_count):
            self.population.append(Individual(gene=create_seed_pattern(random.choice(PATTERN_TYPES))))
        for _ in range(self.population_size - seed_count):
            self.population.append(Individual(gene=random_node(max_depth=MAX_DEPTH)))

    def _score_individual(self, individual: Individual) -> None:
        if (
            individual.last_outputs is not None
            and individual.opcodes is not None
            and individual.operands is not None
            and individual.fitness != float("inf")
        ):
            return

        if individual.opcodes is None or individual.operands is None:
            opcodes, operands = compile_gene_to_postfix(individual.gene)
            individual.opcodes = opcodes
            individual.operands = operands
            if not HAS_NUMBA:
                individual.opcodes_py = tuple(opcodes.tolist())
                individual.operands_py = tuple(operands.tolist())

        cache_key = (individual.opcodes.tobytes(), individual.operands.tobytes())
        cached = self._score_cache.get(cache_key)
        if cached is not None:
            self.score_cache_hits += 1
            self._score_cache.move_to_end(cache_key)
            fitness, raw_error, sym_error, outputs, complexity = cached
            individual.fitness = fitness
            individual.raw_error = raw_error
            individual.symmetry_error = sym_error
            individual.last_outputs = outputs
            individual.complexity = complexity
            return

        self.score_cache_misses += 1
        if HAS_NUMBA:
            fitness, raw_error, sym_error, outputs = _score_program(
                individual.opcodes,
                individual.operands,
                self.x_inputs,
                self.y_inputs,
                self.expected_flat,
                self.symmetry_left,
                self.symmetry_right,
                self._complexity_weight64,
                self._diversity_weight64,
                self._symmetry_weight64,
                self._range_weight64,
                self._out_low64,
                self._out_high64,
                self._range_target64,
                self._output_abs_clip64,
                self._pow_exp_clip64,
            )
        else:
            assert individual.opcodes_py is not None
            assert individual.operands_py is not None
            assert self.x_inputs_py is not None
            assert self.y_inputs_py is not None
            assert self.expected_flat_py is not None
            assert self.symmetry_left_py is not None
            assert self.symmetry_right_py is not None
            fitness, raw_error, sym_error, outputs = _score_program_py(
                individual.opcodes_py,
                individual.operands_py,
                self.x_inputs_py,
                self.y_inputs_py,
                self.expected_flat_py,
                self.symmetry_left_py,
                self.symmetry_right_py,
                self.complexity_weight,
                self.diversity_weight,
                self.symmetry_weight,
                RANGE_WEIGHT,
                OUT_OF_RANGE_LOW,
                OUT_OF_RANGE_HIGH,
                RANGE_TARGET,
                OUTPUT_ABS_CLIP,
                POW_EXP_CLIP,
            )

        scored_fitness = float(fitness)
        scored_raw_error = float(raw_error)
        scored_symmetry_error = float(sym_error)
        scored_complexity = int(individual.opcodes.shape[0])

        individual.fitness = scored_fitness
        individual.raw_error = scored_raw_error
        individual.symmetry_error = scored_symmetry_error
        individual.last_outputs = outputs
        individual.complexity = scored_complexity

        self._score_cache[cache_key] = (
            scored_fitness,
            scored_raw_error,
            scored_symmetry_error,
            outputs,
            scored_complexity,
        )
        if len(self._score_cache) > self._score_cache_limit:
            self._score_cache.popitem(last=False)

    def evaluate_population(self) -> None:
        for individual in self.population:
            self._score_individual(individual)
        self.population.sort(key=lambda ind: ind.fitness)

    def tournament_selection(self) -> Individual:
        pool_limit = max(2, len(self.population) // 2)
        pool = self.population[:pool_limit]
        k = min(TOURNAMENT_SIZE, len(pool))
        tournament = random.sample(pool, k)
        return min(tournament, key=lambda ind: ind.fitness)

    def _clone_elite(self, parent: Individual) -> Individual:
        elite = Individual(gene=parent.gene.clone())
        elite.fitness = parent.fitness
        elite.raw_error = parent.raw_error
        elite.symmetry_error = parent.symmetry_error
        elite.complexity = parent.complexity
        elite.last_outputs = parent.last_outputs
        if parent.opcodes is not None and parent.operands is not None:
            elite.opcodes = parent.opcodes
            elite.operands = parent.operands
            elite.opcodes_py = parent.opcodes_py
            elite.operands_py = parent.operands_py
        return elite

    def _bounded_gene(self, gene: Node) -> Node:
        if gene.get_depth() > MAX_DEPTH:
            gene = random_node(max_depth=MAX_DEPTH)
        if gene.contains_variable():
            return gene
        if gene.get_depth() < MAX_DEPTH:
            candidate = AddNode(gene.clone(), VarNode(0))
            if candidate.get_depth() <= MAX_DEPTH:
                return candidate
        for _ in range(8):
            candidate = random_node(max_depth=MAX_DEPTH)
            if candidate.contains_variable():
                return candidate
        return VarNode(0)

    def crossover(self, parent_a: Individual, parent_b: Individual, max_attempts: int = 8) -> Node:
        paths_a = _collect_subtree_paths(parent_a.gene)
        paths_b = _collect_subtree_paths(parent_b.gene)

        for _ in range(max_attempts):
            path_a = random.choice(paths_a)
            path_b = random.choice(paths_b)
            donor_subtree = _get_subtree(parent_b.gene, path_b).clone()
            candidate = _replace_subtree(parent_a.gene, path_a, donor_subtree)
            if candidate.get_depth() <= MAX_DEPTH and candidate.contains_variable():
                return candidate

        return parent_a.gene.mutate(self.mutation_rate)

    def _targeted_edit(self, gene: Node) -> Node:
        working = gene.clone()
        roll = random.random()

        if roll < 0.40:
            const_paths = [
                path
                for path in _collect_subtree_paths(working)
                if isinstance(_get_subtree(working, path), ConstNode)
            ]
            if const_paths:
                path = random.choice(const_paths)
                old_const = _get_subtree(working, path)
                assert isinstance(old_const, ConstNode)
                jitter = random.choice((-2, -1, 1, 2))
                replacement = ConstNode(max(-6, min(12, old_const.value + jitter)))
                return self._bounded_gene(_replace_subtree(working, path, replacement))

        if roll < 0.80:
            binary_paths = [
                path
                for path in _collect_subtree_paths(working)
                if isinstance(_get_subtree(working, path), BinaryNode)
            ]
            if binary_paths:
                path = random.choice(binary_paths)
                old_node = _get_subtree(working, path)
                assert isinstance(old_node, BinaryNode)
                alternatives = [op for op in BINARY_NODE_TYPES if not isinstance(old_node, op)]
                replacement_op = random.choice(alternatives)
                replacement = replacement_op(old_node.left.clone(), old_node.right.clone())
                return self._bounded_gene(_replace_subtree(working, path, replacement))

        random_paths = _collect_subtree_paths(working)
        path = random.choice(random_paths)
        replacement = random_node(max_depth=min(3, MAX_DEPTH))
        return self._bounded_gene(_replace_subtree(working, path, replacement))

    def _apply_stagnation_escape(self) -> None:
        self.escape_events += 1
        replace_count = max(1, int(self.population_size * self.stagnation_refresh))
        start_idx = max(0, self.population_size - replace_count)

        for idx in range(start_idx, self.population_size):
            if random.random() < 0.70:
                donor = self.population[random.randrange(max(2, self.population_size // 2))]
                new_gene = self._targeted_edit(donor.gene)
            elif random.random() < 0.50:
                new_gene = create_seed_pattern(random.choice(PATTERN_TYPES))
            else:
                new_gene = random_node(max_depth=MAX_DEPTH)

            self.population[idx] = Individual(gene=self._bounded_gene(new_gene))

        self.mutation_rate = min(0.72, self.mutation_rate * 1.08)

    def reproduce(self) -> None:
        survivors = max(2, int(self.population_size * self.survival_rate))
        elite_count = min(12, survivors, self.population_size)

        new_population: list[Individual] = []
        for i in range(elite_count):
            new_population.append(self._clone_elite(self.population[i]))

        while len(new_population) < self.population_size:
            use_crossover = random.random() < self.crossover_rate and len(self.population) > 1
            if use_crossover:
                parent_a = self.tournament_selection()
                parent_b = self.tournament_selection()
                child_gene = self.crossover(parent_a, parent_b)
                child_gene = child_gene.mutate(max(0.04, self.mutation_rate * 0.35))
                self.crossover_events += 1
            else:
                parent = self.tournament_selection()
                child_gene = parent.gene.mutate(self.mutation_rate)

            new_population.append(Individual(gene=self._bounded_gene(child_gene)))

        self.population = new_population

    def adapt_parameters(self) -> None:
        if len(self.best_fitness_history) >= self.stagnation_window:
            recent_improvement = (
                self.best_fitness_history[-self.stagnation_window] - self.best_fitness_history[-1]
            )
            self.last_recent_improvement = recent_improvement

            if recent_improvement < self.stagnation_threshold:
                self.stagnation_counter += 1
                self.mutation_rate = min(0.68, self.mutation_rate * 1.02)
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
                self.mutation_rate = max(0.10, self.mutation_rate * 0.992)

            if self.stagnation_counter >= self.stagnation_trigger_rounds:
                self._apply_stagnation_escape()
                self.stagnation_counter = 0

        if self.mutation_rate > self.base_mutation_rate:
            self.mutation_rate = max(self.base_mutation_rate, self.mutation_rate * 0.998)
        else:
            self.mutation_rate = min(self.base_mutation_rate, self.mutation_rate * 1.001)
        self.mutation_rate = min(0.75, max(0.08, self.mutation_rate))

    def run(self, generations: int) -> Individual:
        try:
            for gen in range(1, generations + 1):
                self.evaluate_population()
                best = self.population[0]
                self.generations_completed = gen

                self.best_fitness_history.append(best.fitness)
                if self.best_seen is None or best.fitness < self.best_seen.fitness:
                    self.best_seen = best

                if best.last_outputs is not None and np.array_equal(best.last_outputs, self.expected_flat):
                    self.exact_match = True
                    return best
                if best.raw_error == 0.0 and best.symmetry_error == 0.0:
                    return best

                self.reproduce()
                self.adapt_parameters()

        except KeyboardInterrupt:
            pass

        self.evaluate_population()
        final_best = self.population[0]
        if self.best_seen is None:
            return final_best
        return self.best_seen if self.best_seen.fitness <= final_best.fitness else final_best


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------
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


def _resolve_solver_config(fast: bool) -> SolverConfig:
    if fast:
        return SolverConfig(
            population=FAST_POPULATION,
            generations=FAST_GENERATIONS,
            mutation_rate=FAST_MUTATION_RATE,
        )
    return SolverConfig(
        population=DEFAULT_POPULATION,
        generations=DEFAULT_GENERATIONS,
        mutation_rate=DEFAULT_MUTATION_RATE,
    )


def _new_run_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / f"{ts}-{rid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _run_once(
    *,
    grid: list[list[int]],
    cfg: SolverConfig,
    seed: int | None,
    backend: str,
) -> dict[str, object]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

    solver = GeneticApproximator(grid=grid, cfg=cfg)

    started = time.perf_counter()
    best = solver.run(cfg.generations)
    elapsed = time.perf_counter() - started

    exact_match = bool(best.last_outputs is not None and np.array_equal(best.last_outputs, solver.expected_flat))

    return {
        "seed": seed,
        "fitness": float(best.fitness),
        "raw_error": float(best.raw_error),
        "symmetry_error": float(best.symmetry_error),
        "complexity": int(best.complexity),
        "exact_match": exact_match,
        "elapsed_seconds": elapsed,
        "generations_executed": int(solver.generations_completed),
        "population": cfg.population,
        "expression": best.gene.to_string(),
        "crossover_events": int(solver.crossover_events),
        "escape_events": int(solver.escape_events),
        "score_cache_hits": int(solver.score_cache_hits),
        "score_cache_misses": int(solver.score_cache_misses),
        "score_cache_size": int(len(solver._score_cache)),
        "recent_improvement_window_delta": float(solver.last_recent_improvement),
        "acceleration_backend": backend,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final function finder with integrated run pipeline. "
            "Always writes JSON logs under runs/<run-id>/run.json."
        )
    )
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--fast", action="store_true", help="Use faster preset")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    runtime_warning = _enforce_acceleration_policy()
    backend = "numba-jit" if HAS_NUMBA else "python-fallback"

    target_grid = GRID

    cfg = _resolve_solver_config(args.fast)
    seed = args.seed

    run_dir = _new_run_dir()
    run_json_path = run_dir / "run.json"

    if runtime_warning:
        print(runtime_warning)

    print(f"[run] {run_dir}")
    print(
        f"[config] mode=single "
        f"pop={cfg.population} gen={cfg.generations} fast={args.fast} backend={backend}"
    )

    result = _run_once(grid=target_grid, cfg=cfg, seed=seed, backend=backend)

    payload: dict[str, object] = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "mode": "single",
        "backend": {
            "name": backend,
            "numba_available": HAS_NUMBA,
            "python": sys.version.split()[0],
        },
        "config": {
            "fast": bool(args.fast),
            "population": cfg.population,
            "generations": cfg.generations,
            "mutation_rate": cfg.mutation_rate,
            "survival_rate": cfg.survival_rate,
            "crossover_rate": cfg.crossover_rate,
            "stagnation_window": cfg.stagnation_window,
            "stagnation_threshold": cfg.stagnation_threshold,
            "stagnation_refresh": cfg.stagnation_refresh,
            "seed": seed,
        },
        "target_shape": [len(target_grid), len(target_grid[0])],
    }

    if runtime_warning:
        payload["runtime_warning"] = runtime_warning

    payload["results"] = result
    print(
        f"[result] raw={result['raw_error']:.3f} fit={result['fitness']:.3f} "
        f"exact={result['exact_match']} seed={result['seed']}"
    )
    print(f"[expr] {result['expression']}")

    _write_json(run_json_path, payload)
    print(f"[log] {run_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
