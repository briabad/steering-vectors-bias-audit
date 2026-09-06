"""Layer selection and H1 evaluation over already-extracted activations
(tasks 4.1-4.6). Pure indexing/aggregation over in-memory arrays: testable
with tiny synthetic activations, no GPU (design.md Goals).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from bbq_gate.application.contrast_builder import PairedExample
from bbq_gate.domain.directions import Pair, Vector, auc_of_direction
from bbq_gate.domain.entities import BBQItem

# spec/task 4.6: a baseline above this AUC on the holdout is a possible leak
# in the contrast design and must halt the pipeline, not be reported as a
# footnote.
BASELINE_LEAK_THRESHOLD = 0.65

ActivationsByKey = dict[str, np.ndarray]  # item_key -> array shape (n_layers, dim)
ConstructFn = Callable[[Sequence[Pair]], Vector]


def item_key(item: BBQItem) -> str:
    """Globally unique key for a BBQItem: `item_id` alone is only unique
    within (category, context_condition) -- see BBQItem.item_id docstring."""
    return f"{item.category}|{item.context_condition}|{item.item_id}"


@dataclass(frozen=True)
class StratifiedSplit:
    """Task 4.1: three disjoint parts, stratified by category."""

    sub_adjuste: tuple[PairedExample, ...]
    layer_selection: tuple[PairedExample, ...]
    holdout: tuple[PairedExample, ...]


def stratified_split(
    examples: Sequence[PairedExample],
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
    seed: int = 20260906,
) -> StratifiedSplit:
    """Split pairs into sub-ajuste / selección-de-capa / holdout, stratified
    by category, so the holdout is never touched before the layer is frozen
    (spec: "Selección de capa sin contaminar la evaluación").

    Args:
        examples: The full ajuste-eligible pair list (already excludes
            reserved categories -- see `domain.contrast.ContrastDataset`).
        fractions: (sub_adjuste, layer_selection, holdout) proportions,
            must sum to 1.0.
        seed: RNG seed for the per-category shuffle.

    Returns:
        StratifiedSplit with three pairwise-disjoint, category-proportional lists.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")

    by_category: dict[str, list[PairedExample]] = {}
    for ex in examples:
        by_category.setdefault(ex.category, []).append(ex)

    rng = random.Random(seed)
    sub_adjuste: list[PairedExample] = []
    layer_selection: list[PairedExample] = []
    holdout: list[PairedExample] = []
    for _category, items in by_category.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_sub = round(n * fractions[0])
        n_sel = round(n * fractions[1])
        sub_adjuste.extend(shuffled[:n_sub])
        layer_selection.extend(shuffled[n_sub : n_sub + n_sel])
        holdout.extend(shuffled[n_sub + n_sel :])

    return StratifiedSplit(
        sub_adjuste=tuple(sub_adjuste), layer_selection=tuple(layer_selection), holdout=tuple(holdout)
    )


def pairs_at_layer(
    examples: Sequence[PairedExample], activations: ActivationsByKey, layer: int
) -> list[Pair]:
    """Look up the (positive, negative) activation vectors at `layer` for
    each example."""
    return [
        (activations[item_key(ex.positive)][layer], activations[item_key(ex.negative)][layer])
        for ex in examples
    ]


def layer_auc_curve(
    sub_adjuste: Sequence[PairedExample],
    layer_selection: Sequence[PairedExample],
    activations: ActivationsByKey,
    construct_fn: ConstructFn,
    n_layers: int,
) -> dict[int, float]:
    """Task 4.2: AUC per layer, built on `sub_adjuste`, evaluated on
    `layer_selection` -- the reserved cut, never the holdout (spec:
    "Registro de la curva": "se expone el rendimiento en TODAS las capas
    evaluadas, no solo en la elegida").
    """
    curve: dict[int, float] = {}
    for layer in range(n_layers):
        adj_pairs = pairs_at_layer(sub_adjuste, activations, layer)
        direction = construct_fn(adj_pairs)
        sel_pairs = pairs_at_layer(layer_selection, activations, layer)
        sel_pos = [p for p, _ in sel_pairs]
        sel_neg = [n for _, n in sel_pairs]
        curve[layer] = auc_of_direction(direction, sel_pos, sel_neg)
    return curve


def select_best_layer(curve: dict[int, float]) -> int:
    """Task 4.3: freeze the layer with the highest selection-set AUC, using
    ONLY `curve` (built from sub_adjuste/layer_selection) -- never the
    holdout."""
    return max(curve, key=lambda layer: curve[layer])


@dataclass(frozen=True)
class BaselineCheck:
    """Task 4.6: a baseline above BASELINE_LEAK_THRESHOLD is a possible leak
    and must be flagged, not silently accepted."""

    name: str
    auc: float

    @property
    def is_possible_leak(self) -> bool:
        return self.auc > BASELINE_LEAK_THRESHOLD


def check_baselines_for_leakage(baselines: dict[str, float]) -> list[BaselineCheck]:
    """Task 4.6: evaluate every declared baseline against the leak threshold.

    Returns:
        The list of `BaselineCheck`s that are possible leaks (empty if none).
        Callers MUST stop and report, not continue, if this is non-empty
        (spec: "Baseline inesperadamente fuerte").
    """
    return [
        BaselineCheck(name=name, auc=auc)
        for name, auc in baselines.items()
        if auc > BASELINE_LEAK_THRESHOLD
    ]
