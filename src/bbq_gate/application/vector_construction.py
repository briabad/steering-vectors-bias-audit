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
    """Task 4.6 (ORIGINAL, retired as a gate -- see `MarginCheck` below):
    evaluate every declared baseline against the fixed absolute leak
    threshold.

    Kept for its own regression test coverage and as a diagnostic print in
    the pipeline (`scripts/06_build_operation_vector.py`), but no longer used
    to abort the pipeline: `EXP-002/hypothesis.md` Addendum 7, Revision 1,
    found the absolute threshold could not distinguish a real lexical leak
    (baseline ~= direction, e.g. 0.9079 vs 0.9174) from an intrinsic, expected
    lexical signal for this contrast (baseline 0.7576, direction 0.8532 --
    the model's behaviour genuinely depends on who is named). The pipeline's
    actual gate is `best_baseline_margin_check`.

    Returns:
        The list of `BaselineCheck`s that are possible leaks (empty if none).
    """
    return [
        BaselineCheck(name=name, auc=auc)
        for name, auc in baselines.items()
        if auc > BASELINE_LEAK_THRESHOLD
    ]


@dataclass(frozen=True)
class MarginCheck:
    """`EXP-002/hypothesis.md` Addendum 7 (Revisions 1-2): the margin of the
    direction's holdout AUC over ONE baseline's holdout AUC, with a 95%
    bootstrap CI obtained by resampling holdout pairs jointly for both AUCs
    (`domain.directions.bootstrap_margin_ci`). This is the pipeline's actual
    abort gate, replacing the retired absolute `BASELINE_LEAK_THRESHOLD`.
    """

    baseline_name: str
    baseline_auc: float
    margin_mean: float
    margin_ci_lower: float
    margin_ci_upper: float

    @property
    def excludes_zero(self) -> bool:
        """True if the margin's 95% CI does not straddle 0 -- i.e. the
        direction's advantage over this baseline is distinguishable from
        sampling noise at this holdout size."""
        return self.margin_ci_lower > 0.0 or self.margin_ci_upper < 0.0


def best_baseline_margin_check(checks: Sequence[MarginCheck]) -> MarginCheck:
    """Pick the 'mejor baseline' for the abort gate: the one HARDEST to
    beat, i.e. the one with the SMALLEST margin (a smaller margin means a
    stronger, more competitive baseline). Gating on the smallest margin is
    the conservative choice -- if the direction clears its toughest
    baseline with a margin distinguishable from zero, it clears the rest
    too.
    """
    if not checks:
        raise ValueError("best_baseline_margin_check requires at least one MarginCheck")
    return min(checks, key=lambda c: c.margin_mean)
