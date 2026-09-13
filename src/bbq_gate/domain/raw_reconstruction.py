"""Positional reconstruction of item identity from the defective raw JSONL.

`existence_gate_raw.jsonl` (287,976 rows) was written by
`scripts/02_bbq_existence_gate.py` with a bug: the persisted `item_id` field
holds BBQ's `question_index` (the question TEMPLATE number, 19-50 distinct
values per category, shared by up to 720 rows), not a unique item identifier.
The loader is fixed (`infrastructure/loaders.py` now uses `example_id`), but
the raw file already on disk still carries the defect and cannot be
re-generated without repeating the ~2h GPU run (design.md D1).

The identity of a distinct BBQ item is recoverable from POSITION, because
`evaluate_items` (application/evaluator.py) writes rows in a fixed nested
order: for each item, for each format, for each permutation order (in
`ORDER_CYCLE`). So within a (category, format, context_condition,
raw_item_id=template) bucket, every consecutive run of 3 rows belongs to one
distinct underlying item, and the K-th such triplet (0-indexed) is the K-th
occurrence of that (category, condition, template) combination in the
original evaluation order.

This module is a straight extraction of `scripts/05_recompute_gate.py`
(`consolidate`/`_consolidate_one`), so the script and this change's contrast
builder share one implementation (design.md D1: "reutilizarse, no
reimplementarse").
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

ORDER_CYCLE: list[list[int]] = [[0, 1, 2], [2, 1, 0], [1, 2, 0]]


class PermutationCycleError(RuntimeError):
    """Raised when a bucket of raw rows does not follow the expected
    ORDER_CYCLE pattern of 3 consecutive permutations per item.

    This is the abort condition design.md D1 requires: silently ignoring a
    broken pattern would misattribute observations across distinct items.
    """


@dataclass(frozen=True)
class ConsolidatedItem:
    """One reconstructed BBQ item, consolidated by majority vote across its
    3 permutations, with its POSITIONAL identity (not a real `example_id`,
    which is unrecoverable from this raw file -- see module docstring).

    Attributes:
        category: BBQ category.
        format: Prompt format ('plano' or 'chat').
        condition: 'ambig' or 'disambig'.
        template: The raw file's `item_id` field, which is actually BBQ's
            `question_index` (the question template number, persisted as a
            string in the raw JSONL, e.g. "21").
        position: 0-indexed rank of this item among all items sharing
            (category, format, condition, template), in original evaluation
            order. Stable across formats for the same underlying item,
            because format is a per-row loop nested inside the item loop.
        role: Majority role across the 3 permutations: 'stereotyped',
            'anti_stereotyped', or 'unknown'. None if unstable (3-way tie).
        unstable: True if no role reached a strict majority.
        correct_majority: True if the majority of permutations answered
            correctly (matches `correct_position`).
        n_obs: Number of raw observations consolidated (always 3).
        question_polarity: BBQ's `question_polarity` ('neg'/'nonneg') for this
            item, or None if not yet resolved. NOT present in the raw JSONL
            (design.md D2 Addendum 6): populated later, by position, from the
            real BBQ dataset via `application.contrast_builder.
            attach_polarity_and_text` -- required because the corrected
            pairing key is `(categoría, question_index, question_polarity)`,
            not `(categoría, question_index)` alone (343/343 templates
            contain both polarities' opposite-worded questions).
        question_text: The real question string for this item, or None if
            not yet resolved. Used as a tie-breaker within the rare (1.0 %)
            (category, template, polarity) groups that still contain more
            than one distinct wording (spec: "Plantilla y polaridad no fijan
            el enunciado").
    """

    category: str
    format: str
    condition: str
    template: str
    position: int
    role: str | None
    unstable: bool
    correct_majority: bool
    n_obs: int
    question_polarity: str | None = None
    question_text: str | None = None


def _consolidate_one(obs: list[dict[str, Any]]) -> dict[str, Any]:
    """Consolidate one item's 3 permutation observations by majority vote."""
    counts = collections.Counter(o["chosen_role"] for o in obs)
    top, n_top = counts.most_common(1)[0]
    unstable = sum(1 for _, c in counts.items() if c == n_top) > 1
    correct = sum(o["chosen_position"] == o["correct_position"] for o in obs)
    return {
        "category": obs[0]["category"],
        "format": obs[0]["format"],
        "condition": obs[0]["context_condition"],
        "role": None if unstable else top,
        "unstable": unstable,
        "correct_majority": correct * 2 > len(obs),
        "n_obs": len(obs),
    }


def reconstruct_items(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], ConsolidatedItem]:
    """Reconstruct per-item identity and consolidated role from raw observation rows.

    Args:
        rows: Raw JSONL rows (dicts with at least `category`, `format`,
            `context_condition`, `item_id`, `permutation_order`,
            `chosen_role`, `chosen_position`, `correct_position`).

    Returns:
        Dict keyed by (category, format, condition, template, position) ->
        ConsolidatedItem.

    Raises:
        PermutationCycleError: if any bucket's size is not a multiple of 3,
            or a consecutive triplet does not follow `ORDER_CYCLE` exactly.
            This is the abort-on-broken-pattern safeguard design.md D1
            requires instead of silently misattributing observations.
    """
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        key = (r["category"], r["format"], r["context_condition"], r["item_id"])
        buckets[key].append(r)

    for key, obs in buckets.items():
        if len(obs) % 3 != 0:
            raise PermutationCycleError(
                f"ABORTA: {key} tiene {len(obs)} obs, no múltiplo de 3"
            )
        for start in range(0, len(obs), 3):
            got = [o["permutation_order"] for o in obs[start : start + 3]]
            if got != ORDER_CYCLE:
                raise PermutationCycleError(
                    f"ABORTA: patrón de permutaciones roto en {key} pos {start}: {got}"
                )

    items: dict[tuple[Any, ...], ConsolidatedItem] = {}
    for key, obs in buckets.items():
        category, format_name, condition, template = key
        for start in range(0, len(obs), 3):
            trio = obs[start : start + 3]
            consolidated = _consolidate_one(trio)
            position = start // 3
            items[key + (position,)] = ConsolidatedItem(
                category=category,
                format=format_name,
                condition=condition,
                template=template,
                position=position,
                role=consolidated["role"],
                unstable=consolidated["unstable"],
                correct_majority=consolidated["correct_majority"],
                n_obs=consolidated["n_obs"],
            )
    return items
