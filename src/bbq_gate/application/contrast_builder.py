"""Orchestrates the paired contrast end to end: reads the raw JSONL,
reconstructs item identity positionally, re-loads real BBQ text (with
template index), resolves the positional identity to real item text, and
assembles the pairs ready for activation extraction.

This module is I/O-heavy (reads a 288k-line file, loads 11 HF dataset
configs) and is exercised by `scripts/06_build_operation_vector.py`, not by
fast unit tests; the pure logic it calls (`domain.raw_reconstruction`,
`domain.contrast`) is unit-tested without I/O.
"""
from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path

from bbq_gate.domain.contrast import ContrastDataset, TransferDataset
from bbq_gate.domain.entities import BBQItem
from bbq_gate.domain.raw_reconstruction import ConsolidatedItem
from bbq_gate.infrastructure.loaders import BBQ_CATEGORIES, load_bbq_category_with_template_index

# Same alias table `scripts/02_bbq_existence_gate.py` used to produce the raw
# corpus; must match exactly for the positional reconstruction to resolve to
# the same set and order of real items.
DEFAULT_ALIAS = {"f": {"woman", "girl", "female"}, "m": {"man", "boy", "male"}}

RealItemIndex = dict[tuple[str, str, str], list[BBQItem]]


def load_raw_rows(raw_path: Path) -> list[dict]:
    """Read `existence_gate_raw.jsonl`, tolerating a truncated final line."""
    rows = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def build_real_item_index(
    categories: tuple[str, ...] = BBQ_CATEGORIES,
    alias: dict[str, set[str]] | None = None,
    verbose: bool = False,
) -> RealItemIndex:
    """Load real BBQ text for every category and index it by
    (category, context_condition, template), preserving within-group order.

    This is what lets `resolve_item` turn a positionally-reconstructed
    `ConsolidatedItem` (design.md D1) back into real prompt text: the K-th
    item in `index[(category, condition, template)]` is, by construction,
    the same underlying item as the K-th `ConsolidatedItem.position` sharing
    that (category, condition, template) -- see
    `bbq_gate.infrastructure.loaders.load_bbq_category_with_template_index`.
    """
    if alias is None:
        alias = DEFAULT_ALIAS

    index: RealItemIndex = collections.defaultdict(list)
    for category in categories:
        items_with_template, _stats = load_bbq_category_with_template_index(
            category, alias, verbose=verbose
        )
        for item, template in items_with_template:
            index[(item.category, item.context_condition, str(template))].append(item)
    return dict(index)


def resolve_item(consolidated: ConsolidatedItem, real_index: RealItemIndex) -> BBQItem:
    """Resolve one positionally-reconstructed item to its real BBQ text.

    Raises:
        ValueError: if the (category, condition, template) group does not
            exist, or does not have enough items for this position -- this
            means the real BBQ dataset or the resolution logic has changed
            since the raw corpus was produced, and the reconstruction can no
            longer be trusted (design.md D1's abort-on-mismatch principle
            extended to this second reconstruction step).
    """
    key = (consolidated.category, consolidated.condition, consolidated.template)
    candidates = real_index.get(key)
    if candidates is None:
        raise ValueError(f"no real items found for {key}; cannot resolve positional identity")
    if consolidated.position >= len(candidates):
        raise ValueError(
            f"position {consolidated.position} out of range for {key} "
            f"({len(candidates)} real candidates available) -- the BBQ dataset or "
            f"the resolution logic may have changed since the raw corpus was produced"
        )
    return candidates[consolidated.position]


@dataclass(frozen=True)
class PairedExample:
    """One P+/P- pair resolved to real text, ready for activation extraction."""

    positive: BBQItem
    negative: BBQItem
    category: str
    template: str


def resolve_pairs(
    pairs: tuple[tuple[ConsolidatedItem, ConsolidatedItem], ...],
    real_index: RealItemIndex,
) -> list[PairedExample]:
    """Resolve a sequence of (positive, negative) `ConsolidatedItem` pairs to
    real text pairs."""
    return [
        PairedExample(
            positive=resolve_item(pos, real_index),
            negative=resolve_item(neg, real_index),
            category=pos.category,
            template=pos.template,
        )
        for pos, neg in pairs
    ]


def resolve_contrast_dataset(dataset: ContrastDataset, real_index: RealItemIndex) -> list[PairedExample]:
    """Resolve every pair in a `ContrastDataset` to real text."""
    return resolve_pairs(dataset.pairs, real_index)


def resolve_transfer_dataset(dataset: TransferDataset, real_index: RealItemIndex) -> list[PairedExample]:
    """Resolve every pair in a `TransferDataset` to real text (H3 reserve,
    never used for fitting in this change -- only for persisting the
    material a future change would use)."""
    return resolve_pairs(dataset.pairs, real_index)
