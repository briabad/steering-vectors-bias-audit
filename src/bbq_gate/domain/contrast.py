"""Paired contrast construction over ambig items (design.md D1-D3, Addendum 5).

Builds the operation-level contrast (P+ = chose a group, P- = abstained) from
`ConsolidatedItem`s already reconstructed positionally from the raw JSONL
(`domain.raw_reconstruction`), matches P+/P- 1:1 by (category, template) per
spec "Contraste emparejado 1:1 por categorÃ­a y plantilla", tracks the residue,
and isolates the H3 transfer categories as a distinct type (design.md D6).
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

from bbq_gate.domain.raw_reconstruction import ConsolidatedItem

POSITIVE_ROLES = frozenset({"stereotyped", "anti_stereotyped"})
NEGATIVE_ROLE = "unknown"

# hypothesis.md Addendum 5: categories reserved for H3, untouched by this change.
RESERVED_CATEGORIES = frozenset({"Race_x_gender", "Gender_identity", "Sexual_orientation"})


def derive_operation_sets(
    items: list[ConsolidatedItem], format_name: str
) -> tuple[list[ConsolidatedItem], list[ConsolidatedItem]]:
    """Split ambig, format-matching items into P+ (chose a group) and P- (abstained).

    Task 1.2 / spec "Contraste emparejado 1:1...": operates on the level-1
    operation contrast (hypothesis.md Addendum 4), not the level-2 direction
    contrast: P+ includes BOTH stereotyped and anti_stereotyped choices.

    Args:
        items: Consolidated items (may span multiple formats/conditions;
            non-matching rows are filtered out here).
        format_name: Only items with this `format` are considered.

    Returns:
        (positives, negatives): unstable items are excluded from both (an
        unstable item has `role is None` and cannot be classified).
    """
    ambig = [
        it
        for it in items
        if it.format == format_name and it.condition == "ambig" and not it.unstable
    ]
    positives = [it for it in ambig if it.role in POSITIVE_ROLES]
    negatives = [it for it in ambig if it.role == NEGATIVE_ROLE]
    return positives, negatives


@dataclass(frozen=True)
class TemplateStats:
    """Per-(category, template) bookkeeping for the pairing step.

    Attributes:
        n_positive: Number of P+ items in this template.
        n_negative_available: Number of P- items available in this template
            (the pool size BEFORE consumption by pairing).
        n_paired: Number of pairs actually formed.
        n_residue: Number of P+ items left unpaired.
    """

    n_positive: int
    n_negative_available: int
    n_paired: int
    n_residue: int


def match_pairs_by_template(
    positives: list[ConsolidatedItem],
    negatives: list[ConsolidatedItem],
) -> tuple[
    list[tuple[ConsolidatedItem, ConsolidatedItem]],
    list[ConsolidatedItem],
    dict[tuple[str, str], TemplateStats],
]:
    """1:1 pairing of P+ with P- sharing the same (category, template).

    Spec "Contraste emparejado 1:1 por categorÃ­a y plantilla": every P+ item is
    matched with at most one P- item of the same category and template;
    unmatched P- items are discarded; unmatched P+ items become residue.

    Pairing within a template is deterministic (by `position`, ascending), not
    random: reproducibility (spec "Reproducibilidad") does not require a seed
    here since there is no randomness to seed.

    Args:
        positives: P+ ConsolidatedItems (any templates/categories).
        negatives: P- ConsolidatedItems (any templates/categories).

    Returns:
        (pairs, residue, stats_by_template):
        - pairs: list of (positive, negative) tuples, matched 1:1.
        - residue: leftover, unmatched P+ items.
        - stats_by_template: per (category, template) bookkeeping, including
          templates with zero available negatives (spec: "Plantilla sin
          ningÃºn Ã­tem negativo").
    """
    neg_by_template: dict[tuple[str, str], list[ConsolidatedItem]] = collections.defaultdict(list)
    for neg in negatives:
        neg_by_template[(neg.category, neg.template)].append(neg)
    for pool in neg_by_template.values():
        pool.sort(key=lambda it: it.position)

    pos_by_template: dict[tuple[str, str], list[ConsolidatedItem]] = collections.defaultdict(list)
    for pos in positives:
        pos_by_template[(pos.category, pos.template)].append(pos)

    pairs: list[tuple[ConsolidatedItem, ConsolidatedItem]] = []
    residue: list[ConsolidatedItem] = []
    stats: dict[tuple[str, str], TemplateStats] = {}

    for key, pos_items in pos_by_template.items():
        pos_items_sorted = sorted(pos_items, key=lambda it: it.position)
        pool = neg_by_template.get(key, [])
        n_negative_available = len(pool)
        n_paired = min(len(pos_items_sorted), n_negative_available)

        for i in range(n_paired):
            pairs.append((pos_items_sorted[i], pool[i]))
        for i in range(n_paired, len(pos_items_sorted)):
            residue.append(pos_items_sorted[i])

        stats[key] = TemplateStats(
            n_positive=len(pos_items_sorted),
            n_negative_available=n_negative_available,
            n_paired=n_paired,
            n_residue=len(pos_items_sorted) - n_paired,
        )

    return pairs, residue, stats


def templates_without_negatives(
    stats_by_template: dict[tuple[str, str], TemplateStats],
) -> list[tuple[str, str]]:
    """Templates whose P+ items had zero P- available (spec: "Plantilla sin
    ningÃºn Ã­tem negativo"). Task 1.4: `Religion 21` and `Religion 24` must
    appear here for the `chat` format on the real corpus.
    """
    return sorted(
        key for key, s in stats_by_template.items() if s.n_positive > 0 and s.n_negative_available == 0
    )


def category_composition(items: list[ConsolidatedItem]) -> dict[str, int]:
    """Count of items per category (spec: "ComposiciÃ³n resultante" -- the
    two sides of a pairing must show an IDENTICAL distribution)."""
    counts: dict[str, int] = collections.defaultdict(int)
    for it in items:
        counts[it.category] += 1
    return dict(counts)


@dataclass(frozen=True)
class ContrastDataset:
    """The adjustable contrast: paired items usable for direction construction,
    layer selection, and any fitting decision.

    design.md D6: reserved categories are excluded BY CONSTRUCTION. Any
    attempt to build one with a reserved-category item raises, so isolation
    does not depend on every caller remembering to filter (task 1.5).
    """

    pairs: tuple[tuple[ConsolidatedItem, ConsolidatedItem], ...]
    residue: tuple[ConsolidatedItem, ...]

    def __post_init__(self) -> None:
        for pos, neg in self.pairs:
            if pos.category in RESERVED_CATEGORIES or neg.category in RESERVED_CATEGORIES:
                raise ValueError(
                    f"ContrastDataset cannot contain reserved category items "
                    f"(got pair with categories {pos.category!r}/{neg.category!r}); "
                    f"reserved categories {sorted(RESERVED_CATEGORIES)} are for H3 only "
                    f"(design.md D6)"
                )
        for item in self.residue:
            if item.category in RESERVED_CATEGORIES:
                raise ValueError(
                    f"ContrastDataset residue cannot contain reserved category "
                    f"{item.category!r} (design.md D6)"
                )


@dataclass(frozen=True)
class TransferDataset:
    """The H3 transfer holdout: reserved-category pairs, isolated by TYPE.

    No fitting function in this change accepts this type (design.md D6):
    functions that build directions or select layers take `ContrastDataset`
    only, so passing a `TransferDataset` is a type error, not a silent bug.
    """

    pairs: tuple[tuple[ConsolidatedItem, ConsolidatedItem], ...]

    def __post_init__(self) -> None:
        for pos, neg in self.pairs:
            if pos.category not in RESERVED_CATEGORIES or neg.category not in RESERVED_CATEGORIES:
                raise ValueError(
                    f"TransferDataset must contain ONLY reserved category items "
                    f"(got {pos.category!r}/{neg.category!r})"
                )


def split_reserved(
    pairs: list[tuple[ConsolidatedItem, ConsolidatedItem]],
    residue: list[ConsolidatedItem],
) -> tuple[ContrastDataset, TransferDataset]:
    """Partition pairs/residue into the adjustable contrast and the H3 transfer
    reserve (design.md D6, Addendum 5's declared reserved-category list).

    A pair is routed to the transfer set only if BOTH sides are reserved
    categories, which always holds here because pairing is per-category.
    """
    contrast_pairs = tuple(
        (pos, neg)
        for pos, neg in pairs
        if pos.category not in RESERVED_CATEGORIES and neg.category not in RESERVED_CATEGORIES
    )
    transfer_pairs = tuple(
        (pos, neg)
        for pos, neg in pairs
        if pos.category in RESERVED_CATEGORIES and neg.category in RESERVED_CATEGORIES
    )
    contrast_residue = tuple(it for it in residue if it.category not in RESERVED_CATEGORIES)

    return (
        ContrastDataset(pairs=contrast_pairs, residue=contrast_residue),
        TransferDataset(pairs=transfer_pairs),
    )


def match_residue_by_category(
    residue: list[ConsolidatedItem],
    all_negatives: list[ConsolidatedItem],
    used_negatives: list[ConsolidatedItem],
) -> list[tuple[ConsolidatedItem, ConsolidatedItem]]:
    """Pair residue P+ items (no template match available, design.md D3) with
    UNUSED P- items sharing only CATEGORY -- a laxer pairing than
    `match_pairs_by_template`, used to build $v_B$ on a set DISJOINT from the
    $v_A$ pairs (task 5.1: "verificar... que los conjuntos son disjuntos").

    Args:
        residue: Leftover P+ items from `match_pairs_by_template`.
        all_negatives: The full P- pool for this format (same input passed
            to `match_pairs_by_template`).
        used_negatives: The P- items already consumed by the $v_A$ pairing;
            excluded here so $v_B$ never reuses an item from $v_A$.

    Returns:
        List of (positive, negative) pairs, deterministic (by position),
        possibly shorter than `residue` if a category runs out of unused
        negatives.
    """
    used = set(used_negatives)
    available = [n for n in all_negatives if n not in used]

    by_category: dict[str, list[ConsolidatedItem]] = collections.defaultdict(list)
    for neg in available:
        by_category[neg.category].append(neg)
    for pool in by_category.values():
        pool.sort(key=lambda it: it.position)

    consumed_idx: dict[str, int] = collections.defaultdict(int)
    pairs: list[tuple[ConsolidatedItem, ConsolidatedItem]] = []
    for pos in sorted(residue, key=lambda it: (it.category, it.position)):
        pool = by_category.get(pos.category, [])
        idx = consumed_idx[pos.category]
        if idx < len(pool):
            pairs.append((pos, pool[idx]))
            consumed_idx[pos.category] += 1
    return pairs


def assert_no_reserved_categories(items: list[ConsolidatedItem]) -> None:
    """Runtime guard used by fitting functions (task 1.5): raise if any item
    belongs to a reserved category. A second line of defense in addition to
    the type-level isolation of `ContrastDataset`/`TransferDataset`.
    """
    offenders = sorted({it.category for it in items if it.category in RESERVED_CATEGORIES})
    if offenders:
        raise ValueError(
            f"Reserved categories {offenders} must not participate in fitting "
            f"(construction, layer selection, or any adjustment) -- design.md D6"
        )
