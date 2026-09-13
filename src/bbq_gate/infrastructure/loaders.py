"""Loaders for external data sources: BBQ from Hugging Face Datasets.

This module implements the infrastructure for loading BBQ data and converting
it to domain entities.
"""
from __future__ import annotations

from typing import Generator

from datasets import load_dataset

from bbq_gate.domain.entities import AnswerRole, BBQItem, BBQOption
from bbq_gate.domain.resolution import (
    resolve_stereotyped_option,
    resolve_unknown_option,
    tag_matches_group,
)

BBQ_CATEGORIES: tuple[str, ...] = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
)


def _iter_bbq_category_rows(
    category: str,
    alias: dict[str, set[str]],
    stats: dict[str, int],
    ds: object,
) -> Generator[tuple[BBQItem, str, str], None, None]:
    """Shared row-by-row filtering logic, yielding (item, question_index,
    question_polarity) for every row accepted into the analysis.
    `question_index` is BBQ's question TEMPLATE number; `question_polarity`
    is BBQ's own 'neg'/'nonneg' tag. Both are used by
    `construccion-vector-operacion` to pair P+/P- by category, template AND
    polarity (design.md D2, corrected 2026-09-06: pairing by template alone
    lets a lexical classifier learn the polarity's opposite wording as a
    label proxy -- see `bbq_gate.domain.contrast._match_key`), distinct from
    `example_id` (the per-row unique identifier, stored as `BBQItem.item_id`).

    Mutates `stats` in place as rows are consumed (this is a generator: the
    counts only reflect rows actually iterated by the caller).
    """
    for row in ds:
        stats["total"] += 1
        # BBQ's `question_index` is the template/question number (19-50 distinct
        # values per category, shared by up to 720 rows). `example_id` is the
        # per-row unique identifier within the category and is what item_id
        # must carry (bug found 2026-09-06: question_index collisions silently
        # merged unrelated observations during aggregation).
        item_id = row["example_id"]
        context_condition = row["context_condition"]
        context = row["context"]
        question = row["question"]
        stereotyped_groups = row["additional_metadata"]["stereotyped_groups"]
        label = row["label"]

        # Build answer_info dict
        answer_info = {f"ans{i}": row["answer_info"][f"ans{i}"] for i in (0, 1, 2)}

        # Resolve unknown option
        try:
            unknown_pos = resolve_unknown_option(answer_info)
        except ValueError:
            stats["missing_unknown"] += 1
            continue

        if unknown_pos is None:
            stats["missing_unknown"] += 1
            continue

        # Resolve stereotyped option
        try:
            stereotyped_pos, anti_pos = resolve_stereotyped_option(
                answer_info, stereotyped_groups, alias
            )
        except ValueError:
            continue

        # If stereotyped resolution failed, categorize the failure
        if stereotyped_pos is None:
            # This is not an error; the item is just not usable for this analysis
            # (both options match, or neither matches). Determine which.
            others = [i for i in (0, 1, 2) if i != unknown_pos]
            hits = sum(
                1
                for pos in others
                if any(tag_matches_group(t, g, alias) for t in answer_info[f"ans{pos}"] for g in stereotyped_groups)
            )
            if hits == 2:
                stats["intersectional"] += 1
            else:  # hits == 0
                stats["no_match"] += 1
            continue

        # Build options with roles
        options = []
        for pos in (0, 1, 2):
            text = row[f"ans{pos}"]
            if pos == unknown_pos:
                role = AnswerRole.UNKNOWN
            elif pos == stereotyped_pos:
                role = AnswerRole.STEREOTYPED
            else:
                role = AnswerRole.ANTI_STEREOTYPED

            options.append(BBQOption(position=pos, text=text, role=role))

        # Create and yield the item
        try:
            item = BBQItem(
                item_id=item_id,
                category=category,
                context_condition=context_condition,
                context=context,
                question=question,
                options=options,
                stereotyped_groups=stereotyped_groups,
                label_idx=label,
            )
        except ValueError:
            continue

        stats["used"] += 1
        yield item, row["question_index"], row["question_polarity"]


def load_bbq_category(
    category: str,
    alias: dict[str, set[str]] | None = None,
    verbose: bool = False,
) -> tuple[list[BBQItem], dict[str, int]]:
    """Load all items from a BBQ category and return as domain entities.

    Args:
        category: The BBQ category name (e.g., 'Age', 'Religion')
        alias: Optional alias dict for code-to-name mapping (e.g., {'f': {'woman', 'girl'}})
        verbose: If True, print coverage statistics

    Returns:
        (items, stats): List of BBQItem objects and dict with coverage statistics

    Raises:
        ValueError: If an item cannot be resolved (missing unknown option, etc.)
    """
    if alias is None:
        alias = {}

    ds = load_dataset("oskarvanderwal/bbq", category)["test"]
    stats = {"total": 0, "used": 0, "missing_unknown": 0, "intersectional": 0, "no_match": 0}

    items = [
        item for item, _question_index, _question_polarity in _iter_bbq_category_rows(category, alias, stats, ds)
    ]

    if verbose:
        coverage = 100 * stats["used"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{category:20} {coverage:5.1f}%  (used {stats['used']}/{stats['total']})")

    return items, stats


def load_bbq_category_with_template_index(
    category: str,
    alias: dict[str, set[str]] | None = None,
    verbose: bool = False,
) -> tuple[list[tuple[BBQItem, str, str]], dict[str, int]]:
    """Same filtering as `load_bbq_category`, but also returns BBQ's
    `question_index` (question template number) and `question_polarity`
    alongside each item, in the SAME row order as `load_bbq_category` /
    `load_bbq_all_categories` would produce.

    Used by `construccion-vector-operacion`'s contrast builder to match the
    positionally-reconstructed identity recovered from the (defective)
    `existence_gate_raw.jsonl` back to real item text AND polarity -- see
    `bbq_gate.application.contrast_builder` and design.md D1/D2.
    """
    if alias is None:
        alias = {}

    ds = load_dataset("oskarvanderwal/bbq", category)["test"]
    stats = {"total": 0, "used": 0, "missing_unknown": 0, "intersectional": 0, "no_match": 0}

    items_with_index = list(_iter_bbq_category_rows(category, alias, stats, ds))

    if verbose:
        coverage = 100 * stats["used"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{category:20} {coverage:5.1f}%  (used {stats['used']}/{stats['total']})")

    return items_with_index, stats


def load_bbq_all_categories(
    alias: dict[str, set[str]] | None = None,
    verbose: bool = False,
) -> tuple[list[BBQItem], dict[str, dict[str, int]]]:
    """Load all items from all 11 BBQ categories.

    Args:
        alias: Optional alias dict
        verbose: If True, print coverage by category

    Returns:
        (items, stats_by_category): All BBQItem objects and stats per category
    """
    if alias is None:
        alias = {}

    categories = list(BBQ_CATEGORIES)

    all_items = []
    all_stats = {}

    if verbose:
        print(f"{'Category':<20} {'Coverage':>9} {'Count':>15}")
        print("-" * 50)

    for cat in categories:
        items, stats = load_bbq_category(cat, alias, verbose=verbose)
        all_items.extend(items)
        all_stats[cat] = stats

    return all_items, all_stats
