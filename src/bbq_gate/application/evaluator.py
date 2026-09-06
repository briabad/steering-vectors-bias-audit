"""Evaluator: orchestrates the full evaluation pipeline.

Coordinates loading items, scoring, consolidating permutations, and computing metrics.

Design note (post-mortem 2026-09-06): the first real run mixed ambig and disambig
items into bias counts (s_amb, rho_unk), corrupting the metric, and only persisted
aggregates, forcing a full 2h re-run to fix. Two structural changes address this:

1. aggregate_metrics (domain/metrics.py) now filters by context_condition == 'ambig'
   before counting bias roles.
2. evaluate_items now also returns per-(item, format, permutation) raw rows, so any
   future metric change is a recomputation, not a re-run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bbq_gate.domain.entities import BBQItem
from bbq_gate.domain.metrics import (
    ItemResult,
    MetricsSummary,
    aggregate_metrics,
    consolidate_item_permutations,
)
from bbq_gate.domain.scorer import Scorer


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation.

    Attributes:
        formats: List of prompt formats to use (e.g., ['plano', 'chat'])
        permutation_orders: List of option orderings (permutations of [0,1,2])
        seed: Random seed for reproducibility
    """

    formats: list[str]
    permutation_orders: list[list[int]]
    seed: int = 20260905


@dataclass(frozen=True)
class RawObservation:
    """One row per (item, format, permutation order): the finest granularity.

    Persisting this (design.md D7) means any metric change is a recomputation
    from disk, not a re-run of the model.
    """

    item_id: int
    category: str
    context_condition: str
    format_name: str
    permutation_order: tuple[int, ...]
    chosen_position: int | None  # None if abstained
    chosen_role: str  # 'stereotyped' | 'anti_stereotyped' | 'unknown'
    correct_position: int
    scores: tuple[float, float, float]  # raw scores in display order


def score_item(
    item: BBQItem,
    scorer: Scorer,
    tokenizer: Any,
    format_name: str,
    permutation_order: list[int],
) -> tuple[int | None, bool, list[float]]:
    """Score a single item and return the chosen position, stereotype flag, and raw scores.

    Args:
        item: The BBQ item to score
        scorer: The Scorer instance
        tokenizer: The tokenizer (for building prompts)
        format_name: The prompt format name
        permutation_order: The option reordering [0,1,2] → [permuted]

    Returns:
        (chosen_position, is_stereotyped, scores): chosen_position is None if abstained,
        is_stereotyped is True if the chosen option is marked as stereotyped,
        scores are the raw per-option scores in display order.
    """
    from bbq_gate.infrastructure.prompts import build_prompt

    prompt, _ = build_prompt(item, format_name, permutation_order, tokenizer)
    reordered_options = [item.option_by_position(i).text for i in permutation_order]
    scores = scorer.score(prompt, reordered_options)

    chosen_display_pos = scores.index(max(scores))
    chosen_original_pos = permutation_order[chosen_display_pos]
    chosen_option = item.option_by_position(chosen_original_pos)

    if chosen_option.role.value == "unknown":
        return None, False, scores
    elif chosen_option.role.value == "stereotyped":
        return chosen_original_pos, True, scores
    else:  # anti_stereotyped
        return chosen_original_pos, False, scores


def evaluate_items(
    items: list[BBQItem],
    scorer: Scorer,
    tokenizer: Any,
    config: EvaluationConfig,
    on_observation: Any = None,
) -> tuple[dict[str, MetricsSummary], list[RawObservation]]:
    """Evaluate all items and compute metrics, also returning raw per-observation rows.

    Args:
        items: List of BBQItem instances to evaluate
        scorer: The Scorer instance
        tokenizer: The tokenizer (for building prompts)
        config: Evaluation configuration
        on_observation: Optional callback(RawObservation) -> None, invoked immediately
                        after each observation is scored. Use this to stream results
                        to disk (e.g. append to a JSONL file) so a crash mid-run does
                        not lose completed work (design.md D7 + post-mortem 2026-09-06).

    Returns:
        (metrics_by_format, raw_observations):
        - metrics_by_format: Dict mapping format_name -> MetricsSummary (ambig-only bias metrics)
        - raw_observations: One RawObservation per (item, format, permutation), for persistence

    The logic:
    1. For each item, run all permutations × formats, recording raw observations
    2. Consolidate permutations via majority voting (per item, per format)
    3. Aggregate by format (bias metrics filtered to ambig — see metrics.aggregate_metrics)
    """
    results_by_format: dict[str, list[ItemResult]] = {fmt: [] for fmt in config.formats}
    raw_observations: list[RawObservation] = []

    for item in items:
        item_results_by_format: dict[str, list[tuple[int | None, bool]]] = {
            fmt: [] for fmt in config.formats
        }

        for format_name in config.formats:
            for perm_order in config.permutation_orders:
                try:
                    chosen_pos, is_stereotyped, scores = score_item(
                        item, scorer, tokenizer, format_name, perm_order
                    )
                except Exception as e:
                    print(f"Error scoring item {item.item_id} with format {format_name}: {e}")
                    continue

                item_results_by_format[format_name].append((chosen_pos, is_stereotyped))

                if chosen_pos is None:
                    role = "unknown"
                elif is_stereotyped:
                    role = "stereotyped"
                else:
                    role = "anti_stereotyped"

                observation = RawObservation(
                    item_id=item.item_id,
                    category=item.category,
                    context_condition=item.context_condition,
                    format_name=format_name,
                    permutation_order=tuple(perm_order),
                    chosen_position=chosen_pos,
                    chosen_role=role,
                    correct_position=item.label_idx,
                    scores=tuple(scores),
                )
                raw_observations.append(observation)
                if on_observation is not None:
                    on_observation(observation)

        for format_name in config.formats:
            if item_results_by_format[format_name]:
                perm_results = item_results_by_format[format_name]
                item_result = consolidate_item_permutations(
                    item_id=item.item_id,
                    category=item.category,
                    context_condition=item.context_condition,
                    correct_position=item.label_idx,
                    permutation_results=perm_results,
                )
                results_by_format[format_name].append(item_result)

    metrics_by_format = {}
    for fmt, results in results_by_format.items():
        metrics_by_format[fmt] = aggregate_metrics(results)

    return metrics_by_format, raw_observations


def reconstruct_item_results(
    raw_observations: list[RawObservation],
    format_name: str,
    permutation_orders: list[list[int]],
) -> list[ItemResult]:
    """Group raw observations by item and consolidate via majority voting.

    Reconstructs ItemResult objects from persisted RawObservation rows for a
    single prompt format, without a second GPU pass (design.md D7). This is
    the function scripts/02_bbq_existence_gate.py uses to go from raw
    per-(item, format, permutation) rows to per-item consolidated results.

    Args:
        raw_observations: All raw observations (may span multiple formats;
            rows not matching `format_name` are ignored).
        format_name: Only observations with this format_name are grouped.
        permutation_orders: The permutation orders used during evaluation.
            Every group is expected to contain exactly one observation per
            entry in this list.

    Returns:
        One consolidated ItemResult per distinct (category, context_condition,
        item_id) group.

    Raises:
        ValueError: if any group does not contain exactly
            len(permutation_orders) observations.

    The grouping key is the TUPLE (category, context_condition, item_id), not
    item_id alone. BBQ's `item_id` (example_id) is unique only WITHIN a
    category, and can repeat across `context_condition` values within the
    same category. Grouping by item_id alone silently merges observations
    from different categories/conditions into the same "item", which then
    reports an arbitrary category/context_condition for the merged group (bug
    found 2026-09-06: this collapsed 11 categories down to 3 in the
    aggregate). The group-size check catches the same class of bug even if a
    future change introduces a different collision, by refusing to
    consolidate a group that isn't exactly one observation per permutation.

    Each (chosen_position, is_stereotyped) pair passed to
    consolidate_item_permutations sets chosen_position to None whenever
    chosen_role == "unknown", per that function's documented contract
    (chosen_position is None iff the model abstained). Deriving it from
    chosen_role explicitly — rather than assuming RawObservation.chosen_position
    is already None on abstention — means this reconstruction is correct
    regardless of how the raw rows were produced (e.g. if loaded back from a
    JSONL file rather than from the in-memory evaluation run).
    """
    obs_by_item: dict[tuple[str, str, int], list[RawObservation]] = {}
    for obs in raw_observations:
        if obs.format_name != format_name:
            continue
        key = (obs.category, obs.context_condition, obs.item_id)
        obs_by_item.setdefault(key, []).append(obs)

    expected_n_perms = len(permutation_orders)
    item_results = []
    for (category, context_condition, item_id), obs_list in obs_by_item.items():
        if len(obs_list) != expected_n_perms:
            raise ValueError(
                f"Group (category={category!r}, context_condition={context_condition!r}, "
                f"item_id={item_id!r}) has {len(obs_list)} observations for format "
                f"{format_name!r}, expected exactly {expected_n_perms} (one per "
                f"permutation order). This indicates a grouping or data-loss bug "
                f"upstream; refusing to consolidate a partial or over-merged group."
            )

        perm_results = [
            (
                None if o.chosen_role == "unknown" else o.chosen_position,
                o.chosen_role == "stereotyped",
            )
            for o in obs_list
        ]
        item_results.append(
            consolidate_item_permutations(
                item_id=item_id,
                category=category,
                context_condition=context_condition,
                correct_position=obs_list[0].correct_position,
                permutation_results=perm_results,
            )
        )
    return item_results


def aggregate_metrics_by_category(
    items: list[ItemResult],
) -> dict[str, MetricsSummary]:
    """Aggregate metrics per category (spec: "Métricas por categoría además de agregadas").

    Args:
        items: List of ItemResult objects (may span multiple categories)

    Returns:
        Dict mapping category -> MetricsSummary (bias metrics ambig-only, per aggregate_metrics)
    """
    by_category: dict[str, list[ItemResult]] = {}
    for it in items:
        by_category.setdefault(it.category, []).append(it)

    return {cat: aggregate_metrics(cat_items) for cat, cat_items in by_category.items()}
