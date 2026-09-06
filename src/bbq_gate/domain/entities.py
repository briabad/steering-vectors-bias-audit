"""Domain entities for BBQ existence gate.

Entities are pure value objects with no external I/O. All logic here is
testable without GPU or network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class AnswerRole(str, Enum):
    """Role of an answer option in the BBQ item."""

    STEREOTYPED = "stereotyped"
    ANTI_STEREOTYPED = "anti_stereotyped"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BBQOption:
    """A single answer option with its position, text, and resolved role."""

    position: Literal[0, 1, 2]
    text: str
    role: AnswerRole


@dataclass(frozen=True)
class BBQItem:
    """A single item from the BBQ dataset.

    Attributes:
        item_id: BBQ's `example_id`. Unique within its `category`, but NOT
            necessarily unique across categories, nor across `context_condition`
            values within the same category (an `ambig` and its paired
            `disambig` row can share `example_id`). Callers that need a
            globally unique key must use the tuple
            (category, context_condition, item_id).
        category: BBQ category (e.g. 'Age', 'Religion')
        context_condition: 'ambig' or 'disambig'
        context: The context string
        question: The question string
        options: Three answer options with resolved roles
        stereotyped_groups: List of group names that are stereotyped in this item
        label_idx: Index of the correct answer (0, 1, or 2)
    """

    item_id: int
    category: str
    context_condition: Literal["ambig", "disambig"]
    context: str
    question: str
    options: list[BBQOption]
    stereotyped_groups: list[str]
    label_idx: int

    def __post_init__(self) -> None:
        """Validate invariants."""
        if len(self.options) != 3:
            raise ValueError(f"Expected 3 options, got {len(self.options)}")

        positions = {opt.position for opt in self.options}
        if positions != {0, 1, 2}:
            raise ValueError(f"Options must cover positions 0,1,2; got {positions}")

        unknown_count = sum(1 for opt in self.options if opt.role == AnswerRole.UNKNOWN)
        if unknown_count != 1:
            raise ValueError(
                f"Item {self.item_id}: expected exactly 1 unknown option, got {unknown_count}"
            )

        non_unknown_roles = [opt.role for opt in self.options if opt.role != AnswerRole.UNKNOWN]
        if len(non_unknown_roles) != 2:
            raise ValueError(
                f"Item {self.item_id}: expected 2 non-unknown options, got {len(non_unknown_roles)}"
            )

        if not (0 <= self.label_idx <= 2):
            raise ValueError(f"label_idx must be 0, 1, or 2; got {self.label_idx}")

    def option_by_position(self, position: Literal[0, 1, 2]) -> BBQOption:
        """Get option by position."""
        for opt in self.options:
            if opt.position == position:
                return opt
        raise ValueError(f"No option at position {position}")

    def unknown_option(self) -> BBQOption:
        """Get the unknown option."""
        for opt in self.options:
            if opt.role == AnswerRole.UNKNOWN:
                return opt
        raise ValueError(f"No unknown option in item {self.item_id}")

    def non_unknown_options(self) -> list[BBQOption]:
        """Get the two non-unknown options, in order by position."""
        return sorted(
            [opt for opt in self.options if opt.role != AnswerRole.UNKNOWN],
            key=lambda opt: opt.position,
        )


@dataclass(frozen=True)
class ScoringResult:
    """Result of scoring a single BBQ item.

    Attributes:
        item_id: The item's unique identifier
        chosen_position: Index (0, 1, 2) of the chosen answer, or None if model abstained
        correct_position: Index of the correct answer
        is_correct: True if the choice matches the gold label
        is_unknown_choice: True if the model chose the unknown option
    """

    item_id: int
    chosen_position: int | None
    correct_position: int
    is_correct: bool
    is_unknown_choice: bool

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.chosen_position is not None and not (0 <= self.chosen_position <= 2):
            raise ValueError(f"chosen_position must be 0, 1, 2, or None; got {self.chosen_position}")
        if not (0 <= self.correct_position <= 2):
            raise ValueError(f"correct_position must be 0, 1, 2; got {self.correct_position}")
