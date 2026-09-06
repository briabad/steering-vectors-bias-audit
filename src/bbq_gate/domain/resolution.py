"""Option resolution: mapping raw data to domain entities.

This module implements the logic for:
1. Resolving the unknown option from answer_info annotations
2. Matching the stereotyped and anti-stereotyped options using a cascading matcher
3. Validating that exactly one non-unknown option matches the stereotype

All logic is domain-specific to BBQ and is testable without I/O.
"""
from __future__ import annotations

import re
from typing import Any


def norm(s: str) -> str:
    """Normalize a string: lowercase, alphanumeric only."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def parts_of(tag: str) -> list[str]:
    """Split a tag by delimiters and normalize each part."""
    parts = re.split(r"[-_ ]+", str(tag))
    return [p for p in (norm(p) for p in parts) if p]


def resolve_unknown_option(answer_info: dict[str, list[str]]) -> int | None:
    """Resolve which position (0, 1, 2) holds the unknown option.

    Args:
        answer_info: Dict mapping 'ans0', 'ans1', 'ans2' to lists of group names.
                     The unknown option's list contains a tag normalized to 'unknown'.

    Returns:
        Position (0, 1, or 2) of the unknown option, or None if not found.

    Raises:
        ValueError: If the structure is malformed or no unique unknown found.
    """
    unk_positions = []
    for position_str in ("ans0", "ans1", "ans2"):
        if position_str not in answer_info:
            raise ValueError(f"Missing {position_str} in answer_info")
        tags = answer_info[position_str]
        if not isinstance(tags, list):
            raise ValueError(f"{position_str} must be a list of tags")
        if any(norm(t) == "unknown" for t in tags):
            unk_positions.append(int(position_str[-1]))

    if len(unk_positions) != 1:
        raise ValueError(
            f"Expected exactly 1 unknown option, found {len(unk_positions)} at positions {unk_positions}"
        )
    return unk_positions[0]


def tag_matches_group(tag: str, group: str, alias: dict[str, set[str]] | None = None) -> bool:
    """Check if a tag matches a group name using cascading rules.

    Rules (in order):
    1. Exact match (normalized)
    2. Negation guard: if tag starts with "non", only accept if it's a different concept
    3. Parts of compound tag: if group is a component of the tag
    4. Substring: if group appears anywhere in the tag

    Args:
        tag: The tag to match (from answer_info)
        group: The group name (from stereotyped_groups)
        alias: Optional dict mapping short codes to word sets (e.g. {'f': {'woman', 'girl'}})

    Returns:
        True if the tag matches the group by any rule.
    """
    if alias is None:
        alias = {}

    t, g = norm(tag), norm(group)
    if not t or not g:
        return False

    # Rule 1: Exact match
    if t == g:
        return True

    # Rule 2: Negation guard (before substring)
    # If tag starts with "non", only match if it's the negation of something else
    if t.startswith("non"):
        rest = t[3:]
        if rest:
            # "nonOld" matches "old"? No.
            # "nonOld" matches "young"? No.
            # Only if the rest is exactly the group or vice versa, it's a negation guard
            if rest == g or g in rest or rest in g:
                return False
    # If group itself is "nonbinary", don't auto-exclude
    # (the tag starting with "non" doesn't mean it can't match a group called "nonbinary")

    # Rule 3: Alias code mapping (e.g. 'f' → 'woman', 'girl', 'female')
    if g in alias and any(norm(w) in t for w in alias[g]):
        return True

    # Rule 4: Parts of compound tag
    tag_parts = parts_of(tag)
    if g in tag_parts:
        return True

    # Rule 5: Substring (with length guard to avoid false positives)
    if len(g) > 3 and g in t and not t.startswith("non"):
        return True

    return False


def resolve_stereotyped_option(
    answer_info: dict[str, list[str]],
    stereotyped_groups: list[str],
    alias: dict[str, set[str]] | None = None,
) -> tuple[int, int] | tuple[None, None]:
    """Resolve which positions are stereotyped and anti-stereotyped.

    Args:
        answer_info: Dict mapping 'ans0', 'ans1', 'ans2' to group tags.
        stereotyped_groups: List of groups that are stereotyped in this item.
        alias: Optional alias dict for code-to-name mapping.

    Returns:
        (stereotyped_pos, anti_stereotyped_pos) if exactly one of each can be resolved,
        (None, None) otherwise.

    The hard rule: exactly one non-unknown option must match a stereotyped group,
    and exactly one must not. Items where both match (intersectional items where
    both axes are covered) or neither matches are discarded.
    """
    unk_idx = resolve_unknown_option(answer_info)
    if unk_idx is None:
        return None, None

    # Non-unknown positions
    others = [i for i in (0, 1, 2) if i != unk_idx]
    if len(others) != 2:
        return None, None

    # Check which of the two non-unknown options match a stereotyped group
    hits = []
    for pos in others:
        tags = answer_info[f"ans{pos}"]
        if any(tag_matches_group(t, g, alias) for t in tags for g in stereotyped_groups):
            hits.append(pos)

    # Hard rule: exactly one match
    if len(hits) != 1:
        return None, None

    stereotyped_pos = hits[0]
    anti_stereotyped_pos = next(p for p in others if p != stereotyped_pos)
    return stereotyped_pos, anti_stereotyped_pos
