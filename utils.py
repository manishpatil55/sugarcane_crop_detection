"""
utils.py
========
Shared utility functions used across the sugarcane detection pipeline.

This module is the **single source of truth** for time-tag handling. All
feature builders, sample generators, and trainers must use the helpers
defined here so the pipeline never silently sees zero-feature columns
because two modules disagreed on the tag format.

Two supported time-tag formats
------------------------------
1. ``YYYY_MM_DD``  e.g. ``2025_03_01``  — absolute calendar tag.
   Produced by ``data.polygon_extractor`` after fetching GEE composites.

2. ``tNN``  e.g. ``t00``, ``t01``, ..., ``t20``  — relative timestep tag.
   Produced by ``features.feature_table_builder`` for **year-independent**
   model features. ``t00`` = first timestep in the window (months_before
   the anchor date), ``tNN`` = last timestep (months_after).

A column name is ``<BAND>_<TAG>``  e.g. ``B8_2025_03_01`` (absolute) or
``B8_t05`` (relative).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_ABS_TAG_RE = re.compile(r"_(20\d{2})_(0[1-9]|1[0-2])_(0[1-9]|[12]\d|3[01])$")
_REL_TAG_RE = re.compile(r"_(t\d{2,3})$")


def detect_time_tags(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect time tags (absolute YYYY_MM_DD or relative tNN) from column names.

    Returns
    -------
    list[str], sorted. If columns mix both formats, both are returned.
    """
    tags = set()
    for col in df.columns:
        m_abs = _ABS_TAG_RE.search(col)
        if m_abs:
            tags.add(f"{m_abs.group(1)}_{m_abs.group(2)}_{m_abs.group(3)}")
            continue
        m_rel = _REL_TAG_RE.search(col)
        if m_rel:
            tags.add(m_rel.group(1))
    return sorted(tags)


def detect_absolute_time_tags(df: pd.DataFrame) -> List[str]:
    """Auto-detect ONLY absolute YYYY_MM_DD tags (chronological sort)."""
    tags = set()
    for col in df.columns:
        m = _ABS_TAG_RE.search(col)
        if m:
            tags.add(f"{m.group(1)}_{m.group(2)}_{m.group(3)}")
    return sorted(tags)


def detect_relative_time_tags(df: pd.DataFrame) -> List[str]:
    """Auto-detect ONLY relative tNN tags."""
    tags = set()
    for col in df.columns:
        m = _REL_TAG_RE.search(col)
        if m:
            tags.add(m.group(1))
    return sorted(tags)


def parse_tag_to_ymd(tag: str) -> Optional[Tuple[int, int, int]]:
    """Parse "YYYY_MM_DD" -> (year, month, day). Returns None for relative tags or bad input."""
    if tag.startswith("t"):
        return None
    parts = tag.split("_")
    if len(parts) != 3:
        return None
    try:
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if not (2000 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    return y, m, d


def parse_tag_to_step(tag: str) -> Optional[int]:
    """Parse relative tag 'tNN' -> int. Returns None for absolute tags or bad input."""
    if not isinstance(tag, str) or not tag.startswith("t"):
        return None
    try:
        return int(tag[1:])
    except ValueError:
        return None


def detect_bands(df: pd.DataFrame, time_tags: List[str]) -> List[str]:
    """
    Auto-detect band/index names by stripping each known time tag from the
    end of column names.

    Returns
    -------
    Sorted list of unique band/index prefixes.
    """
    bands = set()
    for col in df.columns:
        for tag in time_tags:
            suffix = f"_{tag}"
            if col.endswith(suffix):
                bands.add(col[: -len(suffix)])
                break
    return sorted(bands)


def split_columns_by_phase(time_tags: List[str], months: List[int]) -> List[str]:
    """
    Return time tags whose calendar month is in `months` (absolute tags only).

    For relative tags (which have no inherent calendar month), this returns []
    -- the caller should use ``split_columns_by_step_range`` instead.
    """
    out = []
    for tag in time_tags:
        ymd = parse_tag_to_ymd(tag)
        if ymd is not None and ymd[1] in months:
            out.append(tag)
    return out


def split_columns_by_step_range(time_tags: List[str],
                                step_range: Tuple[int, int]) -> List[str]:
    """
    Return relative tags whose step index falls in [start, end] inclusive.

    For absolute tags (no step), they are skipped.
    """
    s0, s1 = step_range
    out = []
    for tag in time_tags:
        step = parse_tag_to_step(tag)
        if step is not None and s0 <= step <= s1:
            out.append(tag)
    return out


def step_to_synthetic_month(step: int, n_total_steps: int,
                            anchor_month: int = 9) -> int:
    """
    Map a relative timestep to a synthetic calendar month, assuming the
    middle step (n_total_steps // 2) corresponds to ``anchor_month``.

    Used by temporal_stats.seasonal_contrast to keep working with relative tags.
    """
    mid = n_total_steps // 2
    step_months = 0.5  # 15-day windows -> 0.5 month each
    month = anchor_month + (step - mid) * step_months
    month = int(round(month)) % 12
    if month <= 0:
        month += 12
    return month
