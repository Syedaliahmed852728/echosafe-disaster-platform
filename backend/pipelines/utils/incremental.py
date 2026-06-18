"""
Shared helpers for incremental pipeline execution.

Every stage of every module (download -> bronze -> silver -> gold) targets the
same rolling window ``[today - SETTINGS.pipeline.lookback_years, today]``. This
module centralises:

* ``target_window()`` - resolve that window from settings.
* ``covers_window()``  - decide whether an existing dataset already spans it,
  so a stage can skip work entirely on a re-run.
* ``missing_windows()``- compute the head/tail gaps that still need fetching
  when the existing dataset only partially overlaps the target.

The helpers are intentionally tolerant on the upper end: upstream sources
typically lag the current day by 1-3 days (NASA POWER reanalysis, USGS
catalogue cut-off, IEM METAR rollover) so we accept a small ``end_lag_days``
when checking coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from backend.config.settings import SETTINGS


def target_window(
    lookback_years: Optional[int] = None,
    *,
    now: Optional[datetime] = None,
) -> Tuple[datetime, datetime]:
    """Return ``(start, end)`` for the project's rolling history window."""
    years = lookback_years if lookback_years is not None else SETTINGS.pipeline.lookback_years
    end = (now or datetime.now(timezone.utc)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = end - timedelta(days=365 * years)
    return start, end


def covers_window(
    existing: Optional[Tuple[datetime, datetime]],
    target: Tuple[datetime, datetime],
    *,
    end_lag_days: int = 2,
    start_slack_days: int = 1,
) -> bool:
    """True if ``existing`` already covers ``target`` within tolerance.

    ``end_lag_days`` absorbs the natural lag of upstream sources so a daily
    re-run doesn't refetch just because the very last day hasn't published.
    """
    if existing is None:
        return False
    e_min, e_max = existing
    t_start, t_end = target
    return (
        e_min <= t_start + timedelta(days=start_slack_days)
        and e_max >= t_end - timedelta(days=end_lag_days)
    )


def missing_windows(
    existing: Optional[Tuple[datetime, datetime]],
    target: Tuple[datetime, datetime],
) -> List[Tuple[datetime, datetime]]:
    """Gaps that need fetching to cover ``target`` given ``existing``."""
    t_start, t_end = target
    if existing is None:
        return [(t_start, t_end)]
    e_min, e_max = existing
    gaps: List[Tuple[datetime, datetime]] = []
    if e_min > t_start:
        gaps.append((t_start, e_min))
    if e_max < t_end:
        gaps.append((e_max, t_end))
    return gaps


def covers_upstream(
    existing: Optional[Tuple[datetime, datetime]],
    upstream: Optional[Tuple[datetime, datetime]],
    target: Tuple[datetime, datetime],
    *,
    slack_days: int = 0,
) -> bool:
    """True if ``existing`` already covers everything ``upstream`` exposes.

    Used by silver and gold stages: a transform has nothing to do when the
    output already spans at least as far back as ``target.start`` and at least
    as far forward as the upstream layer does. This way the transform doesn't
    re-run just because the upstream source itself hasn't published new data.
    """
    if existing is None:
        return False
    if upstream is None:
        # Nothing upstream to ingest; treat as covered iff existing reaches target.
        return covers_window(existing, target)
    e_min, e_max = existing
    u_min, u_max = upstream
    t_start, _ = target
    return (
        e_min <= max(t_start, u_min) + timedelta(days=slack_days)
        and e_max >= u_max - timedelta(days=slack_days)
    )
