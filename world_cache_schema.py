"""Shared schema validation for cached global city datasets."""

from __future__ import annotations

from typing import Set

import pandas as pd

WORLD_CACHE_REQUIRED_COLUMNS: Set[str] = {
    "city",
    "state",
    "country",
    "continent",
    "country_code",
    "iso3",
    "population",
    "lat",
    "lon",
    "avg_rent_2_bed",
    "monthly_living_cost",
    "cost_of_living",
    "quality_of_life",
    "crime",
    "pollution_level",
    "summer_high",
    "winter_low",
    "earthquakes",
    "tornadoes",
    "beach",
    "mountains",
    "government_form",
    "language",
    "currency",
    "life_expectancy",
    "migration_rate",
    "freedom_score",
    "lgbt_equality_index",
    "lgbt_legal_index",
    "lgbt_public_opinion_index",
}


def validate_world_cache(data: pd.DataFrame) -> None:
    """Raise a ValueError when required columns are missing."""
    missing = WORLD_CACHE_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"World cache missing columns: {', '.join(sorted(missing))}")
