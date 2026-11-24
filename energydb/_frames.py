"""Pandas/polars boundary adapters for the public API.

Public ``Client`` and scope methods accept either pandas or polars
on input and return polars by default (``backend="polars"``). The internal
manifest pipeline (``_io``, ``_join``, ``paths``, ``series``) is polars
throughout, so these helpers are the only conversion points.
"""

from __future__ import annotations

from typing import Literal, cast

import pandas as pd
import polars as pl

from energydb._join import EdgeSeriesKey, RowResult, SeriesKey
from energydb.errors import ValidationError

Backend = Literal["polars", "pandas"]
Output = Literal["frame", "by_path", "by_row"]


def to_polars(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Coerce a user-supplied DataFrame to polars."""
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def to_backend(
    result: pl.DataFrame | dict[SeriesKey, pl.DataFrame] | dict[EdgeSeriesKey, pl.DataFrame] | list[RowResult],
    backend: Backend,
) -> (
    pl.DataFrame
    | pd.DataFrame
    | dict[SeriesKey, pl.DataFrame]
    | dict[SeriesKey, pd.DataFrame]
    | dict[EdgeSeriesKey, pl.DataFrame]
    | dict[EdgeSeriesKey, pd.DataFrame]
    | list[RowResult]
):
    """Convert an internal polars result to the user-requested backend.

    Handles the long-format ``pl.DataFrame``, the per-series
    ``dict[SeriesKey | EdgeSeriesKey, pl.DataFrame]`` shape, and the
    ``output="by_row"`` ``list[RowResult]``. When ``backend="pandas"`` and
    the input is a dict or list, each frame inside is converted in turn.
    """
    if backend == "polars":
        return result
    if backend == "pandas":
        if isinstance(result, list):
            return [
                RowResult(entry.key, entry.status, entry.df.to_pandas() if entry.df is not None else None)
                for entry in result
            ]
        if isinstance(result, dict):
            # ty narrows the value type to object after the isinstance.
            d = cast(dict[SeriesKey | EdgeSeriesKey, pl.DataFrame], result)
            # dict invariance: one key type per call at runtime, but the union
            # must be collapsed statically before assigning to the return type.
            return cast(
                dict[SeriesKey, pd.DataFrame] | dict[EdgeSeriesKey, pd.DataFrame],
                {k: v.to_pandas() for k, v in d.items()},
            )
        return result.to_pandas()
    raise ValidationError(f"backend must be 'polars' or 'pandas', got {backend!r}")
