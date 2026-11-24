"""``output="by_row"``: a per-manifest-row status report for batch reads.

A batch read manifest mixes three kinds of rows — registered series with
data in the window, registered series with no data yet, and names that were
never registered — and the caller wants one statused entry per *row*, in
the manifest's own order, with repeated tags answered per occurrence rather
than collapsed onto a dict key (which is what ``output="by_path"`` does).

The pure tests drive ``execute_read`` over a mocked PG connection and a
stubbed timedb client; the live tests cover the client surface end to end,
including engine-vs-sequential parity.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import energydb as edb
import pandas as pd
import polars as pl
import pytest
from energydb import Client, EdgeSeriesKey, ReadResult, RowResult, SeriesKey
from energydb._frames import to_backend
from energydb._io import execute_read
from energydb.client import AsyncClient
from energydb.errors import SeriesNotFoundError, ValidationError

BASE = datetime(2026, 3, 1, tzinfo=UTC)
_TS = pl.Datetime("us", "UTC")


# ---------------------------------------------------------------------------
# Fakes: PG resolve + ClickHouse read
# ---------------------------------------------------------------------------


def _mock_conn_with_series(rows: list[tuple]) -> MagicMock:
    """An async-aware mock connection for the ``await (await execute).fetchall()`` path."""
    cursor = MagicMock()
    cursor.fetchall = AsyncMock(return_value=rows)
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=cursor)
    return conn


class _FakeClient:
    """The ``execute_read`` client surface: engine-disabled session + read conn."""

    def __init__(self, conn) -> None:
        self._engine_unavailable = True
        self._conn = conn

    @asynccontextmanager
    async def _read_conn(self):
        yield self._conn


class _FakeTd:
    """Stands in for ``TimeDBClient``: replays a canned CH frame for the requested ids."""

    def __init__(self, ch_result: pl.DataFrame) -> None:
        self._ch_result = ch_result
        self.calls: list[list[int]] = []

    def _replay(self, series_ids) -> pl.DataFrame:
        self.calls.append(list(series_ids))
        return self._ch_result.filter(pl.col("series_id").is_in(series_ids))

    def read(self, *, series_ids, retention=None, meta_source=None, **kwargs):
        return self._replay(series_ids)

    def read_relative(self, *, series_ids, retention=None, meta_source=None, **kwargs):
        return self._replay(series_ids)


# Path-route PG row: path, data_type, name, series_id, canonical_unit,
# timeseries_type, retention, node_uuid::text.
def _path_row(path: str, series_id: int, *, dt: str = "actual", name: str = "power") -> tuple:
    return (path, dt, name, series_id, "MW", "FLAT", "forever", str(uuid4()))


def _ch_frame(series_rows: dict[int, list[float]]) -> pl.DataFrame:
    """A CH value frame: one row per (series_id, hourly valid_time, value)."""
    sids, vts, values = [], [], []
    for sid, vals in series_rows.items():
        for i, v in enumerate(vals):
            sids.append(sid)
            vts.append(BASE + timedelta(hours=i))
            values.append(v)
    return pl.DataFrame(
        {"series_id": sids, "valid_time": vts, "value": values},
        schema={"series_id": pl.UInt64, "valid_time": _TS, "value": pl.Float64},
    )


def _manifest(*paths: str, dt: str = "actual", name: str = "power") -> pl.DataFrame:
    return pl.DataFrame({"path": list(paths), "data_type": [dt] * len(paths), "name": [name] * len(paths)})


def _execute(manifest: pl.DataFrame, pg_rows: list[tuple], ch: pl.DataFrame, **kwargs):
    """Run ``execute_read`` over the fakes; return ``((entries, n_series, missing), td)``."""
    td = _FakeTd(ch)
    client = _FakeClient(_mock_conn_with_series(pg_rows))
    kwargs.setdefault("output", "by_row")
    result = asyncio.run(execute_read(None, td, client, manifest=manifest, **kwargs))
    return result, td


# ---------------------------------------------------------------------------
# The mixed batch: data / no data yet / unregistered / a repeated tag
# ---------------------------------------------------------------------------


def test_mixed_batch_reports_every_row_in_manifest_order():
    """**The feature's core.** One entry per manifest row, in the manifest's
    own order — not the resolver's, not the dict's."""
    manifest = _manifest("P/B", "P/A", "P/C", "P/A")
    pg = [_path_row("P/A", 1), _path_row("P/B", 2)]  # P/C is not registered
    ch = _ch_frame({1: [1.0, 2.0]})  # P/B's series has no rows in the window

    (entries, n_series, missing), td = _execute(manifest, pg, ch, on_missing="skip")

    assert [e.status for e in entries] == ["empty", "ok", "missing", "ok"]
    assert [e.key for e in entries] == [
        SeriesKey("P/B", "actual", "power"),
        SeriesKey("P/A", "actual", "power"),
        SeriesKey("P/C", "actual", "power"),
        SeriesKey("P/A", "actual", "power"),
    ]
    # ok: the values came back, attached to the row that asked for them.
    assert entries[1].df["value"].to_list() == [1.0, 2.0]
    assert set(entries[1].df.columns) == {"valid_time", "value"}  # data columns only
    # empty: registered, nothing in the window — an empty, schema-ful frame.
    assert entries[0].df.is_empty()
    assert set(entries[0].df.columns) == {"valid_time", "value"}
    # missing: unregistered — df is None, and only None marks it.
    assert entries[2].df is None
    assert n_series == 2
    assert missing.rows() == [("P/C", "actual", "power")]
    # The CH read went out once, for exactly the resolved series.
    assert len(td.calls) == 1
    assert set(td.calls[0]) == {1, 2}


def test_a_repeated_tag_gets_one_entry_per_occurrence():
    """**No dict-key collapse.** ``by_path`` would key all three requests of
    ``P/A`` onto one entry; ``by_row`` answers each occurrence."""
    manifest = _manifest("P/A", "P/A", "P/A")
    (entries, n_series, _missing), _td = _execute(manifest, [_path_row("P/A", 1)], _ch_frame({1: [1.0]}))

    assert len(entries) == 3
    assert n_series == 1  # one underlying series, three requested rows
    assert all(e.status == "ok" for e in entries)
    assert all(e.key == SeriesKey("P/A", "actual", "power") for e in entries)
    # The three entries share the one series' frame — same series, same data.
    assert entries[0].df is entries[1].df is entries[2].df


def test_raise_still_fails_the_whole_batch_on_an_unregistered_row():
    """``by_row`` does not change ``on_missing="raise"``: catalog gaps raise."""
    with pytest.raises(SeriesNotFoundError):
        _execute(_manifest("P/A", "P/C"), [_path_row("P/A", 1)], _ch_frame({1: [1.0]}))


def test_raise_mode_with_everything_registered_reports_ok_and_empty():
    (entries, n_series, missing), _td = _execute(
        _manifest("P/A", "P/B"),
        [_path_row("P/A", 1), _path_row("P/B", 2)],
        _ch_frame({1: [1.0]}),
    )

    assert [e.status for e in entries] == ["ok", "empty"]
    assert n_series == 2
    assert missing.is_empty()


def test_everything_unregistered_reports_all_missing_without_a_ch_read():
    """The all-gaps case: no series resolved, so no CH read happens, and every
    row still gets its own ``missing`` entry (the ``_empty`` branch)."""
    (entries, n_series, missing), td = _execute(_manifest("P/X", "P/Y"), [], _ch_frame({}), on_missing="skip")

    assert [e.status for e in entries] == ["missing", "missing"]
    assert [e.key for e in entries] == [SeriesKey("P/X", "actual", "power"), SeriesKey("P/Y", "actual", "power")]
    assert all(e.df is None for e in entries)
    assert n_series == 0
    assert td.calls == []
    assert missing.height == 2


def test_data_type_case_is_normalized_on_missing_rows_too():
    """A missing row's key must lowercase ``data_type`` exactly like a resolved
    row's, or the same logical tag reports two different keys."""
    manifest = pl.DataFrame(
        {"path": ["P/A", "P/B"], "data_type": ["Actual", "ACTUAL"], "name": ["power", "power"]}
    )
    (entries, _n, missing), _td = _execute(manifest, [_path_row("P/A", 1)], _ch_frame({1: [1.0]}), on_missing="skip")

    assert [e.status for e in entries] == ["ok", "missing"]
    assert entries[0].key == SeriesKey("P/A", "actual", "power")
    assert entries[1].key == SeriesKey("P/B", "actual", "power")
    assert missing.rows() == [("P/B", "actual", "power")]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_node_uuid_route_keys_resolved_rows_by_db_path_and_missing_rows_none():
    """Resolved rows get the same DB-derived key ``by_path`` would produce.
    An unregistered uuid-route row has no public path identity, so its key is
    ``None`` — the entry's position (and ``missing``) still identifies it."""
    present, absent = uuid4(), uuid4()
    # Owner-route PG row: owner::text, data_type, name, series_id, unit, ts_type, retention, path.
    pg = [(str(present), "actual", "power", 7, "MW", "FLAT", "forever", "P/A")]
    manifest = pl.DataFrame(
        {
            "node_uuid": [present, absent],  # UUID objects, not strings
            "data_type": ["actual", "actual"],
            "name": ["power", "power"],
        }
    )
    (entries, _n, missing), _td = _execute(manifest, pg, _ch_frame({7: [3.0]}), on_missing="skip")

    assert entries[0].status == "ok"
    assert entries[0].key == SeriesKey("P/A", "actual", "power")
    assert entries[1].status == "missing"
    assert entries[1].key is None
    assert entries[1].df is None
    assert missing.rows() == [(str(absent), "actual", "power")]


def test_edge_triple_route_reports_the_full_quintuple_key():
    """A missing edge-triple row keeps its whole edge identity in the key —
    a three-column report would not say which edge was missing."""
    # Edge-triple PG row: from, to, edge_type, edge_name, data_type, series name,
    # series_id, unit, ts_type, retention, edge_uuid::text.
    pg = [("G/A", "G/B", "Line", None, "actual", "flow", 11, "MW", "FLAT", "forever", str(uuid4()))]
    manifest = pl.DataFrame(
        {
            "from_path": ["G/A", "G/A"],
            "to_path": ["G/B", "G/NOPE"],
            "edge_type": ["Line", "Line"],
            "data_type": ["actual", "actual"],
            "name": ["flow", "flow"],
        }
    )
    (entries, _n, missing), _td = _execute(manifest, pg, _ch_frame({11: [5.0]}), on_missing="skip")

    assert entries[0].status == "ok"
    assert entries[0].key == EdgeSeriesKey("G/A", "G/B", "Line", None, "actual", "flow")
    assert entries[1].status == "missing"
    assert entries[1].key == EdgeSeriesKey("G/A", "G/NOPE", "Line", None, "actual", "flow")
    assert missing.rows() == [("G/A", "G/NOPE", "Line", "actual", "flow")]


# ---------------------------------------------------------------------------
# Pipeline plumbing
# ---------------------------------------------------------------------------


def test_by_row_requires_a_manifest():
    """Scope reads resolve a subtree, not a caller-ordered manifest."""

    async def _resolve():
        return None

    with pytest.raises(ValidationError, match="by_row"):
        asyncio.run(
            execute_read(None, None, _FakeClient(MagicMock()), resolve=_resolve, is_edge=False, output="by_row")
        )


def test_an_invalid_output_value_is_rejected():
    with pytest.raises(ValidationError, match="output"):
        _execute(_manifest("P/A"), [_path_row("P/A", 1)], _ch_frame({1: [1.0]}), output="not_a_mode")


def test_structural_manifest_errors_still_raise_under_by_row():
    """A malformed manifest is a caller bug, not a per-row status."""
    no_data_type = pl.DataFrame({"path": ["P/A"], "name": ["power"]})
    with pytest.raises(ValidationError, match="required columns"):
        _execute(no_data_type, [], _ch_frame({}))


def test_opt_in_columns_ride_into_the_sub_frames():
    """``include_knowledge_time`` columns are data columns: they belong on the
    per-row frames just like on ``by_path`` sub-frames."""
    ch = _ch_frame({1: [1.0]}).with_columns(pl.lit(BASE, dtype=_TS).alias("knowledge_time"))
    (entries, _n, _m), _td = _execute(
        _manifest("P/A"), [_path_row("P/A", 1)], ch, include_knowledge_time=True
    )

    assert entries[0].status == "ok"
    assert "knowledge_time" in entries[0].df.columns


def test_relative_reads_report_per_row_too():
    (entries, _n, missing), _td = _execute(
        _manifest("P/A", "P/C"),
        [_path_row("P/A", 1)],
        _ch_frame({1: [1.0, 2.0]}),
        relative=True,
        td_kwargs={},
        on_missing="skip",
    )

    assert [e.status for e in entries] == ["ok", "missing"]
    assert entries[0].df.height == 2
    assert missing.rows() == [("P/C", "actual", "power")]


# ---------------------------------------------------------------------------
# Backend conversion and the ReadResult wrap
# ---------------------------------------------------------------------------


def _two_entries() -> list[RowResult]:
    return [
        RowResult(SeriesKey("P/A", "actual", "power"), "ok", pl.DataFrame({"value": [1.0]})),
        RowResult(SeriesKey("P/B", "actual", "power"), "missing", None),
    ]


def test_polars_backend_returns_the_entry_list_untouched():
    entries = _two_entries()
    assert to_backend(entries, "polars") is entries


def test_pandas_backend_converts_each_entry_frame_and_keeps_none():
    out = to_backend(_two_entries(), "pandas")

    assert isinstance(out, list)
    assert isinstance(out[0].df, pd.DataFrame)
    assert out[0].df["value"].to_list() == [1.0]
    assert out[1].df is None
    # Keys and statuses survive conversion untouched.
    assert out[0].key == SeriesKey("P/A", "actual", "power")
    assert [e.status for e in out] == ["ok", "missing"]


def test_with_missing_wraps_by_row_in_a_read_result_only_on_skip():
    entries = _two_entries()
    missing = pl.DataFrame(schema={"path": pl.Utf8, "data_type": pl.Utf8, "name": pl.Utf8})

    wrapped = AsyncClient._with_missing(entries, missing, backend="polars", on_missing="skip")
    assert isinstance(wrapped, ReadResult)
    assert wrapped.data is entries
    assert wrapped.missing.is_empty()

    bare = AsyncClient._with_missing(entries, missing, backend="polars", on_missing="raise")
    assert bare is entries


def test_row_result_is_a_public_unpackable_namedtuple():
    assert edb.RowResult is RowResult
    key, status, df = RowResult(SeriesKey("P/A", "actual", "power"), "ok", None)
    assert key == ("P/A", "actual", "power")  # SeriesKey stays tuple-compatible
    assert status == "ok"
    assert df is None


def test_field_aliases_and_direct_key_delegation():
    row = RowResult(SeriesKey("P/A", "actual", "power"), "ok", "frame")
    # The fetched frame is reachable under its common names.
    assert row.data is row.df
    assert row.frame is row.df
    assert row.values is row.df
    assert row.series_key is row.key
    # Identity fields delegate straight to the key.
    assert row.path == "P/A"
    assert row.data_type == "actual"
    assert row.name == "power"
    # A missing uuid-routed row has key=None: every delegate is None, no raise.
    missing = RowResult(None, "missing", None)
    assert missing.path is None and missing.name is None and missing.data is None


def test_alternate_type_names_are_the_same_class():
    from energydb import ManifestRow, ReadRow, Row

    assert ReadRow is RowResult
    assert ManifestRow is RowResult
    assert Row is RowResult
    assert issubclass(ReadRow, tuple)


@pytest.mark.parametrize("mode", ["by_rows", "rows", "row", "per_row"])
def test_output_mode_spelling_aliases_return_rows(mode):
    """``by_row`` is canonical, but the obvious spelling variants behave alike."""
    manifest = _manifest("P/A", "P/Ghost")
    pg = [_path_row("P/A", 1)]
    ch = _ch_frame({1: [1.0]})

    (entries, _n, missing), _td = _execute(manifest, pg, ch, output=mode, on_missing="skip")

    assert isinstance(entries, list)
    assert [e.status for e in entries] == ["ok", "missing"]
    assert missing.height == 1


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

pytestmark_live = pytest.mark.skipif(
    not (os.environ.get("TIMEDB_PG_DSN") and os.environ.get("TIMEDB_CH_URL")),
    reason="TIMEDB_PG_DSN / TIMEDB_CH_URL not set",
)


@pytest.fixture
def client():
    """T1 has a series with data, T2 a series with no data yet, T3 no series."""
    c = Client()
    c.delete()
    c.create()
    c.register_tree(
        edb.Portfolio(
            name="P",
            members=[edb.wind.WindTurbine(name=n, capacity=3.0) for n in ("T1", "T2", "T3")],
        )
    )
    for name in ("T1", "T2"):
        c.get_node("P", name).register_series(
            name="power", canonical_unit="MW", data_type="actual", timeseries_type="FLAT", retention="forever"
        )
    c.write(
        pl.DataFrame(
            {
                "path": ["P/T1", "P/T1"],
                "data_type": ["actual"] * 2,
                "name": ["power"] * 2,
                "valid_time": [BASE, BASE + timedelta(hours=1)],
                "value": [1.0, 2.0],
            }
        )
    )
    yield c
    c.delete()
    c.close()


@pytestmark_live
def test_live_mixed_batch_reports_each_row(client):
    """The three row kinds plus a repeat, through the real stack."""
    data, missing = client.read(_manifest("P/T1", "P/T2", "P/T3", "P/T1"), output="by_row", on_missing="skip")

    assert [e.status for e in data] == ["ok", "empty", "missing", "ok"]
    assert [e.key for e in data] == [
        SeriesKey("P/T1", "actual", "power"),
        SeriesKey("P/T2", "actual", "power"),
        SeriesKey("P/T3", "actual", "power"),
        SeriesKey("P/T1", "actual", "power"),
    ]
    assert data[0].df["value"].to_list() == [1.0, 2.0]
    assert data[1].df.is_empty()
    assert data[2].df is None
    assert data[3].df is data[0].df
    assert missing.rows() == [("P/T3", "actual", "power")]


@pytestmark_live
def test_live_raise_still_fails_the_batch(client):
    with pytest.raises(SeriesNotFoundError):
        client.read(_manifest("P/T1", "P/T3"), output="by_row")

    entries = client.read(_manifest("P/T1", "P/T2"), output="by_row")
    assert isinstance(entries, list)
    assert [e.status for e in entries] == ["ok", "empty"]


@pytestmark_live
def test_live_by_row_pandas_backend(client):
    data, _missing = client.read(_manifest("P/T1", "P/T3"), output="by_row", backend="pandas", on_missing="skip")

    assert isinstance(data[0].df, pd.DataFrame)
    assert data[0].df["value"].to_list() == [1.0, 2.0]
    assert data[1].df is None


@pytestmark_live
def test_live_read_relative_takes_by_row_too(client):
    data, missing = client.read_relative(
        _manifest("P/T1", "P/T3"),
        output="by_row",
        on_missing="skip",
        window_length=timedelta(hours=2),
        issue_offset=timedelta(days=365),
        start_window=BASE,
    )

    assert [e.status for e in data] == ["ok", "missing"]
    assert data[0].df.height == 2
    assert missing.rows() == [("P/T3", "actual", "power")]


@pytestmark_live
def test_live_sync_client_returns_a_plain_list_of_row_results(client):
    """End-to-end through the portal: no ``_SyncProxy``, and the entry unpacks."""
    result = client.read(_manifest("P/T1"), output="by_row")

    assert type(result) is list
    assert type(result[0]) is RowResult
    key, status, df = result[0]
    assert key.path == "P/T1"
    assert status == "ok"
    assert df["value"].to_list() == [1.0, 2.0]


@pytestmark_live
def test_live_engine_read_matches_sequential(client, monkeypatch):
    """The engine-parallel path must produce byte-identical per-row entries."""
    import energydb._io as _io
    from energydb._ch_meta_engine import CH_ENGINE_TABLE

    try:
        client.setup_ch_meta_engine()
        client.td._ch.command(f"SELECT count() FROM {CH_ENGINE_TABLE}")  # force a CH->PG connection
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"CH meta engine not usable in this env: {exc}")
    monkeypatch.setattr(_io, "_ENGINE_STRICT", True)  # an engine failure raises instead of hiding
    manifest = _manifest("P/T1", "P/T2", "P/T3", "P/T1")

    client._async._engine_unavailable = True
    sequential = client.read(manifest, output="by_row", on_missing="skip")
    client._async._engine_unavailable = False
    engine = client.read(manifest, output="by_row", on_missing="skip")

    assert client._async._engine_unavailable is False  # the engine really was used
    assert [e.status for e in engine.data] == [e.status for e in sequential.data]
    assert [e.key for e in engine.data] == [e.key for e in sequential.data]
    for eng, seq in zip(engine.data, sequential.data, strict=True):
        if eng.df is None:
            assert seq.df is None
        else:
            assert eng.df.equals(seq.df)
    assert engine.missing.equals(sequential.missing)
