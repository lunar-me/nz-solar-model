"""Supabase (PostgREST) data access for the NZ solar model.

Replaces direct reads of the local CAMS / electricity CSVs with live
queries against the Supabase project. Credentials are read from the
environment (``SUPABASE_URL``, ``SUPABASE_PUBLISHABLE_KEY``), loaded from
the repo-root ``.env`` via python-dotenv when present.

The publishable key is used as both the ``apikey`` and ``Authorization``
bearer token against the PostgREST endpoint (``/rest/v1/<table>``), which
is the standard client-side access pattern for Supabase.
"""
from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

# Load .env from the repo root if present (keeps credentials out of code).
# api/supabase_client.py -> parent.parent == project root.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    from dotenv import load_dotenv

    load_dotenv(_ENV_PATH)

# Tolerate a UTF-8 BOM that some editors prepend to the first .env key.
SUPABASE_URL = os.environ.get("SUPABASE_URL") or os.environ.get("\ufeffSUPABASE_URL", "")
SUPABASE_KEY = (
    os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    or os.environ.get("\ufeffSUPABASE_PUBLISHABLE_KEY", "")
)
SUPABASE_URL = SUPABASE_URL.strip().rstrip("/")
SUPABASE_KEY = SUPABASE_KEY.strip()

RADIATION_TABLE = "cams_radiation"
ELECTRICITY_TABLE = "christchurch_electricity_consumption"

# Only the columns the app actually reads — we never SELECT * (dropping
# `location`, `end_ts_utc`, `toa`, `clear_sky_bhi`, `bhi`, ... reduces payload).
RADIATION_COLUMNS = (
    "start_ts_utc,ghi,dhi,bni,clear_sky_ghi,clear_sky_dhi,clear_sky_bni,reliability"
)
ELECTRICITY_COLUMNS = "datetime_utc,usage_kWh,dollars"

# PostgREST default / maximum rows returned per request.
_PAGE_SIZE = 1000

# Audit log: every Supabase fetch is recorded here (table, filters, rows,
# elapsed time) so data provenance / performance can be inspected later.
#
# The log FILE is only written when running locally — detected by the presence
# of the repo-root `.env` (gitignored, so it never exists on Vercel; credentials
# there come from env vars). Vercel's serverless filesystem is ephemeral and
# nothing persists, so on Vercel we send the same records to stdout instead,
# where they surface in the platform's function logs.
_IS_LOCAL = _ENV_PATH.exists()
_LOG_FILE = Path(__file__).resolve().parent / "supabase_requests.log"
_logger = logging.getLogger("supabase_client")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _fmt = logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    _handler = (
        logging.FileHandler(_LOG_FILE, encoding="utf-8")
        if _IS_LOCAL
        else logging.StreamHandler()
    )
    _handler.setFormatter(_fmt)
    _logger.addHandler(_handler)
    _logger.propagate = False


def _headers() -> dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Supabase credentials missing. Set SUPABASE_URL and "
            "SUPABASE_PUBLISHABLE_KEY in the .env file."
        )
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json",
    }


def _fetch_all(table: str, select: str, filters: list[tuple[str, str]],
               order: str) -> list[dict]:
    """Fetch every row of ``table`` (paginated) matching ``filters``.

    ``select`` is the comma-separated list of columns to return (only what the
    app actually reads — never ``*``).  ``filters`` is a list of
    ``(column, operator.value)`` pairs; PostgREST accepts duplicate keys (e.g.
    ``start_ts_utc=gte.X`` + ``start_ts_utc=lte.Y``), so it must be a list
    rather than a dict.

    Pages are fetched concurrently (PostgREST caps each request at 1000 rows)
    over a single connection-pooled client, so the TCP/TLS handshake is paid
    once and the full multi-year dataset loads in a few seconds.
    """
    url = f"https://{SUPABASE_URL}/rest/v1/{table}"
    headers = _headers()
    base: list[tuple[str, str]] = [("select", select), ("order", order), *filters]
    _t0 = time.perf_counter()

    with httpx.Client(timeout=60.0) as client:
        # Probe with count=exact to learn the total row count, so we can size
        # the page fan-out up front instead of walking pages one at a time.
        probe = client.get(
            url,
            headers={**headers, "Prefer": "count=exact"},
            params=[*base, ("limit", "1"), ("offset", "0")],
        )
        probe.raise_for_status()
        total: int | None = None
        content_range = probe.headers.get("content-range", "")
        if "/" in content_range:
            try:
                total = int(content_range.split("/")[1])
            except ValueError:
                total = None

        if total is None:
            # Fall back to sequential pagination if the count is unavailable.
            rows: list[dict] = []
            offset = 0
            while True:
                resp = client.get(
                    url,
                    headers=headers,
                    params=[*base, ("limit", str(_PAGE_SIZE)), ("offset", str(offset))],
                )
                resp.raise_for_status()
                batch = resp.json()
                rows.extend(batch)
                if len(batch) < _PAGE_SIZE:
                    break
                offset += _PAGE_SIZE
            return _log_fetch(table, select, filters, order, rows, _t0)

        n_pages = math.ceil(total / _PAGE_SIZE)

        def fetch_page(page: int) -> list[dict]:
            offset = page * _PAGE_SIZE
            query = [*base, ("limit", str(_PAGE_SIZE)), ("offset", str(offset))]
            # Retry transient server errors (503/429/5xx) with short backoff.
            for attempt in range(5):
                try:
                    resp = client.get(url, headers=headers, params=query)
                    resp.raise_for_status()
                    return resp.json()
                except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                    status = getattr(exc, "response", None)
                    code = status.status_code if status is not None else None
                    if code in (503, 429) or (code is not None and code >= 500):
                        time.sleep(0.2 * (2 ** attempt))
                        continue
                    raise
            raise RuntimeError(f"Failed to fetch page {page} after retries")

        with ThreadPoolExecutor(max_workers=12) as pool:
            pages = list(pool.map(fetch_page, range(n_pages)))

    rows = [row for page in pages for row in page]
    return _log_fetch(table, select, filters, order, rows, _t0)


# PostgREST operator suffix -> SQL operator.
_OP_MAP = {
    "eq": "=", "neq": "<>", "gt": ">", "gte": ">=",
    "lt": "<", "lte": "<=", "like": "LIKE", "ilike": "ILIKE",
    "is": "IS", "in": "IN",
}


def _sql_str(v: str) -> str:
    """Quote a scalar as a SQL string literal (escaping embedded quotes)."""
    return "'" + v.replace("'", "''") + "'"


def _build_sql(table: str, select: str, filters: list[tuple[str, str]],
               order: str) -> str:
    """Render the SQL that the PostgREST request corresponds to."""
    clauses: list[str] = []
    for col, expr in filters:
        op, sep, val = expr.partition(".")
        if not sep:
            continue
        op_sql = _OP_MAP.get(op, op)
        if op == "in":
            items = ", ".join(
                _sql_str(x.strip()) for x in val.strip("()").split(",") if x.strip()
            )
            clauses.append(f'"{col}" IN ({items})')
        else:
            clauses.append(f'"{col}" {op_sql} {_sql_str(val)}')
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    order_col, _, direction = order.rpartition(".")
    dir_sql = "DESC" if direction == "desc" else "ASC"
    sel = "*" if select == "*" else ", ".join(
        f'"{c.strip()}"' for c in select.split(",") if c.strip()
    )
    return f"SELECT {sel} FROM \"{table}\"{where} ORDER BY \"{order_col}\" {dir_sql}"


def _log_fetch(table: str, select: str, filters: list[tuple[str, str]], order: str,
               rows: list[dict], t0: float) -> list[dict]:
    """Log one Supabase fetch (table, SQL, row count, elapsed) to the log."""
    elapsed = time.perf_counter() - t0
    sql = _build_sql(table, select, filters, order)
    _logger.info("%s | rows=%d | %.2fs | SQL: %s", table, len(rows), elapsed, sql)
    return rows


def fetch_radiation(location: str, start: str | None = None,
                    end: str | None = None) -> list[dict]:
    """Fetch ``cams_radiation`` rows for ``location`` (paginated).

    ``start`` / ``end`` are ISO-8601 UTC timestamps that restrict the range via
    PostgREST ``start_ts_utc`` filters (``>=`` and ``<=`` inclusive).
    """
    filters: list[tuple[str, str]] = [("location", f"eq.{location}")]
    if start:
        filters.append(("start_ts_utc", f"gte.{start}"))
    if end:
        filters.append(("start_ts_utc", f"lte.{end}"))
    return _fetch_all(
        RADIATION_TABLE, RADIATION_COLUMNS, filters=filters, order="start_ts_utc.asc"
    )


def fetch_electricity() -> list[dict]:
    """Fetch every ``christchurch_electricity_consumption`` row (paginated)."""
    return _fetch_all(
        ELECTRICITY_TABLE,
        ELECTRICITY_COLUMNS,
        filters=[],
        order="datetime_utc.asc",
    )


class RPCFunctionNotFoundError(RuntimeError):
    """A PostgREST RPC function is not installed on the Supabase project yet."""


def fetch_data_quality(location: str) -> dict:
    """Compute the Data Quality report server-side via the PostgREST RPC.

    Delegates the whole report to the ``get_data_quality`` Postgres function
    (see ``supabase/get_data_quality.sql``), which aggregates over the table
    in the database and returns a single JSON object — so the app never
    downloads the full multi-year dataset for this tab.

    Raises :class:`RPCFunctionNotFoundError` if the function hasn't been
    created yet (the caller may fall back to the in-app computation).
    """
    url = f"https://{SUPABASE_URL}/rest/v1/rpc/get_data_quality"
    _t0 = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, headers=_headers(), json={"location": location})
        if resp.status_code == 404:
            raise RPCFunctionNotFoundError("get_data_quality")
        resp.raise_for_status()
    elapsed = time.perf_counter() - _t0
    _logger.info("rpc.get_data_quality | %.2fs | location=%s",
                 elapsed, location)
    return resp.json()
