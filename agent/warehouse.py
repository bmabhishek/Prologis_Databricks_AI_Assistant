"""
Databricks SQL Warehouse status / wake-up helpers.

Serverless warehouses auto-stop after `auto_stop_mins` of inactivity, and the
first query against a stopped warehouse blocks until it has started. These
helpers let the UI show the warehouse state and start it on demand via the
SQL Warehouses REST API (https://docs.databricks.com/api/workspace/warehouses),
falling back to a trivial query through the SQL connector — which also
auto-starts the warehouse — if the token is not allowed to call `/start`.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

WAREHOUSES_API = "/api/2.0/sql/warehouses"

# Databricks warehouse states -> (emoji, human label)
STATE_DISPLAY = {
    "RUNNING":  ("🟢", "Running"),
    "STARTING": ("🟡", "Starting"),
    "STOPPED":  ("⚪", "Asleep"),
    "STOPPING": ("🟠", "Stopping"),
    "DELETING": ("🔴", "Deleting"),
    "DELETED":  ("🔴", "Deleted"),
}


def _config() -> tuple[str, str, dict]:
    """Return (host, warehouse_id, auth headers) from the DATABRICKS_* env vars."""
    host = os.getenv("DATABRICKS_SERVER_HOSTNAME", "").strip().rstrip("/")
    http_path = os.getenv("DATABRICKS_HTTP_PATH", "")
    token = os.getenv("DATABRICKS_ACCESS_TOKEN")
    if not host or not http_path or not token:
        raise RuntimeError(
            "DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, and "
            "DATABRICKS_ACCESS_TOKEN must be set."
        )
    match = re.search(r"warehouses/([A-Za-z0-9]+)", http_path)
    if not match:
        raise RuntimeError(f"Cannot derive a warehouse id from DATABRICKS_HTTP_PATH={http_path!r}")
    host = host.replace("https://", "").replace("http://", "")
    return host, match.group(1), {"Authorization": f"Bearer {token}"}


class WarehouseUnavailable(RuntimeError):
    """The warehouse is not running and Databricks refused to start it
    (e.g. Free Edition's daily compute quota). Carries Databricks' message."""


def _error_message(resp: requests.Response) -> str:
    try:
        body = resp.json()
        return body.get("message") or body.get("error_code") or resp.text
    except ValueError:
        return resp.text or f"HTTP {resp.status_code}"


def get_warehouse_status(timeout: float = 10) -> dict:
    """Fetch the warehouse's current state and a few descriptive fields."""
    host, warehouse_id, headers = _config()
    resp = requests.get(f"https://{host}{WAREHOUSES_API}/{warehouse_id}", headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return {
        "id": warehouse_id,
        "name": data.get("name"),
        "state": data.get("state", "UNKNOWN"),
        "type": data.get("warehouse_type"),
        "serverless": bool(data.get("enable_serverless_compute")),
        "size": data.get("cluster_size"),
        "auto_stop_mins": data.get("auto_stop_mins"),
        "health": (data.get("health") or {}).get("status"),
        "url": f"https://{host}/sql/warehouses/{warehouse_id}",
        "checked_at": time.time(),
    }


def start_warehouse(timeout: float = 10) -> None:
    """Ask Databricks to start the warehouse (no-op if already running).

    Raises WarehouseUnavailable when Databricks refuses (most commonly the
    Free Edition daily limit: "you have hit your free daily limit"), and
    requests.HTTPError for auth/permission problems (401/403).
    """
    host, warehouse_id, headers = _config()
    resp = requests.post(f"https://{host}{WAREHOUSES_API}/{warehouse_id}/start", headers=headers, timeout=timeout)
    if resp.status_code in (401, 403):
        resp.raise_for_status()
    if not resp.ok:
        raise WarehouseUnavailable(_error_message(resp))


def ping_warehouse() -> float:
    """Run `SELECT 1` through the SQL connector. Blocks until the warehouse
    serves it (starting it if needed). Returns elapsed seconds."""
    from agent.tools import _get_databricks_connection
    t0 = time.time()
    with _get_databricks_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchall()
    return time.time() - t0


def wake_warehouse(
    poll_seconds: float = 3,
    max_wait: float = 180,
    on_update: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Start the warehouse if it is stopped and poll until it is RUNNING.

    `on_update(status)` is called after every poll so a UI can show progress.
    If the REST start call is refused (e.g. a token scoped to SQL only), fall
    back to `ping_warehouse()`, which starts the warehouse implicitly.
    """
    status = get_warehouse_status()
    if status["state"] == "RUNNING":
        return status

    if status["state"] == "STOPPED":
        try:
            start_warehouse()
        except requests.HTTPError as e:
            # Token not allowed to call /start — a query starts the warehouse implicitly.
            code = e.response.status_code if e.response is not None else "?"
            if on_update:
                on_update({**status, "note": f"Token may not call /start (HTTP {code}); waking via SELECT 1 instead…"})
            ping_warehouse()
            return get_warehouse_status()

    deadline = time.time() + max_wait
    while time.time() < deadline:
        status = get_warehouse_status()
        if on_update:
            on_update(status)
        if status["state"] == "RUNNING":
            break
        time.sleep(poll_seconds)
    if status["state"] != "RUNNING":
        raise WarehouseUnavailable(
            f"Warehouse is still {status['state']} after {int(max_wait)} s — try again shortly."
        )
    return status


def ensure_warehouse_running(on_update: Optional[Callable[[dict], None]] = None) -> Optional[bool]:
    """Make sure the warehouse is up before a query.

    Returns True if it had to be woken, False if it was already running, or
    None if the state could not be checked (the query is then attempted as-is,
    since the SQL connector starts a warehouse implicitly).
    Raises WarehouseUnavailable if it is asleep and cannot be started.
    """
    try:
        status = get_warehouse_status()
    except Exception:
        return None
    if status["state"] == "RUNNING":
        return False
    if on_update:
        on_update(status)
    wake_warehouse(on_update=on_update)
    return True


if __name__ == "__main__":
    s = get_warehouse_status()
    print(f"{s['name']} [{s['id']}] -> {s['state']} (serverless={s['serverless']}, auto_stop={s['auto_stop_mins']} min)")
