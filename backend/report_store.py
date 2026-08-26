"""SQLite storage for generated industrial operation reports."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


REPORT_COLUMNS = (
    "id",
    "user_id",
    "dataset_signature",
    "dataset_name",
    "created_at",
    "event_start_time",
    "event_end_time",
    "duration_hours",
    "anomaly_count",
    "suspected_cause",
    "diagnosis",
    "suggestion",
    "report",
)


def _default_db_path() -> Path:
    """Return the project-local report database path."""
    return Path(__file__).resolve().parents[1] / "data" / "report_history.db"


def _resolve_db_path(db_path: str | Path | None) -> Path:
    """Resolve an optional database path without modifying user input."""
    return Path(db_path) if db_path is not None else _default_db_path()


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a database connection and initialize its parent directory."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def _database(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection for one operation and always close it afterward."""
    connection = _connect(db_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Create the report database and reports table if they do not exist."""
    create_table_sql = """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL DEFAULT 'legacy',
            dataset_signature TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            event_start_time TEXT NOT NULL,
            event_end_time TEXT NOT NULL,
            duration_hours REAL NOT NULL,
            anomaly_count INTEGER NOT NULL,
            suspected_cause TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            report TEXT NOT NULL
        )
    """
    try:
        with _database(db_path) as connection:
            connection.execute(create_table_sql)
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(reports)")
            }
            if "user_id" not in columns:
                connection.execute(
                    "ALTER TABLE reports ADD COLUMN user_id TEXT NOT NULL DEFAULT 'legacy'"
                )
    except sqlite3.Error as exc:
        raise RuntimeError(f"初始化报告数据库失败: {exc}") from exc


def save_report(report_data: dict[str, Any], db_path: str | Path | None = None) -> int:
    """Save one report and return its generated database id."""
    if not isinstance(report_data, dict):
        raise TypeError("report_data 必须是字典。")

    required_fields = REPORT_COLUMNS[1:]
    missing_fields = [field for field in required_fields if field not in report_data]
    if missing_fields:
        raise ValueError(f"报告数据缺少字段: {', '.join(missing_fields)}")

    init_db(db_path)
    values = tuple(report_data[field] for field in required_fields)
    insert_sql = """
        INSERT INTO reports (
            user_id, dataset_signature, dataset_name, created_at,
            event_start_time, event_end_time, duration_hours,
            anomaly_count, suspected_cause, diagnosis, suggestion, report
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with _database(db_path) as connection:
            cursor = connection.execute(insert_sql, values)
            return int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise RuntimeError(f"保存报告失败: {exc}") from exc


def list_reports(
    user_id: str,
    dataset_signature: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return one user's reports newest first, optionally filtered by dataset."""
    init_db(db_path)
    try:
        with _database(db_path) as connection:
            if dataset_signature is None:
                cursor = connection.execute(
                    "SELECT * FROM reports WHERE user_id = ? "
                    "ORDER BY created_at DESC, id DESC",
                    (user_id,),
                )
            else:
                cursor = connection.execute(
                    "SELECT * FROM reports WHERE user_id = ? AND dataset_signature = ? "
                    "ORDER BY created_at DESC, id DESC",
                    (user_id, dataset_signature),
                )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise RuntimeError(f"读取报告历史失败: {exc}") from exc


def get_report(
    report_id: int,
    user_id: str,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return one report owned by the user, or None when it does not exist."""
    init_db(db_path)
    try:
        with _database(db_path) as connection:
            cursor = connection.execute(
                "SELECT * FROM reports WHERE id = ? AND user_id = ?",
                (report_id, user_id),
            )
            row = cursor.fetchone()
            return dict(row) if row is not None else None
    except sqlite3.Error as exc:
        raise RuntimeError(f"读取报告失败: {exc}") from exc


def delete_report(
    report_id: int,
    user_id: str,
    db_path: str | Path | None = None,
) -> bool:
    """Delete one report and return whether a record was removed."""
    init_db(db_path)
    try:
        with _database(db_path) as connection:
            cursor = connection.execute(
                "DELETE FROM reports WHERE id = ? AND user_id = ?",
                (report_id, user_id),
            )
            return cursor.rowcount > 0
    except sqlite3.Error as exc:
        raise RuntimeError(f"删除报告失败: {exc}") from exc
