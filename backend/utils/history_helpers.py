# utils/history_helpers.py
import time
import typing as t
import sqlite3


def append_message(session_id: str, role: str, content: str, db_conn: sqlite3.Connection):
    ts = time.time()
    db_conn.execute(
        "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (session_id, role, content, ts),
    )
    db_conn.commit()


def get_history(session_id: str, db_conn: sqlite3.Connection, limit: int = 100) -> t.List[dict]:
    cur = db_conn.execute(
        "SELECT role, content, ts FROM messages WHERE session_id = ? ORDER BY ts ASC",
        (session_id,),
    )
    rows = cur.fetchall()
    result = [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]
    return result[-limit:]


def get_recent_messages(session_id: str, db_conn: sqlite3.Connection, n: int = 6) -> t.List[dict]:
    all_msgs = get_history(session_id, db_conn, limit=n)
    return all_msgs[-n:]


def clear_history(session_id: str, db_conn: sqlite3.Connection):
    db_conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    db_conn.commit()
