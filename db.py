import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "grocery_bot.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS members (
                telegram_id   INTEGER PRIMARY KEY,
                name          TEXT    NOT NULL,
                household_id  TEXT    NOT NULL DEFAULT 'default',
                preferences   TEXT    DEFAULT '',
                joined_at     TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_preferences (
                telegram_id      INTEGER NOT NULL,
                item_keyword     TEXT    NOT NULL,
                zepto_product    TEXT    NOT NULL,
                last_used_at     TEXT    DEFAULT (datetime('now')),
                PRIMARY KEY (telegram_id, item_keyword)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS order_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id    INTEGER NOT NULL,
                summary        TEXT    NOT NULL,
                ordered_at     TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


def register_member(telegram_id: int, name: str, household_id: str = "default"):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO members (telegram_id, name, household_id)
            VALUES (?, ?, ?)
            ON CONFLICT(telegram_id) DO UPDATE SET name=excluded.name
        """, (telegram_id, name, household_id))
        conn.commit()


def get_member(telegram_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM members WHERE telegram_id = ?", (telegram_id,)
        ).fetchone()
        return dict(row) if row else None


def get_household_members(household_id: str = "default") -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM members WHERE household_id = ?", (household_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def set_preferences(telegram_id: int, preferences: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE members SET preferences = ? WHERE telegram_id = ?",
            (preferences, telegram_id)
        )
        conn.commit()


def get_product_pref(telegram_id: int, item_keyword: str) -> str | None:
    """Get the stored Zepto product name for a given item keyword."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT zepto_product FROM product_preferences "
            "WHERE telegram_id = ? AND item_keyword = ?",
            (telegram_id, item_keyword.lower().strip()),
        ).fetchone()
        return row["zepto_product"] if row else None


def set_product_pref(telegram_id: int, item_keyword: str, zepto_product: str):
    """Save or update the preferred Zepto product for an item keyword."""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO product_preferences (telegram_id, item_keyword, zepto_product, last_used_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(telegram_id, item_keyword)
            DO UPDATE SET zepto_product=excluded.zepto_product, last_used_at=datetime('now')
        """, (telegram_id, item_keyword.lower().strip(), zepto_product))
        conn.commit()


def get_all_product_prefs(telegram_id: int) -> dict[str, str]:
    """Get all product preferences for a user. Returns {item_keyword: zepto_product}."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT item_keyword, zepto_product FROM product_preferences WHERE telegram_id = ?",
            (telegram_id,),
        ).fetchall()
        return {row["item_keyword"]: row["zepto_product"] for row in rows}


def save_order(telegram_id: int, summary: str):
    """Save a placed order to history."""
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO order_history (telegram_id, summary) VALUES (?, ?)",
            (telegram_id, summary),
        )
        conn.commit()


def get_recent_orders(telegram_id: int, limit: int = 3) -> list[dict]:
    """Get the last N orders for a user."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT summary, ordered_at FROM order_history "
            "WHERE telegram_id = ? ORDER BY ordered_at DESC LIMIT ?",
            (telegram_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
