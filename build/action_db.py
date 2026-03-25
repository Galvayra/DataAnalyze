from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "action"
DB_PATH = PROJECT_ROOT / "sqlite" / "action_analysis.sqlite"


PATTERN_QUERIES: list[tuple[str, str]] = [
    (
        "테이블별 행 수",
        """
        SELECT 'user_info' AS table_name, COUNT(*) AS row_count FROM user_info
        UNION ALL
        SELECT 'user_event_logs', COUNT(*) FROM user_event_logs
        UNION ALL
        SELECT 'order_info', COUNT(*) FROM order_info
        UNION ALL
        SELECT 'goods_info', COUNT(*) FROM goods_info
        UNION ALL
        SELECT 'shop_info', COUNT(*) FROM shop_info;
        """,
    ),
    (
        "매출 개요",
        """
        SELECT
            COUNT(*) AS orders,
            COUNT(DISTINCT user_id) AS purchasing_users,
            SUM(price) AS revenue,
            ROUND(AVG(price), 2) AS avg_order_value
        FROM order_info;
        """,
    ),
    (
        "카테고리별 매출 TOP 10",
        """
        SELECT
            COALESCE(goods_category, '미분류') AS goods_category,
            COUNT(*) AS orders,
            SUM(price) AS revenue,
            ROUND(AVG(price), 2) AS avg_order_value
        FROM order_enriched
        GROUP BY 1
        ORDER BY revenue DESC
        LIMIT 10;
        """,
    ),
    (
        "카테고리별 구매 유저 TOP 10",
        """
        SELECT
            COALESCE(goods_category, '미분류') AS goods_category,
            COUNT(DISTINCT user_id) AS buyers,
            COUNT(*) AS orders
        FROM order_enriched
        GROUP BY 1
        ORDER BY buyers DESC, orders DESC
        LIMIT 10;
        """,
    ),
    (
        "연령별 구매 전환",
        """
        WITH visitors AS (
            SELECT age, COUNT(DISTINCT user_id) AS visitors
            FROM user_info
            GROUP BY age
        ),
        buyers AS (
            SELECT u.age, COUNT(DISTINCT o.user_id) AS buyers
            FROM order_info o
            JOIN user_info u ON u.user_id = o.user_id
            GROUP BY u.age
        )
        SELECT
            v.age,
            v.visitors,
            COALESCE(b.buyers, 0) AS buyers,
            ROUND(COALESCE(b.buyers, 0) * 100.0 / v.visitors, 2) AS buyer_rate_pct
        FROM visitors v
        LEFT JOIN buyers b ON b.age = v.age
        ORDER BY buyer_rate_pct DESC, v.age;
        """,
    ),
    (
        "OS별 구매 성과",
        """
        WITH visitors AS (
            SELECT os, COUNT(DISTINCT user_id) AS visitors
            FROM user_info
            GROUP BY os
        ),
        orders AS (
            SELECT u.os, COUNT(*) AS orders, COUNT(DISTINCT o.user_id) AS buyers, SUM(o.price) AS revenue
            FROM order_info o
            JOIN user_info u ON u.user_id = o.user_id
            GROUP BY u.os
        )
        SELECT
            v.os,
            v.visitors,
            COALESCE(o.buyers, 0) AS buyers,
            COALESCE(o.orders, 0) AS orders,
            COALESCE(o.revenue, 0) AS revenue,
            ROUND(COALESCE(o.buyers, 0) * 100.0 / v.visitors, 2) AS buyer_rate_pct,
            ROUND(COALESCE(o.revenue, 0) * 1.0 / NULLIF(o.orders, 0), 2) AS avg_order_value
        FROM visitors v
        LEFT JOIN orders o ON o.os = v.os
        ORDER BY revenue DESC;
        """,
    ),
    (
        "시간대별 주문",
        """
        SELECT
            strftime('%H', timestamp) AS hour,
            COUNT(*) AS orders,
            SUM(price) AS revenue
        FROM order_info
        GROUP BY hour
        ORDER BY hour;
        """,
    ),
    (
        "검색 키워드 TOP 15",
        """
        SELECT
            search_keyword,
            COUNT(*) AS search_events,
            COUNT(DISTINCT user_id) AS users
        FROM event_enriched
        WHERE search_keyword IS NOT NULL
        GROUP BY search_keyword
        ORDER BY search_events DESC
        LIMIT 15;
        """,
    ),
    (
        "쇼핑몰별 매출 TOP 10",
        """
        SELECT
            COALESCE(shop_name, '미상') AS shop_name,
            COUNT(*) AS orders,
            SUM(price) AS revenue,
            ROUND(AVG(price), 2) AS avg_order_value
        FROM order_enriched
        GROUP BY 1
        ORDER BY revenue DESC
        LIMIT 10;
        """,
    ),
    (
        "이미지 타입별 판매 성과",
        """
        SELECT
            COALESCE(image_type, '미상') AS image_type,
            COUNT(*) AS orders,
            SUM(price) AS revenue,
            ROUND(AVG(price), 2) AS avg_order_value
        FROM order_enriched
        GROUP BY 1
        ORDER BY revenue DESC;
        """,
    ),
]


def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def recreate_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP VIEW IF EXISTS order_enriched;
        DROP VIEW IF EXISTS event_enriched;

        DROP TABLE IF EXISTS user_info;
        DROP TABLE IF EXISTS user_event_logs;
        DROP TABLE IF EXISTS order_info;
        DROP TABLE IF EXISTS goods_info;
        DROP TABLE IF EXISTS shop_info;

        CREATE TABLE user_info (
            user_id TEXT PRIMARY KEY,
            os TEXT,
            age INTEGER
        );

        CREATE TABLE user_event_logs (
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            event_origin TEXT,
            event_name TEXT,
            event_goods_id INTEGER,
            event_shop_id INTEGER
        );

        CREATE TABLE order_info (
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            goods_id INTEGER NOT NULL,
            shop_id INTEGER NOT NULL,
            price INTEGER NOT NULL
        );

        CREATE TABLE goods_info (
            goods_id INTEGER PRIMARY KEY,
            timestamp TEXT,
            shop_id INTEGER,
            category TEXT,
            price INTEGER,
            image_type TEXT,
            image_width INTEGER,
            image_height INTEGER
        );

        CREATE TABLE shop_info (
            shop_id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            age TEXT,
            style TEXT
        );
        """
    )


def load_csv(
    conn: sqlite3.Connection,
    table_name: str,
    csv_path: Path,
    columns: Sequence[str],
    int_columns: Iterable[str] = (),
) -> None:
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    int_columns = set(int_columns)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for raw_row in reader:
            parsed_row = []
            for column in columns:
                value = raw_row[column].strip()
                if value == "":
                    parsed_row.append(None)
                elif column in int_columns:
                    parsed_row.append(int(value))
                else:
                    parsed_row.append(value)
            rows.append(tuple(parsed_row))

    conn.executemany(insert_sql, rows)


def create_indexes_and_views(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_user_event_logs_user_id ON user_event_logs(user_id);
        CREATE INDEX idx_user_event_logs_goods_id ON user_event_logs(event_goods_id);
        CREATE INDEX idx_user_event_logs_shop_id ON user_event_logs(event_shop_id);
        CREATE INDEX idx_user_event_logs_timestamp ON user_event_logs(timestamp);

        CREATE INDEX idx_order_info_user_id ON order_info(user_id);
        CREATE INDEX idx_order_info_goods_id ON order_info(goods_id);
        CREATE INDEX idx_order_info_shop_id ON order_info(shop_id);
        CREATE INDEX idx_order_info_timestamp ON order_info(timestamp);

        CREATE INDEX idx_goods_info_shop_id ON goods_info(shop_id);

        CREATE VIEW event_enriched AS
        SELECT
            e.timestamp,
            e.user_id,
            u.os,
            u.age AS user_age,
            e.event_origin,
            e.event_name,
            e.event_goods_id,
            g.category AS goods_category,
            g.price AS goods_price,
            e.event_shop_id,
            s.name AS shop_name,
            s.style AS shop_style,
            CASE
                WHEN e.event_origin LIKE 'goods_search_result/%'
                THEN substr(e.event_origin, length('goods_search_result/') + 1)
                ELSE NULL
            END AS search_keyword
        FROM user_event_logs e
        LEFT JOIN user_info u ON u.user_id = e.user_id
        LEFT JOIN goods_info g ON g.goods_id = e.event_goods_id
        LEFT JOIN shop_info s ON s.shop_id = e.event_shop_id;

        CREATE VIEW order_enriched AS
        SELECT
            o.timestamp,
            o.user_id,
            u.os,
            u.age AS user_age,
            o.goods_id,
            g.category AS goods_category,
            g.image_type,
            g.image_width,
            g.image_height,
            o.shop_id,
            s.name AS shop_name,
            s.style AS shop_style,
            o.price
        FROM order_info o
        LEFT JOIN user_info u ON u.user_id = o.user_id
        LEFT JOIN goods_info g ON g.goods_id = o.goods_id
        LEFT JOIN shop_info s ON s.shop_id = o.shop_id;
        """
    )


def bootstrap_database() -> Path:
    conn = connect_db()
    try:
        recreate_tables(conn)
        load_csv(
            conn,
            "user_info",
            DATA_DIR / "user_info.csv",
            ["user_id", "os", "age"],
            int_columns=["age"],
        )
        load_csv(
            conn,
            "user_event_logs",
            DATA_DIR / "user_event_logs.csv",
            [
                "timestamp",
                "user_id",
                "event_origin",
                "event_name",
                "event_goods_id",
                "event_shop_id",
            ],
            int_columns=["event_goods_id", "event_shop_id"],
        )
        load_csv(
            conn,
            "order_info",
            DATA_DIR / "order_info.csv",
            ["timestamp", "user_id", "goods_id", "shop_id", "price"],
            int_columns=["goods_id", "shop_id", "price"],
        )
        load_csv(
            conn,
            "goods_info",
            DATA_DIR / "goods_info.csv",
            [
                "goods_id",
                "timestamp",
                "shop_id",
                "category",
                "price",
                "image_type",
                "image_width",
                "image_height",
            ],
            int_columns=["goods_id", "shop_id", "price", "image_width", "image_height"],
        )
        load_csv(
            conn,
            "shop_info",
            DATA_DIR / "shop_info.csv",
            ["shop_id", "name", "category", "age", "style"],
            int_columns=["shop_id"],
        )
        create_indexes_and_views(conn)
        conn.commit()
        return DB_PATH
    finally:
        conn.close()


def fetch_query_results(title: str, sql: str) -> tuple[str, list[str], list[sqlite3.Row]]:
    conn = connect_db()
    try:
        rows = conn.execute(sql).fetchall()
        headers = list(rows[0].keys()) if rows else []
        return title, headers, rows
    finally:
        conn.close()
