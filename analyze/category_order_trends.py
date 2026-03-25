from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from analyze.common import (
    OUTPUT_DIR,
    build_bar_chart,
    build_bullet_list,
    build_grouped_bar_chart,
    build_heatmap_table,
    build_kpi_cards,
    build_table,
    pearson_correlation,
    price_to_band,
    render_page,
)
from build.action_db import DB_PATH


OUTPUT_PATH = OUTPUT_DIR / "category_order_trends.html"
PRICE_BANDS = ["1만원 미만", "1만원대", "2만원대", "3~4만원대", "5만원 이상"]


@dataclass
class CategoryMetric:
    category: str
    interest_score: float
    enter_browser_events: int
    active_users: int
    orders: int
    buyers: int
    purchase_conversions: int
    revenue: int
    avg_order_value: float
    conversion_rate_pct: float


def fetch_price_band_matrix(categories: list[str]) -> dict[str, dict[str, int]]:
    if not categories:
        return {}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    placeholders = ", ".join("?" for _ in categories)
    try:
        rows = conn.execute(
            f"""
            SELECT
                g.category AS category,
                CASE
                    WHEN o.price < 10000 THEN '1만원 미만'
                    WHEN o.price < 20000 THEN '1만원대'
                    WHEN o.price < 30000 THEN '2만원대'
                    WHEN o.price < 50000 THEN '3~4만원대'
                    ELSE '5만원 이상'
                END AS price_band,
                COUNT(*) AS orders
            FROM order_info o
            JOIN goods_info g ON g.goods_id = o.goods_id
            WHERE g.category IN ({placeholders})
            GROUP BY g.category, price_band
            ORDER BY g.category, price_band;
            """,
            categories,
        ).fetchall()
    finally:
        conn.close()

    matrix = {category: {band: 0 for band in PRICE_BANDS} for category in categories}
    for row in rows:
        matrix[row["category"]][row["price_band"]] = row["orders"]
    return matrix


def _fetch_category_metrics_sqlite_compatible() -> list[CategoryMetric]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            WITH activity AS (
                SELECT
                    g.category AS category,
                    SUM(
                        CASE
                            WHEN e.event_name = 'enter_browser' THEN 0.1
                            WHEN e.event_name = 'add_my_goods' THEN 1.0
                            WHEN e.event_name = 'remove_my_goods' THEN -1.0
                            ELSE 0.0
                        END
                    ) AS interest_score,
                    COUNT(DISTINCT e.user_id) AS active_users
                FROM user_event_logs e
                JOIN goods_info g ON g.goods_id = e.event_goods_id
                WHERE e.event_name IN ('enter_browser', 'add_my_goods', 'remove_my_goods')
                GROUP BY g.category
            ),
            enter_browser_activity AS (
                SELECT
                    g.category AS category,
                    COUNT(*) AS enter_browser_events
                FROM user_event_logs e
                JOIN goods_info g ON g.goods_id = e.event_goods_id
                WHERE e.event_name = 'enter_browser'
                  AND e.event_goods_id IS NOT NULL
                  AND e.event_shop_id IS NOT NULL
                GROUP BY g.category
            ),
            orders AS (
                SELECT
                    g.category AS category,
                    COUNT(*) AS orders,
                    COUNT(DISTINCT o.user_id) AS buyers,
                    SUM(o.price) AS revenue,
                    AVG(o.price) AS avg_order_value
                FROM order_info o
                JOIN goods_info g ON g.goods_id = o.goods_id
                GROUP BY g.category
            ),
            matched_orders AS (
                SELECT
                    o.rowid AS order_rowid,
                    g.category AS category
                FROM order_info o
                JOIN goods_info g ON g.goods_id = o.goods_id
                WHERE EXISTS (
                    SELECT 1
                    FROM user_event_logs e
                    WHERE e.event_name = 'enter_browser'
                      AND e.user_id = o.user_id
                      AND e.event_goods_id = o.goods_id
                      AND e.event_shop_id = o.shop_id
                      AND e.timestamp < o.timestamp
                      AND e.rowid = (
                          SELECT e2.rowid
                          FROM user_event_logs e2
                          WHERE e2.event_name = 'enter_browser'
                            AND e2.user_id = o.user_id
                            AND e2.event_goods_id = o.goods_id
                            AND e2.event_shop_id = o.shop_id
                            AND e2.timestamp < o.timestamp
                          ORDER BY e2.timestamp DESC
                          LIMIT 1
                      )
                )
            ),
            conversions AS (
                SELECT
                    category,
                    COUNT(*) AS purchase_conversions
                FROM matched_orders
                GROUP BY category
            ),
            categories AS (
                SELECT category FROM activity
                UNION
                SELECT category FROM enter_browser_activity
                UNION
                SELECT category FROM orders
                UNION
                SELECT category FROM conversions
            )
            SELECT
                c.category AS category,
                ROUND(COALESCE(a.interest_score, 0), 1) AS interest_score,
                COALESCE(eba.enter_browser_events, 0) AS enter_browser_events,
                COALESCE(a.active_users, 0) AS active_users,
                COALESCE(o.orders, 0) AS orders,
                COALESCE(o.buyers, 0) AS buyers,
                COALESCE(cv.purchase_conversions, 0) AS purchase_conversions,
                COALESCE(o.revenue, 0) AS revenue,
                ROUND(COALESCE(o.avg_order_value, 0), 2) AS avg_order_value,
                ROUND(COALESCE(cv.purchase_conversions, 0) * 100.0 / NULLIF(eba.enter_browser_events, 0), 2) AS conversion_rate_pct
            FROM categories c
            LEFT JOIN activity a ON a.category = c.category
            LEFT JOIN enter_browser_activity eba ON eba.category = c.category
            LEFT JOIN orders o ON o.category = c.category
            LEFT JOIN conversions cv ON cv.category = c.category
            ORDER BY revenue DESC, orders DESC;
            """
        ).fetchall()
        return [
            CategoryMetric(
                category=row["category"],
                interest_score=row["interest_score"],
                enter_browser_events=row["enter_browser_events"],
                active_users=row["active_users"],
                orders=row["orders"],
                buyers=row["buyers"],
                purchase_conversions=row["purchase_conversions"],
                revenue=row["revenue"],
                avg_order_value=row["avg_order_value"],
                conversion_rate_pct=row["conversion_rate_pct"] or 0.0,
            )
            for row in rows
        ]
    finally:
        conn.close()


def fetch_category_metrics() -> list[CategoryMetric]:
    return _fetch_category_metrics_sqlite_compatible()


def build_html(metrics: list[CategoryMetric]) -> str:
    top_revenue = metrics[:8]
    top_interest = sorted(metrics, key=lambda item: item.interest_score, reverse=True)[:8]
    metrics_by_interest = sorted(metrics, key=lambda item: item.interest_score, reverse=True)
    top_activity = max(metrics, key=lambda item: item.interest_score)
    top_sales = max(metrics, key=lambda item: item.revenue)
    top_conversion = max(metrics, key=lambda item: item.conversion_rate_pct)
    corr = pearson_correlation(
        [float(item.interest_score) for item in metrics],
        [float(item.revenue) for item in metrics],
    )

    activity_top5 = {item.category for item in sorted(metrics, key=lambda item: item.interest_score, reverse=True)[:5]}
    revenue_top5 = {item.category for item in sorted(metrics, key=lambda item: item.revenue, reverse=True)[:5]}
    overlap = len(activity_top5 & revenue_top5)

    price_matrix_data = fetch_price_band_matrix([item.category for item in top_revenue])
    price_matrix = [
        [float(price_matrix_data[item.category][band]) for band in PRICE_BANDS]
        for item in top_revenue
    ]

    kpis = build_kpi_cards(
        [
            ("관심도 1위 카테고리", f"{top_activity.category} ({top_activity.interest_score:,.1f}점)"),
            ("매출 1위 카테고리", f"{top_sales.category} ({top_sales.revenue:,}원)"),
            ("최고 전환 카테고리", f"{top_conversion.category} ({top_conversion.conversion_rate_pct:.2f}%)"),
        ]
    )

    sections = [
        build_bullet_list(
            "해석 포인트",
            [
                f"관심도 상위 5개와 매출 상위 5개의 겹침은 {overlap}개 카테고리입니다.",
                "카테고리 관심도는 `enter_browser=0.1`, `add_my_goods=1`, `remove_my_goods=-1` 가중치를 적용한 점수입니다.",
                "구매전환은 주문 1건당 가장 가까운 직전 상품 조회형 `enter_browser` 1건만 연결한 기준입니다.",
                "구매전환율은 카테고리별 상품 조회형 `enter_browser` 수 대비 구매전환 주문 수 비율입니다.",
                "가격대 매트릭스를 보면 카테고리별로 주문이 몰리는 대표 가격 구간을 확인할 수 있습니다.",
            ],
        ),
        build_grouped_bar_chart(
            "카테고리별 관심도와 주문 수",
            "카테고리 관심도 점수 상위 카테고리 기준으로 관심도 점수와 주문 수를 함께 비교합니다.",
            [item.category for item in top_interest],
            [
                ("관심도 점수", "#4f46e5", [float(item.interest_score) for item in top_interest]),
                ("주문 수", "#059669", [float(item.orders) for item in top_interest]),
            ],
        ),
        build_bar_chart(
            "카테고리별 매출",
            "실제 매출 기여도가 큰 카테고리를 비교합니다.",
            [item.category for item in top_revenue],
            [float(item.revenue) for item in top_revenue],
            "#0f766e",
        ),
        build_bar_chart(
            "카테고리별 구매전환율",
            "카테고리별 상품 조회형 enter_browser 수 대비 구매전환 주문 수 비율입니다.",
            [item.category for item in top_revenue],
            [item.conversion_rate_pct for item in top_revenue],
            "#ea580c",
            "%",
            2,
        ),
        build_heatmap_table(
            "카테고리별 가격대 주문 분포",
            "매출 상위 카테고리에서 어느 가격대 주문이 많은지 확인합니다.",
            PRICE_BANDS,
            [item.category for item in top_revenue],
            price_matrix,
        ),
        build_table(
            "카테고리 요약 표",
            "관심도 점수, 상품 조회 수, 구매전환 주문 수, 주문, 매출, 평균 주문 금액을 함께 봅니다.",
            ["카테고리", "관심도 점수", "상품 조회 수", "구매전환 수", "주문 수", "구매전환율", "매출", "객단가"],
            [
                [
                    item.category,
                    f"{item.interest_score:,.1f}",
                    f"{item.enter_browser_events:,}",
                    f"{item.purchase_conversions:,}",
                    f"{item.orders:,}",
                    f"{item.conversion_rate_pct:.2f}%",
                    f"{item.revenue:,}",
                    f"{item.avg_order_value:,.0f}",
                ]
                for item in metrics_by_interest[:12]
            ],
        ),
    ]

    return render_page(
        "2. 카테고리별 주문 추이",
        "카테고리별 관심도를 정의하여 주문 추이를 분석한 결과입니다.",
        kpis,
        sections,
    )


def main() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = fetch_category_metrics()
    html = build_html(metrics)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved visualization to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
