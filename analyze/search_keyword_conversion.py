from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from analyze.common import (
    OUTPUT_DIR,
    build_bar_chart,
    build_bullet_list,
    build_grouped_bar_chart,
    build_kpi_cards,
    build_table,
    render_page,
)
from build.action_db import DB_PATH


OUTPUT_PATH = OUTPUT_DIR / "search_keyword_conversion.html"


@dataclass
class KeywordMetric:
    keyword: str
    search_events: int
    representative_category: str
    category_avg_price: float
    conversion_rate_pct: float
    converted_orders: int
    converted_revenue: int


def fetch_keyword_metrics() -> list[KeywordMetric]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            WITH raw_searches AS (
                SELECT
                    rowid AS event_rowid,
                    user_id,
                    event_goods_id,
                    event_shop_id,
                    timestamp,
                    substr(event_origin, length('goods_search_result/') + 1) AS keyword
                FROM user_event_logs
                WHERE event_name = 'enter_browser'
                  AND event_origin LIKE 'goods_search_result/%'
                  AND event_goods_id IS NOT NULL
                  AND event_shop_id IS NOT NULL
            ),
            search_summary AS (
                SELECT
                    keyword,
                    COUNT(*) AS search_events
                FROM raw_searches
                GROUP BY keyword
            ),
            keyword_category_counts AS (
                SELECT
                    rs.keyword,
                    g.category,
                    COUNT(*) AS category_events
                FROM raw_searches rs
                JOIN goods_info g ON g.goods_id = rs.event_goods_id
                GROUP BY rs.keyword, g.category
            ),
            keyword_representative_category AS (
                SELECT
                    keyword,
                    category AS representative_category
                FROM (
                    SELECT
                        keyword,
                        category,
                        category_events,
                        ROW_NUMBER() OVER (
                            PARTITION BY keyword
                            ORDER BY category_events DESC, category
                        ) AS rn
                    FROM keyword_category_counts
                )
                WHERE rn = 1
            ),
            category_avg_prices AS (
                SELECT
                    category,
                    AVG(price) AS category_avg_price
                FROM goods_info
                WHERE price IS NOT NULL
                GROUP BY category
            ),
            matched_orders AS (
                SELECT
                    rs.keyword,
                    COUNT(*) AS converted_orders,
                    COALESCE(SUM(o.price), 0) AS converted_revenue
                FROM order_info o
                JOIN raw_searches rs
                  ON rs.event_rowid = (
                      SELECT e.rowid
                      FROM user_event_logs e
                      WHERE e.event_name = 'enter_browser'
                        AND e.event_origin LIKE 'goods_search_result/%'
                        AND e.user_id = o.user_id
                        AND e.event_goods_id = o.goods_id
                        AND e.event_shop_id = o.shop_id
                        AND e.timestamp < o.timestamp
                      ORDER BY e.timestamp DESC
                      LIMIT 1
                  )
                GROUP BY rs.keyword
            )
            SELECT
                ss.keyword,
                ss.search_events,
                COALESCE(krc.representative_category, '-') AS representative_category,
                ROUND(COALESCE(cap.category_avg_price, 0), 2) AS category_avg_price,
                COALESCE(mo.converted_orders, 0) AS converted_orders,
                ROUND(COALESCE(mo.converted_orders, 0) * 100.0 / NULLIF(ss.search_events, 0), 2) AS conversion_rate_pct,
                COALESCE(mo.converted_revenue, 0) AS converted_revenue
            FROM search_summary ss
            LEFT JOIN keyword_representative_category krc ON krc.keyword = ss.keyword
            LEFT JOIN category_avg_prices cap ON cap.category = krc.representative_category
            LEFT JOIN matched_orders mo ON mo.keyword = ss.keyword
            ORDER BY ss.search_events DESC, ss.keyword;
            """
        ).fetchall()
        return [
            KeywordMetric(
                keyword=row["keyword"],
                search_events=row["search_events"],
                representative_category=row["representative_category"],
                category_avg_price=row["category_avg_price"],
                conversion_rate_pct=row["conversion_rate_pct"] or 0.0,
                converted_orders=row["converted_orders"],
                converted_revenue=row["converted_revenue"],
            )
            for row in rows
        ]
    finally:
        conn.close()


def build_html(metrics: list[KeywordMetric]) -> str:
    top_search = metrics[:12]
    eligible_conversion = [item for item in metrics if item.search_events >= 20]
    top_conversion = sorted(eligible_conversion, key=lambda item: item.conversion_rate_pct, reverse=True)[:12]

    top_search_keyword = top_search[0]
    top_conversion_keyword = top_conversion[0]
    avg_conversion = sum(item.conversion_rate_pct for item in eligible_conversion) / max(len(eligible_conversion), 1)
    high_search_low_conversion = [
        item for item in eligible_conversion if item.search_events >= top_search_keyword.search_events * 0.25 and item.conversion_rate_pct < avg_conversion
    ]
    high_search_low_conversion = sorted(
        high_search_low_conversion,
        key=lambda item: (item.search_events, -item.conversion_rate_pct),
        reverse=True,
    )[:12]

    kpis = build_kpi_cards(
        [
            ("검색 이벤트 수 1위 키워드", f"{top_search_keyword.keyword} ({top_search_keyword.search_events:,}건)"),
            ("전환율 1위 키워드", f"{top_conversion_keyword.keyword} ({top_conversion_keyword.conversion_rate_pct:.2f}%)"),
            ("평균 구매전환율", f"{avg_conversion:.2f}%"),
        ]
    )

    sections = [
        build_bullet_list(
            "해석 포인트",
            [
                "특정 상품을 검색(`goods_search_result`)하고 해당 상품의 웹페이지로 진입(`enter_browser`)한 이벤트를 검색 이벤트라고 정의했습니다.",
                "구매 전환은 검색 이벤트 발생 이후 가장 가까이 해당 상품을 주문한 1건을 기준으로 정의했습니다.",
                    "따라서 구매 전환율은 검색 이벤트 수 대비 실제 주문으로 연결된 주문 수의 비율입니다.",
                    "검색량이 많은데 구매 전환율이 낮은 키워드는 상품 구성, 가격 경쟁력, 검색 결과 노출 품질을 점검할 후보로 볼 수 있습니다.",
            ],
        ),
        build_grouped_bar_chart(
            "검색 키워드별 검색 이벤트와 구매 전환",
            "검색 횟수가 많은 키워드에서 실제 구매 전환이 얼마나 따라오는지 비교합니다.",
            [item.keyword for item in top_search],
            [
                ("검색 이벤트", "#4f46e5", [float(item.search_events) for item in top_search]),
                ("구매 전환", "#059669", [float(item.converted_orders) for item in top_search]),
            ],
        ),
        build_bar_chart(
            "검색 키워드별 구매 전환율",
            "검색 이벤트 수 20건 이상 키워드 중 실제 구매 전환율 상위 키워드를 비교합니다.",
            [item.keyword for item in top_conversion],
            [item.conversion_rate_pct for item in top_conversion],
            "#ea580c",
            "%",
            2,
        ),
        build_table(
            "검색량 대비 저전환 키워드",
            "검색 이벤트가 많지만 평균 대비 전환율이 낮은 키워드입니다. 개선 우선순위 후보로 볼 수 있습니다.",
            ["키워드", "대표 카테고리", "카테고리 평균 가격", "검색 이벤트", "구매 전환", "구매 전환율", "구매 전환 매출"],
            [
                [
                    item.keyword,
                    item.representative_category,
                    f"{item.category_avg_price:,.0f}",
                    f"{item.search_events:,}",
                    f"{item.converted_orders:,}",
                    f"{item.conversion_rate_pct:.2f}%",
                    f"{item.converted_revenue:,}",
                ]
                for item in high_search_low_conversion
            ],
        ),
        build_table(
            "검색 키워드 요약 표",
            "검색량 상위 키워드 기준으로 실제 구매 전환과 후속 매출을 함께 확인합니다.",
            ["키워드", "대표 카테고리", "카테고리 평균 가격", "검색 이벤트", "구매 전환", "구매 전환율", "구매 전환 매출"],
            [
                [
                    item.keyword,
                    item.representative_category,
                    f"{item.category_avg_price:,.0f}",
                    f"{item.search_events:,}",
                    f"{item.converted_orders:,}",
                    f"{item.conversion_rate_pct:.2f}%",
                    f"{item.converted_revenue:,}",
                ]
                for item in top_search
            ],
        ),
    ]

    return render_page(
        "4. 검색 키워드와 구매 전환",
        "검색 키워드별 수요와 실제 구매 전환을 분석한 결과입니다.",
        kpis,
        sections,
    )


def main() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = fetch_keyword_metrics()
    html = build_html(metrics)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved visualization to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
