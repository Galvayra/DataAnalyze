from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from analyze.common import (
    OUTPUT_DIR,
    age_to_target_band,
    build_bullet_list,
    build_kpi_cards,
    build_table,
    render_page,
    target_age_text_to_bands,
)
from build.action_db import DB_PATH


OUTPUT_PATH = OUTPUT_DIR / "shop_target_age_comparison.html"
ACTUAL_BAND_ORDER = [
    "10대",
    "20대 초반",
    "20대 중반",
    "20대 후반",
    "30대 초반",
    "30대 중반",
    "30대 후반",
    "40대 이상",
]


@dataclass
class ShopMetric:
    shop_name: str
    target_age_raw: str
    target_bands: list[str]
    actual_top_1_summary: str
    actual_top_2_summary: str
    actual_top_3_summary: str
    buyers: int
    orders: int
    revenue: int
    avg_order_value: float
    match_rate_pct: float


def fetch_rows() -> list[sqlite3.Row]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT
                s.shop_id,
                s.name AS shop_name,
                s.age AS target_age_raw,
                o.user_id,
                o.price,
                u.age AS buyer_age
            FROM order_info o
            JOIN shop_info s ON s.shop_id = o.shop_id
            LEFT JOIN user_info u ON u.user_id = o.user_id
            ORDER BY s.shop_id;
            """
        ).fetchall()
    finally:
        conn.close()


def compute_metrics() -> list[ShopMetric]:
    rows = fetch_rows()
    per_shop: dict[str, dict[str, object]] = {}

    for row in rows:
        shop_name = row["shop_name"]
        if shop_name not in per_shop:
            target_bands = target_age_text_to_bands(row["target_age_raw"])
            per_shop[shop_name] = {
                "target_age_raw": row["target_age_raw"] or "-",
                "target_bands": target_bands or ["미상"],
                "orders": 0,
                "revenue": 0,
                "matched_orders": 0,
                "buyer_ids": set(),
                "actual_buyer_sets": {
                    "10대": set(),
                    "20대 초반": set(),
                    "20대 중반": set(),
                    "20대 후반": set(),
                    "30대 초반": set(),
                    "30대 중반": set(),
                    "30대 후반": set(),
                    "40대 이상": set(),
                },
            }

        buyer_band = age_to_target_band(row["buyer_age"])
        if buyer_band == "미상":
            continue
        shop = per_shop[shop_name]
        shop["orders"] = int(shop["orders"]) + 1
        shop["revenue"] = int(shop["revenue"]) + row["price"]
        shop["buyer_ids"].add(row["user_id"])
        shop["actual_buyer_sets"][buyer_band].add(row["user_id"])

        target_bands = shop["target_bands"]
        if buyer_band in target_bands:
            shop["matched_orders"] = int(shop["matched_orders"]) + 1

    metrics: list[ShopMetric] = []
    for shop_name, payload in per_shop.items():
        actual_counts = {
            band: len(user_ids)
            for band, user_ids in payload["actual_buyer_sets"].items()
        }
        buyers = len(payload["buyer_ids"])
        ranked_actual = sorted(
            [(band, count) for band, count in actual_counts.items() if count > 0],
            key=lambda item: (-item[1], ACTUAL_BAND_ORDER.index(item[0])),
        )
        actual_summaries: list[str] = []
        current_rank = 0
        previous_count: int | None = None
        current_bands: list[str] = []
        current_total_count = 0
        for band, count in ranked_actual:
            if previous_count is None or count < previous_count:
                if current_bands and buyers:
                    pct = current_total_count * 100.0 / buyers
                    actual_summaries.append(f"{'/'.join(current_bands)} ({pct:.0f}%)")
                current_rank += 1
                previous_count = count
                if current_rank > 3:
                    break
                current_bands = [band]
                current_total_count = count
            else:
                current_bands.append(band)
                current_total_count += count
        if current_bands and len(actual_summaries) < 3 and buyers:
            pct = current_total_count * 100.0 / buyers
            actual_summaries.append(f"{'/'.join(current_bands)} ({pct:.0f}%)")
        while len(actual_summaries) < 3:
            actual_summaries.append("-")
        orders = int(payload["orders"])
        match_rate = int(payload["matched_orders"]) * 100.0 / orders if orders else 0.0
        avg_order_value = int(payload["revenue"]) / orders if orders else 0.0
        metrics.append(
            ShopMetric(
                shop_name=shop_name,
                target_age_raw=str(payload["target_age_raw"]),
                target_bands=list(payload["target_bands"]),
                actual_top_1_summary=actual_summaries[0],
                actual_top_2_summary=actual_summaries[1],
                actual_top_3_summary=actual_summaries[2],
                buyers=buyers,
                orders=orders,
                revenue=int(payload["revenue"]),
                avg_order_value=round(avg_order_value, 2),
                match_rate_pct=round(match_rate, 2),
            )
        )

    metrics.sort(key=lambda item: (item.match_rate_pct, -item.orders, item.shop_name))
    return metrics


def build_html(metrics: list[ShopMetric]) -> str:
    metrics_with_actual = [item for item in metrics if item.actual_top_1_summary != "-"]
    metrics_with_actual_min_orders = [
        item for item in metrics_with_actual if item.orders >= 10
    ]
    eligible_metrics = [
        item for item in sorted(metrics, key=lambda item: item.orders, reverse=True)
        if "미상" not in item.target_bands
        and item.actual_top_1_summary != "-"
        and item.orders >= 10
    ]
    lowest_match_metrics = sorted(
        eligible_metrics,
        key=lambda item: (item.match_rate_pct, -item.orders, item.shop_name),
    )
    unknown_target_metrics = [
        item for item in sorted(metrics, key=lambda item: (-item.orders, item.shop_name))
        if "미상" in item.target_bands and item.actual_top_1_summary != "-" and item.orders >= 10
    ]

    top_orders_shop = max(metrics_with_actual, key=lambda item: item.orders)
    lowest_match_shop = min(eligible_metrics, key=lambda item: (item.match_rate_pct, -item.orders, item.shop_name))
    top_avg_order_shop = max(
        metrics_with_actual_min_orders, key=lambda item: item.avg_order_value
    )

    kpis = build_kpi_cards(
        [
            ("주문 수 1위 쇼핑몰", f"{top_orders_shop.shop_name} ({top_orders_shop.orders:,}건)"),
            ("타깃 일치율이 가장 낮은 쇼핑몰", f"{lowest_match_shop.shop_name} ({lowest_match_shop.match_rate_pct:.2f}%)"),
            ("평균 주문금액 1위 쇼핑몰의 주 구매 연령", f"{top_avg_order_shop.shop_name} ({top_avg_order_shop.actual_top_1_summary})"),
        ]
    )

    sections = [
        build_bullet_list(
            "해석 포인트",
            [
                "타깃 일치율은 연령 정보가 확인된 주문만 기준으로, 실제 구매 연령대가 쇼핑몰이 설정한 타깃 연령대에 포함되는 비율입니다.",
                "10대와 40대 이상은 각각 하나의 구간으로 묶어 비교하고, 20~30대는 초반·중반·후반으로 세분화해 비교했습니다.",
                "실제 구매한 유저의 연령 정보가 없는 경우에는 이번 집계에서 제외했습니다.",
            ],
        ),
        build_table(
            "쇼핑몰별 타깃 연령 비교 표 - 일치율 낮은 순",
            "주문 수가 10회 이상인 쇼핑몰 중에서 타깃 연령과 실제 주 구매 연령이 모두 명확한 쇼핑몰을 대상으로 하며, 타깃 일치율이 낮은 순으로 전체를 보여줍니다.",
            ["쇼핑몰", "설정 타깃 연령", "주 구매 연령(1위)", "주 구매 연령(2위)", "주 구매 연령(3위)", "총 구매 수", "평균 주문 금액", "타깃 일치율"],
            [
                [
                    item.shop_name,
                    item.target_age_raw,
                    item.actual_top_1_summary,
                    item.actual_top_2_summary,
                    item.actual_top_3_summary,
                    f"{item.buyers:,}",
                    f"{item.avg_order_value:,.0f}",
                    f"{item.match_rate_pct:.2f}%",
                ]
                for item in lowest_match_metrics
            ],
        ),
        build_table(
            "연령대 설정 타깃이 미정인 쇼핑몰",
            "설정 타깃 연령이 미정이며 주문 수가 10회 이상인 쇼핑몰을 주문 수 내림차순으로 전체 표시합니다.",
            ["쇼핑몰", "주 구매 연령(1위)", "주 구매 연령(2위)", "주 구매 연령(3위)", "총 구매 수", "평균 주문 금액"],
            [
                [
                    item.shop_name,
                    item.actual_top_1_summary,
                    item.actual_top_2_summary,
                    item.actual_top_3_summary,
                    f"{item.buyers:,}",
                    f"{item.avg_order_value:,.0f}",
                ]
                for item in unknown_target_metrics
            ],
        ),
    ]

    return render_page(
        "5. 쇼핑몰 타깃 연령과 실제 구매 연령 비교",
        "쇼핑몰이 설정한 타깃 연령과 실제 구매 연령대를 비교해, 입점 쇼핑몰의 노출 전략과 타깃 조정 방향을 검토하기 위한 분석입니다.",
        kpis,
        sections,
    )


def main() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics()
    html = build_html(metrics)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved visualization to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
