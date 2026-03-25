from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from analyze.common import (
    OUTPUT_DIR,
    age_to_target_band,
    build_bar_chart,
    build_bullet_list,
    build_heatmap_table,
    build_kpi_cards,
    build_table,
    price_to_band,
    render_page,
)
from build.action_db import DB_PATH


OUTPUT_PATH = OUTPUT_DIR / "age_order_trends.html"
AGE_BANDS = [
    "10대",
    "20대 초반",
    "20대 중반",
    "20대 후반",
    "30대 초반",
    "30대 중반",
    "30대 후반",
    "40대 이상",
]
HOURS = [f"{hour:02d}" for hour in range(24)]


@dataclass
class AgeMetric:
    age_band: str
    visitors: int
    active_users: int
    buyers: int
    orders: int
    revenue: int
    buyer_rate_pct: float
    avg_order_value: float
    peak_activity_hour: str
    peak_order_hour: str
    top_category_1: str
    top_category_2: str
    top_category_3: str
    top_price_band: str


@dataclass
class SubscriberDistribution:
    label: str
    count: int


def fetch_base_rows() -> tuple[list[sqlite3.Row], list[sqlite3.Row], list[sqlite3.Row]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        users = conn.execute("SELECT user_id, age FROM user_info").fetchall()
        events = conn.execute(
            """
            SELECT
                e.user_id,
                e.timestamp,
                e.event_name,
                e.event_goods_id,
                g.category
            FROM user_event_logs e
            LEFT JOIN goods_info g ON g.goods_id = e.event_goods_id
            """
        ).fetchall()
        orders = conn.execute(
            """
            SELECT o.user_id, o.timestamp, o.price, g.category
            FROM order_info o
            LEFT JOIN goods_info g ON g.goods_id = o.goods_id
            """
        ).fetchall()
        return users, events, orders
    finally:
        conn.close()


def compute_metrics() -> tuple[list[AgeMetric], list[list[float]], list[SubscriberDistribution]]:
    users, events, orders = fetch_base_rows()
    user_band = {
        row["user_id"]: age_to_target_band(row["age"])
        for row in users
        if age_to_target_band(row["age"]) in AGE_BANDS
    }
    subscriber_distribution_counts = {
        "10대": 0,
        "20대 초반": 0,
        "20대 중반": 0,
        "20대 후반": 0,
        "30대 초반": 0,
        "30대 중반": 0,
        "30대 후반": 0,
        "40대 이상": 0,
        "미정": 0,
    }

    visitor_counts = {band: 0 for band in AGE_BANDS}
    active_user_sets = {band: set() for band in AGE_BANDS}
    buyer_user_sets = {band: set() for band in AGE_BANDS}
    order_counts = {band: 0 for band in AGE_BANDS}
    revenue = {band: 0 for band in AGE_BANDS}
    activity_hours = {band: {hour: 0 for hour in HOURS} for band in AGE_BANDS}
    order_hours = {band: {hour: 0 for hour in HOURS} for band in AGE_BANDS}
    category_counts: dict[str, dict[str, float]] = {band: {} for band in AGE_BANDS}
    price_band_counts: dict[str, dict[str, int]] = {band: {} for band in AGE_BANDS}

    for row in users:
        distribution_band = age_to_target_band(row["age"])
        if distribution_band in AGE_BANDS:
            subscriber_distribution_counts[distribution_band] += 1
        else:
            subscriber_distribution_counts["미정"] += 1
        band = user_band.get(row["user_id"])
        if band:
            visitor_counts[band] += 1

    for row in events:
        band = user_band.get(row["user_id"])
        if not band:
            continue
        hour = row["timestamp"][11:13]
        active_user_sets[band].add(row["user_id"])
        activity_hours[band][hour] += 1
        if row["event_goods_id"] is not None and row["event_name"] in {
            "enter_browser",
            "add_my_goods",
            "remove_my_goods",
        }:
            category = row["category"] or "미분류"
            weight = 0.0
            if row["event_name"] == "enter_browser":
                weight = 0.1
            elif row["event_name"] == "add_my_goods":
                weight = 1.0
            elif row["event_name"] == "remove_my_goods":
                weight = -1.0
            category_counts[band][category] = category_counts[band].get(category, 0.0) + weight

    for row in orders:
        band = user_band.get(row["user_id"])
        if not band:
            continue
        hour = row["timestamp"][11:13]
        buyer_user_sets[band].add(row["user_id"])
        order_counts[band] += 1
        revenue[band] += row["price"]
        order_hours[band][hour] += 1
        price_band = price_to_band(row["price"])
        price_band_counts[band][price_band] = price_band_counts[band].get(price_band, 0) + 1

    metrics: list[AgeMetric] = []
    heatmap_matrix: list[list[float]] = []
    for band in AGE_BANDS:
        active_users = len(active_user_sets[band])
        buyers = len(buyer_user_sets[band])
        avg_order_value = revenue[band] / order_counts[band] if order_counts[band] else 0.0
        buyer_rate = buyers * 100.0 / visitor_counts[band] if visitor_counts[band] else 0.0
        peak_activity_hour = max(activity_hours[band], key=activity_hours[band].get)
        peak_order_hour = max(order_hours[band], key=order_hours[band].get)
        ranked_categories = sorted(
            category_counts[band].items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_category_summaries = [
            f"{category} ({score:,.1f})"
            for category, score in ranked_categories[:3]
        ]
        while len(top_category_summaries) < 3:
            top_category_summaries.append("-")
        top_price_band = max(price_band_counts[band], key=price_band_counts[band].get) if price_band_counts[band] else "-"

        metrics.append(
            AgeMetric(
                age_band=band,
                visitors=visitor_counts[band],
                active_users=active_users,
                buyers=buyers,
                orders=order_counts[band],
                revenue=revenue[band],
                buyer_rate_pct=round(buyer_rate, 2),
                avg_order_value=round(avg_order_value, 2),
                peak_activity_hour=peak_activity_hour,
                peak_order_hour=peak_order_hour,
                top_category_1=top_category_summaries[0],
                top_category_2=top_category_summaries[1],
                top_category_3=top_category_summaries[2],
                top_price_band=top_price_band,
            )
        )
        heatmap_matrix.append([float(order_hours[band][hour]) for hour in HOURS])
    subscriber_distribution = [
        SubscriberDistribution(label=label, count=count)
        for label, count in subscriber_distribution_counts.items()
    ]
    return metrics, heatmap_matrix, subscriber_distribution


def build_subscriber_pie_chart(distribution: list[SubscriberDistribution]) -> str:
    filtered = [item for item in distribution if item.count > 0]
    total = sum(item.count for item in filtered) or 1
    colors = [
        "#4f46e5",
        "#2563eb",
        "#0891b2",
        "#059669",
        "#65a30d",
        "#ca8a04",
        "#ea580c",
        "#dc2626",
        "#6b7280",
    ]
    radius = 88
    circumference = 2 * 3.141592653589793 * radius
    current_offset = 0.0
    circles: list[str] = []
    legends: list[str] = []

    for idx, item in enumerate(filtered):
        color = colors[idx % len(colors)]
        length = circumference * (item.count / total)
        pct = item.count * 100.0 / total
        circles.append(
            f"""
              <circle
                cx="0"
                cy="0"
                r="{radius}"
                fill="none"
                stroke="{color}"
                stroke-width="28"
                stroke-linecap="butt"
                stroke-dasharray="{length:.2f} {circumference:.2f}"
                stroke-dashoffset="-{current_offset:.2f}"
              ></circle>
            """
        )
        legends.append(
            f"""
            <div style="display:flex; align-items:center; gap:10px; padding:10px 12px; border:1px solid #e5e7eb; border-radius:12px; background:#f8fafc;">
              <span style="width:12px; height:12px; border-radius:999px; background:{color}; display:inline-block;"></span>
              <div>
                <div style="color:#6b7280; font-size:13px;">{item.label}</div>
                <div style="font-size:18px; font-weight:700;">{item.count:,}명 ({pct:.2f}%)</div>
              </div>
            </div>
            """
        )
        current_offset += length

    return f"""
    <section class="card">
      <h2>전체 가입자수 기준 연령 분포</h2>
      <p class="subtitle">전체 가입자수에서 각 연령대가 차지하는 비율을 보여주며, 미정 연령도 함께 포함했습니다.</p>
      <div style="display:grid; grid-template-columns:minmax(260px, 320px) 1fr; gap:20px; align-items:center;">
        <div style="display:flex; justify-content:center;">
          <svg viewBox="0 0 260 260" role="img" aria-label="전체 가입자수 기준 연령 분포">
            <g transform="translate(130,130) rotate(-90)">
              <circle cx="0" cy="0" r="{radius}" fill="none" stroke="#e5e7eb" stroke-width="28"></circle>
              {''.join(circles)}
            </g>
            <text x="130" y="122" text-anchor="middle" style="fill:#6b7280; font-size:13px;">전체 가입자수</text>
            <text x="130" y="148" text-anchor="middle" style="fill:#111827; font-size:22px; font-weight:700;">{total:,}명</text>
          </svg>
        </div>
        <div style="display:grid; gap:12px;">
          {''.join(legends)}
        </div>
      </div>
    </section>
    """


def build_html(
    metrics: list[AgeMetric],
    heatmap_matrix: list[list[float]],
    subscriber_distribution: list[SubscriberDistribution],
) -> str:
    best_conversion = max(metrics, key=lambda item: item.buyer_rate_pct)
    highest_aov = max(metrics, key=lambda item: item.avg_order_value)
    highest_orders = max(metrics, key=lambda item: item.orders)

    kpis = build_kpi_cards(
        [
            ("구매자 비율 최고 연령대", f"{best_conversion.age_band} ({best_conversion.buyer_rate_pct:.2f}%)"),
            ("객단가 최고 연령대", f"{highest_aov.age_band} ({highest_aov.avg_order_value:,.0f}원)"),
            ("주문 수 최고 연령대", f"{highest_orders.age_band} ({highest_orders.orders:,}건)"),
        ]
    )

    sections = [
        build_bullet_list(
            "해석 포인트",
            [
                "미정 연령(-1)은 제외했고, 10대와 40대 이상은 각각 하나의 구간으로, 20대와 30대는 초반·중반·후반으로 세분화했습니다.",
                "주문 수와 시간대 주문 히트맵은 전체 주문 데이터를 기준으로 집계했습니다.",
                "연령별 카테고리 선호도는 `enter_browser=0.1`, `add_my_goods=1`, `remove_my_goods=-1` 가중치를 적용한 점수입니다.",
            ],
        ),
        build_subscriber_pie_chart(subscriber_distribution),
        build_bar_chart(
            "연령대별 주문 수",
            "전체 주문량 기준으로 어느 연령대가 많이 기여하는지 확인합니다.",
            [item.age_band for item in metrics],
            [float(item.orders) for item in metrics],
            "#059669",
        ),
        build_bar_chart(
            "연령대별 구매자 비율",
            "연령대별 가입자 수 대비 1회 이상 구매 이력이 있는 구매자의 비율을 나타낸 결과입니다.",
            [item.age_band for item in metrics],
            [item.buyer_rate_pct for item in metrics],
            "#ea580c",
            "%",
            2,
        ),
        build_table(
            "연령대별 선호 카테고리",
            "연령별 카테고리 선호도를 나타낸 표입니다.",
            ["연령대", "선호 카테고리 (1위)", "선호 카테고리 (2위)", "선호 카테고리 (3위)"],
            [
                [
                    item.age_band,
                    item.top_category_1,
                    item.top_category_2,
                    item.top_category_3,
                ]
                for item in metrics
            ],
        ),
        build_heatmap_table(
            "연령대별 시간대 주문 히트맵",
            "전체 주문량 기준으로 각 연령대에서 어떤 시간대에 주문이 몰리는지 보여줍니다.",
            HOURS,
            [item.age_band for item in metrics],
            heatmap_matrix,
        ),
        build_table(
            "연령대별 행동/구매 요약",
            "활동 시간과 주문 시간, 선호 카테고리 상위 3개를 함께 확인합니다.",
            ["연령대", "가입자 수", "활동 유저 수", "구매 유저 수", "구매자 비율", "주문 수", "객단가", "활동 피크", "주문 피크", "선호 카테고리 (1위)", "선호 카테고리 (2위)", "선호 카테고리 (3위)"],
            [
                [
                    item.age_band,
                    f"{item.visitors:,}",
                    f"{item.active_users:,}",
                    f"{item.buyers:,}",
                    f"{item.buyer_rate_pct:.2f}%",
                    f"{item.orders:,}",
                    f"{item.avg_order_value:,.0f}원",
                    f"{item.peak_activity_hour}시",
                    f"{item.peak_order_hour}시",
                    item.top_category_1,
                    item.top_category_2,
                    item.top_category_3,
                ]
                for item in metrics
            ],
        ),
    ]

    return render_page(
        "3. 연령대별 주문 추이",
        "연령대별 주문 수, 방문자 대비 구매자 비율, 시간대별 주문 분포와 행동/구매 요약을 비교한 결과입니다.",
        kpis,
        sections,
    )


def main() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, heatmap_matrix, subscriber_distribution = compute_metrics()
    html = build_html(metrics, heatmap_matrix, subscriber_distribution)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved visualization to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
