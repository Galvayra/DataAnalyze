from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from build.action_db import DB_PATH


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "time_of_day_orders.html"


@dataclass
class HourlyMetric:
    hour: str
    total_events: int
    purchase_orders: int
    revenue: int
    enter_browser_events: int
    purchase_conversions: int
    purchase_conversion_rate_pct: float



def fetch_hourly_metrics() -> list[HourlyMetric]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            WITH RECURSIVE hours(hour) AS (
                SELECT 0
                UNION ALL
                SELECT hour + 1 FROM hours WHERE hour < 23
            ),
            hourly_events AS (
                SELECT
                    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                    COUNT(*) AS total_events
                FROM user_event_logs
                GROUP BY 1
            ),
            hourly_orders AS (
                SELECT
                    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                    COUNT(*) AS purchase_orders,
                    SUM(price) AS revenue
                FROM order_info
                GROUP BY 1
            ),
            enter_events AS (
                SELECT
                    rowid AS event_rowid,
                    timestamp AS enter_timestamp,
                    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                    user_id,
                    event_goods_id,
                    event_shop_id
                FROM user_event_logs
                WHERE event_name = 'enter_browser'
                  AND event_goods_id IS NOT NULL
                  AND event_shop_id IS NOT NULL
            ),
            order_matches AS (
                SELECT
                    o.rowid AS order_rowid,
                    (
                        SELECT e.rowid
                        FROM user_event_logs e
                        WHERE e.event_name = 'enter_browser'
                          AND e.user_id = o.user_id
                          AND e.event_goods_id = o.goods_id
                          AND e.event_shop_id = o.shop_id
                          AND e.timestamp < o.timestamp
                        ORDER BY e.timestamp DESC
                        LIMIT 1
                    ) AS matched_event_rowid,
                    (
                        SELECT MAX(e.timestamp)
                        FROM user_event_logs e
                        WHERE e.event_name = 'enter_browser'
                          AND e.user_id = o.user_id
                          AND e.event_goods_id = o.goods_id
                          AND e.event_shop_id = o.shop_id
                          AND e.timestamp < o.timestamp
                    ) AS enter_timestamp,
                    o.timestamp AS order_timestamp
                FROM order_info o
            ),
            matched_orders_by_hour AS (
                SELECT
                    e.hour,
                    COUNT(*) AS purchase_conversions
                FROM order_matches m
                JOIN enter_events e ON e.event_rowid = m.matched_event_rowid
                WHERE m.enter_timestamp IS NOT NULL
                GROUP BY 1
            ),
            hourly_enter_events AS (
                SELECT
                    hour,
                    COUNT(*) AS enter_browser_events
                FROM enter_events
                GROUP BY 1
            )
            SELECT
                printf('%02d', h.hour) AS hour,
                COALESCE(e.total_events, 0) AS total_events,
                COALESCE(o.purchase_orders, 0) AS purchase_orders,
                COALESCE(o.revenue, 0) AS revenue,
                COALESCE(he.enter_browser_events, 0) AS enter_browser_events,
                COALESCE(c.purchase_conversions, 0) AS purchase_conversions,
                ROUND(COALESCE(c.purchase_conversions, 0) * 100.0 / NULLIF(he.enter_browser_events, 0), 2) AS purchase_conversion_rate_pct
            FROM hours h
            LEFT JOIN hourly_events e ON e.hour = h.hour
            LEFT JOIN hourly_orders o ON o.hour = h.hour
            LEFT JOIN hourly_enter_events he ON he.hour = h.hour
            LEFT JOIN matched_orders_by_hour c ON c.hour = h.hour
            ORDER BY h.hour;
            """
        ).fetchall()
        return [
            HourlyMetric(
                hour=row["hour"],
                total_events=row["total_events"],
                purchase_orders=row["purchase_orders"],
                revenue=row["revenue"],
                enter_browser_events=row["enter_browser_events"],
                purchase_conversions=row["purchase_conversions"],
                purchase_conversion_rate_pct=row["purchase_conversion_rate_pct"] or 0.0,
            )
            for row in rows
        ]
    finally:
        conn.close()
def build_bar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    color: str,
    value_suffix: str = "",
    decimals: int = 0,
) -> str:
    width = 860
    height = 280
    margin_left = 48
    margin_right = 16
    margin_top = 36
    margin_bottom = 42
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max(values) if any(values) else 1
    bar_width = chart_width / max(len(values), 1)

    bars: list[str] = []
    guides: list[str] = []
    x_labels: list[str] = []

    for idx, label in enumerate(labels):
        x = margin_left + idx * bar_width + 4
        bar_inner_width = max(bar_width - 8, 6)
        value = values[idx]
        scaled_height = 0 if max_value == 0 else (value / max_value) * chart_height
        y = margin_top + (chart_height - scaled_height)
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_inner_width:.2f}' height='{scaled_height:.2f}' rx='4' fill='{color}' />"
        )
        bars.append(
            f"<text x='{x + bar_inner_width / 2:.2f}' y='{max(y - 6, 18):.2f}' text-anchor='middle' class='bar-value'>{value:,.{decimals}f}{value_suffix}</text>"
        )
        x_labels.append(
            f"<text x='{x + bar_inner_width / 2:.2f}' y='{height - 14}' text-anchor='middle' class='axis-label'>{label}</text>"
        )

    for step in range(5):
        guide_value = max_value * step / 4
        y = margin_top + chart_height - (chart_height * step / 4)
        guides.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' class='guide' />"
        )
        guides.append(
            f"<text x='{margin_left - 8}' y='{y + 4:.2f}' text-anchor='end' class='axis-label'>{guide_value:,.{decimals}f}{value_suffix}</text>"
        )

    return f"""
    <section class="chart-card">
      <h2>{title}</h2>
      <p class="chart-subtitle">{subtitle}</p>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
        {''.join(guides)}
        <line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" class="axis" />
        {''.join(bars)}
        {''.join(x_labels)}
      </svg>
    </section>
    """
def build_html(metrics: list[HourlyMetric]) -> str:
    hours = [item.hour for item in metrics]
    total_event_values = [float(item.total_events) for item in metrics]
    purchase_order_values = [float(item.purchase_orders) for item in metrics]
    revenue_values = [float(item.revenue) for item in metrics]
    conversion_count_values = [float(item.purchase_conversions) for item in metrics]
    conversion_rate_values = [item.purchase_conversion_rate_pct for item in metrics]
    top_revenue = max(metrics, key=lambda item: item.revenue)
    top_conversion_count = max(metrics, key=lambda item: item.purchase_conversions)
    top_conversion_rate = max(metrics, key=lambda item: item.purchase_conversion_rate_pct)

    kpis = f"""
      <div class="kpis">
        <div class="kpi">
          <div class="label">시간대별 매출 1위</div>
          <div class="value">{top_revenue.hour}시 ({top_revenue.revenue:,})</div>
        </div>
        <div class="kpi">
          <div class="label">시간대별 구매 전환수 1위</div>
          <div class="value">{top_conversion_count.hour}시 ({top_conversion_count.purchase_conversions:,}건)</div>
        </div>
        <div class="kpi">
          <div class="label">시간대별 구매 전환율 1위</div>
          <div class="value">{top_conversion_rate.hour}시 ({top_conversion_rate.purchase_conversion_rate_pct:.2f}%)</div>
        </div>
      </div>
    """

    insight_points = f"""
    <h2>해석 포인트</h2>
    <ul class="insight-list">
      <li>첫 번째 차트는 `user_event_logs` 전체 행 수를 시간대별로 단순 집계한 결과입니다.</li>
      <li>두 번째 차트는 `order_info` 전체 행 수를 시간대별로 단순 집계한 결과입니다.</li>
      <li>구매전환 수와 구매전환율은 주문 1건당 가장 가까운 직전 `enter_browser` 1건만 연결한 총 705건 기준으로 집계했습니다.</li>
      <li>구매전환 시간대는 `enter_browser` 발생 시각을 기준으로 집계하며, 예를 들어 `01:20 이벤트 -> 02:17 구매`는 `01시 전환`으로 계산합니다.</li>
      <li>구매전환율의 분모는 전체 `enter_browser` 로그가 아니라, `event_goods_id`와 `event_shop_id`가 모두 존재하는 상품 조회형 `enter_browser` 이벤트 수입니다.</li>
    </ul>
    """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>1. 시간대별 이벤트와 구매 흐름</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
      --blue: #4f46e5;
      --green: #059669;
      --orange: #ea580c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    .hero, .insight-card, .chart-card, .table-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .hero {{
      margin-bottom: 20px;
    }}
    .insight-card {{
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .insight-card h2 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin: 20px 0;
    }}
    .kpi {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .kpi .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .chart-grid {{
      display: grid;
      gap: 20px;
    }}
    .chart-card h2, .table-card h2 {{
      margin: 0 0 6px;
      font-size: 20px;
    }}
    .chart-subtitle {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.5;
    }}
    svg {{
      width: 100%;
      height: auto;
      overflow: visible;
    }}
    .guide {{
      stroke: #e5e7eb;
      stroke-width: 1;
      stroke-dasharray: 4 4;
    }}
    .axis {{
      stroke: #9ca3af;
      stroke-width: 1.2;
    }}
    .axis-label {{
      fill: #6b7280;
      font-size: 11px;
    }}
    .bar-value {{
      fill: #374151;
      font-size: 9px;
    }}
    .insight-list {{
      margin: 16px 0 0;
      padding-left: 20px;
      line-height: 1.7;
    }}
    @media (max-width: 900px) {{
      body {{ padding: 16px; }}
      .kpis {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>1. 시간대별 이벤트와 구매 흐름</h1>
      <p>
        시간대별 주문 추이와 구매전환율을 정의하고 구매 가능성이 높은 시간대를 파악하기 위한 분석입니다.
      </p>
      {kpis}
    </section>
    <section class="insight-card">
      {insight_points}
    </section>

    <div class="chart-grid">
      {build_bar_chart(
          "시간대별 전체 이벤트 수",
          "유저가 행동한 횟수를 시간대별로 단순 집계한 결과입니다.",
          hours,
          total_event_values,
          "#4f46e5",
      )}
      {build_bar_chart(
          "시간대별 구매 수",
          "전체 주문량 기준으로 시간대별로 단순 집계한 결과입니다.",
          hours,
          purchase_order_values,
          "#059669",
      )}
      {build_bar_chart(
          "시간대별 매출",
          "시간대별로 매출을 단순 집계한 결과입니다.",
          hours,
          revenue_values,
          "#0f766e",
      )}
      {build_bar_chart(
          "시간대별 구매전환 수",
          "주문 1건당 가장 가까운 직전 상품 조회형 이벤트를 구매전환으로 정의하고 구매전환을 주문 조회 시점을 기준으로 집계한 결과입니다.",
          hours,
          conversion_count_values,
          "#7c3aed",
      )}
      {build_bar_chart(
          "시간대별 구매전환율",
          "각 시간대의 상품 조회 이벤트 대비 구매 전환 비율입니다.",
          hours,
          conversion_rate_values,
          "#dc2626",
          "%",
          2,
      )}
    </div>
  </div>
</body>
</html>
"""


def main() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = fetch_hourly_metrics()
    html = build_html(metrics)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved visualization to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
