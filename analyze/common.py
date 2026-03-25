from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def age_to_band(age: int | None) -> str:
    if age is None or age < 0:
        return "미상"
    if age < 20:
        return "10대"
    if age < 25:
        return "20대 초반"
    if age < 30:
        return "20대 후반"
    if age < 35:
        return "30대 초반"
    if age < 40:
        return "30대 후반"
    return "40대 이상"


def age_to_decade(age: int | None) -> str:
    if age is None or age < 0:
        return "미상"
    if age < 20:
        return "10대"
    if age < 30:
        return "20대"
    if age < 40:
        return "30대"
    return "40대 이상"


def age_to_target_band(age: int | None) -> str:
    if age is None or age < 0:
        return "미상"
    if age < 20:
        return "10대"
    if 20 <= age < 30:
        last_digit = age % 10
        if last_digit <= 3:
            return "20대 초반"
        if last_digit <= 6:
            return "20대 중반"
        return "20대 후반"
    if 30 <= age < 40:
        last_digit = age % 10
        if last_digit <= 3:
            return "30대 초반"
        if last_digit <= 6:
            return "30대 중반"
        return "30대 후반"
    return "40대 이상"


def price_to_band(price: int | None) -> str:
    if price is None:
        return "미상"
    if price < 10000:
        return "1만원 미만"
    if price < 20000:
        return "1만원대"
    if price < 30000:
        return "2만원대"
    if price < 50000:
        return "3~4만원대"
    return "5만원 이상"


def target_age_text_to_decades(target_age_raw: str | None) -> list[str]:
    if not target_age_raw:
        return []
    decades: list[str] = []
    for token in target_age_raw.split("/"):
        token = token.strip()
        if token.startswith("10대"):
            decades.append("10대")
        elif token.startswith("20대"):
            decades.append("20대")
        elif token.startswith("30대"):
            decades.append("30대")
        elif token.startswith("40대"):
            decades.append("40대 이상")
    seen: list[str] = []
    for decade in decades:
        if decade not in seen:
            seen.append(decade)
    return seen


def target_age_text_to_bands(target_age_raw: str | None) -> list[str]:
    if not target_age_raw:
        return []
    bands: list[str] = []
    for token in target_age_raw.split("/"):
        token = token.strip()
        if not token:
            continue
        if token == "10대":
            bands.append("10대")
        elif token in {
            "20대 초반",
            "20대 중반",
            "20대 후반",
            "30대 초반",
            "30대 중반",
            "30대 후반",
            "40대 이상",
        }:
            bands.append(token)
    seen: list[str] = []
    for band in bands:
        if band not in seen:
            seen.append(band)
    return seen


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    denominator = denom_x * denom_y
    return numerator / denominator if denominator else 0.0


def format_number(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}"


def build_kpi_cards(kpis: Iterable[tuple[str, str]]) -> str:
    cards = []
    for label, value in kpis:
        cards.append(
            f"""
            <div class="kpi">
              <div class="label">{html.escape(label)}</div>
              <div class="value">{html.escape(value)}</div>
            </div>
            """
        )
    return f'<div class="kpis">{"".join(cards)}</div>'


def build_bullet_list(title: str, items: Iterable[str]) -> str:
    rendered_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in items
    )
    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <ul class="insight-list">{rendered_items}</ul>
    </section>
    """


def build_bar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    color: str,
    value_suffix: str = "",
    decimals: int = 0,
) -> str:
    width = 920
    height = 320
    margin_left = 56
    margin_right = 16
    margin_top = 36
    chart_bottom = 58
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - chart_bottom
    max_value = max(values) if any(values) else 1
    bar_width = chart_width / max(len(values), 1)

    bars: list[str] = []
    x_labels: list[str] = []
    guides: list[str] = []

    for idx, label in enumerate(labels):
        x = margin_left + idx * bar_width + 4
        inner = max(bar_width - 8, 6)
        value = values[idx]
        scaled = 0 if max_value == 0 else (value / max_value) * chart_height
        y = margin_top + chart_height - scaled
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{inner:.2f}' height='{scaled:.2f}' rx='4' fill='{color}' />"
        )
        bars.append(
            f"<text x='{x + inner / 2:.2f}' y='{max(y - 6, 18):.2f}' text-anchor='middle' class='bar-value'>{format_number(value, decimals)}{value_suffix}</text>"
        )
        x_labels.append(
            f"<text x='{x + inner / 2:.2f}' y='{height - 18}' text-anchor='middle' class='axis-label'>{html.escape(label)}</text>"
        )

    for step in range(5):
        guide_value = max_value * step / 4
        y = margin_top + chart_height - (chart_height * step / 4)
        guides.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' class='guide' />"
        )
        guides.append(
            f"<text x='{margin_left - 10}' y='{y + 4:.2f}' text-anchor='end' class='axis-label'>{format_number(guide_value, 0)}{value_suffix}</text>"
        )

    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <p class="subtitle">{html.escape(subtitle)}</p>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        {''.join(guides)}
        <line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" class="axis" />
        {''.join(bars)}
        {''.join(x_labels)}
      </svg>
    </section>
    """


def build_grouped_bar_chart(
    title: str,
    subtitle: str,
    labels: list[str],
    series: list[tuple[str, str, list[float]]],
    value_suffix: str = "",
    decimals: int = 0,
) -> str:
    width = 920
    height = 340
    margin_left = 56
    margin_right = 20
    margin_top = 36
    chart_bottom = 68
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - chart_bottom
    all_values = [value for _, _, values in series for value in values]
    max_value = max(all_values) if any(all_values) else 1
    group_width = chart_width / max(len(labels), 1)
    series_count = max(len(series), 1)
    bar_width = max((group_width - 12) / series_count, 6)

    guides: list[str] = []
    bars: list[str] = []
    x_labels: list[str] = []
    legend_items: list[str] = []

    for step in range(5):
        guide_value = max_value * step / 4
        y = margin_top + chart_height - (chart_height * step / 4)
        guides.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' class='guide' />"
        )
        guides.append(
            f"<text x='{margin_left - 10}' y='{y + 4:.2f}' text-anchor='end' class='axis-label'>{format_number(guide_value, 0)}{value_suffix}</text>"
        )

    for series_idx, (name, color, values) in enumerate(series):
        legend_items.append(
            f"<span class='legend-item'><span class='legend-swatch' style='background:{color}'></span>{html.escape(name)}</span>"
        )
        for idx, label in enumerate(labels):
            group_start = margin_left + idx * group_width + 6
            x = group_start + series_idx * bar_width
            value = values[idx]
            scaled = 0 if max_value == 0 else (value / max_value) * chart_height
            y = margin_top + chart_height - scaled
            bars.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width - 2:.2f}' height='{scaled:.2f}' rx='4' fill='{color}' />"
            )
            bars.append(
                f"<text x='{x + (bar_width - 2) / 2:.2f}' y='{max(y - 6, 18):.2f}' text-anchor='middle' class='bar-value'>{format_number(value, decimals)}{value_suffix}</text>"
            )

    for idx, label in enumerate(labels):
        x_labels.append(
            f"<text x='{margin_left + idx * group_width + group_width / 2:.2f}' y='{height - 18}' text-anchor='middle' class='axis-label'>{html.escape(label)}</text>"
        )

    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <p class="subtitle">{html.escape(subtitle)}</p>
      <div class="legend">{''.join(legend_items)}</div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        {''.join(guides)}
        <line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" class="axis" />
        {''.join(bars)}
        {''.join(x_labels)}
      </svg>
    </section>
    """


def build_table(
    title: str,
    subtitle: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <p class="subtitle">{html.escape(subtitle)}</p>
      <table>
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </section>
    """


def build_heatmap_table(
    title: str,
    subtitle: str,
    column_labels: list[str],
    row_labels: list[str],
    matrix: list[list[float]],
    formatter: Callable[[float], str] | None = None,
) -> str:
    formatter = formatter or (lambda value: format_number(value, 0))
    max_value = max((cell for row in matrix for cell in row), default=1)

    head = "".join(f"<th>{html.escape(label)}</th>" for label in column_labels)
    body_rows: list[str] = []
    for row_label, row_values in zip(row_labels, matrix):
        cells = [f"<th>{html.escape(row_label)}</th>"]
        for value in row_values:
            intensity = 0 if max_value == 0 else value / max_value
            alpha = 0.08 + 0.55 * intensity
            cells.append(
                f"<td style='background: rgba(79, 70, 229, {alpha:.3f});'>{formatter(value)}</td>"
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <section class="card">
      <h2>{html.escape(title)}</h2>
      <p class="subtitle">{html.escape(subtitle)}</p>
      <table class="heatmap-table">
        <thead><tr><th></th>{head}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </section>
    """


def render_page(
    title: str,
    description: str,
    kpis: str,
    sections: list[str],
) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
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
      max-width: 1120px;
      margin: 0 auto;
    }}
    .hero, .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .hero {{
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    h2 {{
      margin: 0 0 6px;
      font-size: 20px;
    }}
    .hero p, .subtitle {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    .subtitle {{
      margin-bottom: 14px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}
    .kpi {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      gap: 20px;
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
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
      text-align: right;
      font-size: 14px;
      vertical-align: middle;
      word-break: keep-all;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    .heatmap-table th, .heatmap-table td {{
      text-align: center;
    }}
    .insight-list {{
      margin: 0;
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
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(description)}</p>
      {kpis}
    </section>
    <div class="grid">
      {''.join(sections)}
    </div>
  </div>
</body>
</html>
"""
