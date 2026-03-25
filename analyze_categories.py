from __future__ import annotations

import json
from dataclasses import dataclass, replace
from html import escape
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
ANALYZE_OUTPUT_DIR = ROOT_DIR / "analyze" / "output"
OUTPUT_PATH = ANALYZE_OUTPUT_DIR / "classify_categories_comparision.html"

REPORT_SOURCES: list[tuple[str, str, Path]] = [
    ("classification", "TF-IDF 베이스라인", OUTPUTS_DIR / "classification"),
    ("classification_with_emb", "임베딩 + SVM", OUTPUTS_DIR / "classification_with_emb"),
    ("classification_with_ffnn", "FFNN", OUTPUTS_DIR / "classification_with_ffnn"),
    ("classification_with_search", "semantic search", OUTPUTS_DIR / "classification_with_search"),
    ("classification_with_rag", "semantic search + LLM", OUTPUTS_DIR / "classification_with_rag"),
]

EXCLUDED_KEYS = {"accuracy", "macro avg", "weighted avg", "process_time"}


@dataclass(frozen=True)
class CategoryMetric:
    label: str
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class ModelReport:
    source_key: str
    source_title: str
    experiment_name: str
    display_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    process_time: float | None
    sample_count: int
    categories: list[CategoryMetric]


def format_float(value: float) -> str:
    return f"{value:.4f}"


def format_avg_process_time_ms(total_seconds: float | None, sample_count: int) -> str:
    if total_seconds is None or sample_count <= 0:
        return "-"
    avg_ms = (float(total_seconds) * 1000.0) / float(sample_count)
    return f"{avg_ms:.3f}ms"


def format_summary_process_time(report: ModelReport) -> str:
    base_text = format_avg_process_time_ms(report.process_time, report.sample_count)
    if base_text == "-":
        return base_text
    if report.source_key == "classification":
        return base_text
    return f"{base_text} + α"


def friendly_embedding_model_name(raw_name: str) -> str:
    mapping = {
        "KoE5": "KoE5",
        "KURE-v1": "KURE-v1",
        "multilingual-e5-small-ko-v2": "mE5-small-ko-v2",
    }
    return mapping.get(raw_name, raw_name)


def get_display_name(source_key: str, experiment_name: str) -> str:
    if source_key == "classification":
        if experiment_name.startswith("tfidf_word_unigram_bigram_plus_char"):
            feature_name = "word + char TFIDF"
        elif experiment_name.startswith("tfidf_word_unigram_bigram"):
            feature_name = "word TFIDF"
        else:
            feature_name = experiment_name

        classifier_name = "SVM" if "__linear_svm" in experiment_name else "분류기"
        return f"{feature_name} + {classifier_name}"

    if source_key == "classification_with_emb":
        if experiment_name.startswith("tfidf_plus_") and experiment_name.endswith(
            "_description_dense__linear_svm"
        ):
            model_name = experiment_name[
                len("tfidf_plus_") : -len("_description_dense__linear_svm")
            ]
            return (
                f"word + char TFIDF + "
                f"{friendly_embedding_model_name(model_name)} + SVM"
            )

        if experiment_name.endswith("_description_dense__linear_svm"):
            model_name = experiment_name[: -len("_description_dense__linear_svm")]
            return f"{friendly_embedding_model_name(model_name)} + SVM"

    if source_key == "classification_with_ffnn":
        if experiment_name.startswith("ffnn_"):
            model_name = experiment_name[len("ffnn_") :]
            return f"{friendly_embedding_model_name(model_name)} + FFNN"

    if source_key == "classification_with_search":
        model_name = experiment_name.split("_faiss_search_rr_top", 1)[0]
        return f"{friendly_embedding_model_name(model_name)} 모델 기반 검색 시스템"

    if source_key == "classification_with_rag":
        model_name = experiment_name.split("_faiss_rag_top", 1)[0]
        return f"{friendly_embedding_model_name(model_name)} 모델 기반 검색 시스템 + LLM"

    return experiment_name


def get_source_title(source_key: str, experiment_name: str, default_title: str) -> str:
    if source_key == "classification":
        if experiment_name.startswith("tfidf_word_unigram_bigram_plus_char"):
            return "TF-IDF plus + SVM"
        return "TF-IDF + SVM"

    if source_key == "classification_with_emb":
        if experiment_name.startswith("tfidf_plus_"):
            return "TF-IDF plus + 임베딩 + SVM"
        return "임베딩 + SVM"

    return default_title


def is_tfidf_baseline_report(report: ModelReport) -> bool:
    return report.source_key == "classification"


def baseline_sort_key(report: ModelReport) -> tuple[float, float, float]:
    return (report.macro_f1, report.accuracy, report.weighted_f1)


def annotate_tfidf_baseline(reports: list[ModelReport]) -> list[ModelReport]:
    baseline_reports = [report for report in reports if is_tfidf_baseline_report(report)]
    if not baseline_reports:
        return reports

    best_baseline = max(baseline_reports, key=baseline_sort_key)
    annotated_reports: list[ModelReport] = []
    for report in reports:
        if report.experiment_name != best_baseline.experiment_name:
            annotated_reports.append(report)
            continue

        annotated_reports.append(
            replace(
                report,
                source_title=f"{report.source_title} (baseline)",
                display_name=f"{report.display_name} (baseline)",
            )
        )
    return annotated_reports


def load_report_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload)!r}")
    return payload


def parse_process_time(payload: dict[str, Any]) -> float | None:
    process_time = payload.get("process_time")
    if isinstance(process_time, (int, float)):
        return float(process_time)
    if isinstance(process_time, dict):
        total_time = process_time.get("total_time")
        if isinstance(total_time, (int, float)):
            return float(total_time)
    return None


def parse_model_report(source_key: str, source_title: str, report_path: Path) -> ModelReport:
    experiment_name = report_path.name.removesuffix("__test_classification_report.json")
    payload = load_report_json(report_path)

    categories: list[CategoryMetric] = []
    for label, metrics in payload.items():
        if label in EXCLUDED_KEYS or not isinstance(metrics, dict):
            continue
        categories.append(
            CategoryMetric(
                label=str(label),
                precision=float(metrics.get("precision", 0.0)),
                recall=float(metrics.get("recall", 0.0)),
                f1_score=float(metrics.get("f1-score", 0.0)),
                support=int(float(metrics.get("support", 0.0))),
            )
        )

    categories.sort(key=lambda item: item.label)

    macro_avg = payload.get("macro avg", {})
    weighted_avg = payload.get("weighted avg", {})
    sample_count = int(float(weighted_avg.get("support", 0.0)))
    return ModelReport(
        source_key=source_key,
        source_title=get_source_title(source_key, experiment_name, source_title),
        experiment_name=experiment_name,
        display_name=get_display_name(source_key, experiment_name),
        accuracy=float(payload.get("accuracy", 0.0)),
        macro_f1=float(macro_avg.get("f1-score", 0.0)),
        weighted_f1=float(weighted_avg.get("f1-score", 0.0)),
        process_time=parse_process_time(payload),
        sample_count=sample_count,
        categories=categories,
    )


def discover_reports() -> list[ModelReport]:
    reports: list[ModelReport] = []
    for source_key, source_title, directory in REPORT_SOURCES:
        if not directory.exists():
            continue
        for report_path in sorted(directory.glob("*__test_classification_report.json")):
            reports.append(parse_model_report(source_key, source_title, report_path))
    reports = annotate_tfidf_baseline(reports)
    reports.sort(key=lambda item: (item.macro_f1, item.accuracy, item.weighted_f1), reverse=True)
    return reports


def build_summary_table(reports: list[ModelReport]) -> str:
    if not reports:
        return "<p>표시할 분류 결과가 없습니다.</p>"

    best_accuracy = max(report.accuracy for report in reports)
    best_macro_f1 = max(report.macro_f1 for report in reports)
    best_weighted_f1 = max(report.weighted_f1 for report in reports)

    rows: list[str] = []
    for index, report in enumerate(reports, start=1):
        accuracy_class = "best" if report.accuracy == best_accuracy else ""
        macro_class = "best" if report.macro_f1 == best_macro_f1 else ""
        weighted_class = "best" if report.weighted_f1 == best_weighted_f1 else ""
        rows.append(
            f"""
            <tr>
              <td>{index}</td>
              <td>{escape(report.source_title)}</td>
              <td><a href="#{escape(report.experiment_name)}">{escape(report.display_name)}</a></td>
              <td class="{accuracy_class}">{format_float(report.accuracy)}</td>
              <td class="{macro_class}">{format_float(report.macro_f1)}</td>
              <td class="{weighted_class}">{format_float(report.weighted_f1)}</td>
              <td>{format_summary_process_time(report)}</td>
            </tr>
            """
        )

    return f"""
    <section class="card">
      <h2>모델별 최종 점수 비교</h2>
      <p class="subtitle">
        Accuracy, Macro F1, Weighted F1 기준으로 전체 모델을 비교하고, Macro F1이 높은 순으로 정렬했습니다.

        평균 소요 시간은 샘플당 평균 추론 시간(ms)이며, 임베딩 계열 모델은 임베딩 처리 시간을 계산하지 않아 해당 비용이 빠져 있습니다.

        따라서 `TF-IDF + SVM`, `TF-IDF plus + SVM`을 제외한 모델의 시간 표기에는 `+ α`를 함께 붙였습니다.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>실험군</th>
              <th>모델</th>
              <th>Accuracy</th>
              <th>Macro F1</th>
              <th>Weighted F1</th>
              <th>평균 소요 시간</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </section>
    """


def build_category_table(report: ModelReport) -> str:
    rows: list[str] = []
    for metric in report.categories:
        rows.append(
            f"""
            <tr>
              <td>{escape(metric.label)}</td>
              <td>{format_float(metric.precision)}</td>
              <td>{format_float(metric.recall)}</td>
              <td>{format_float(metric.f1_score)}</td>
              <td>{metric.support:,}</td>
            </tr>
            """
        )

    return f"""
    <section class="card model-card" id="{escape(report.experiment_name)}">
      <div class="model-header">
        <div>
          <div class="eyebrow">{escape(report.source_title)}</div>
          <h3>{escape(report.display_name)}</h3>
          <p class="subtitle code-name">원본 파일 prefix: <code>{escape(report.experiment_name)}</code></p>
        </div>
        <div class="mini-kpis">
          <div class="mini-kpi">
            <span>Accuracy</span>
            <strong>{format_float(report.accuracy)}</strong>
          </div>
          <div class="mini-kpi">
            <span>Macro F1</span>
            <strong>{format_float(report.macro_f1)}</strong>
          </div>
          <div class="mini-kpi">
            <span>Weighted F1</span>
            <strong>{format_float(report.weighted_f1)}</strong>
          </div>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>카테고리</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-score</th>
              <th>샘플 수</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </section>
    """


def build_html(reports: list[ModelReport]) -> str:
    total_models = len(reports)
    best_report = max(reports, key=lambda item: item.macro_f1) if reports else None

    hero = f"""
    <section class="card hero">
      <h1>카테고리 분류 결과 비교</h1>
      <div class="hero-kpis">
        <div class="hero-kpi">
          <span>비교 모델 수</span>
          <strong>{total_models}</strong>
        </div>
        <div class="hero-kpi">
          <span>Macro F1 최고 모델</span>
          <strong>{escape(best_report.display_name) if best_report else '-'}</strong>
        </div>
        <div class="hero-kpi">
          <span>최고 Macro F1</span>
          <strong>{format_float(best_report.macro_f1) if best_report else '-'}</strong>
        </div>
      </div>
    </section>
    """

    detail_sections = "".join(build_category_table(report) for report in reports)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>카테고리 분류 결과 비교</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --best: #dcfce7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      white-space: pre-line;
    }}
    .hero-kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}
    .hero-kpi, .mini-kpi {{
      border: 1px solid var(--border);
      background: #f8fafc;
      border-radius: 14px;
      padding: 16px;
    }}
    .hero-kpi span, .mini-kpi span, .eyebrow {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .hero-kpi strong {{
      font-size: 22px;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin-top: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      background: #f8fafc;
      font-weight: 700;
      position: sticky;
      top: 0;
    }}
    td.best {{
      background: var(--best);
      font-weight: 700;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .model-card h3 {{
      margin: 4px 0 0;
      font-size: 22px;
    }}
    .model-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 20px;
      flex-wrap: wrap;
    }}
    .mini-kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 12px;
      min-width: 360px;
    }}
    .mini-kpi strong {{
      font-size: 18px;
    }}
    .code-name code {{
      font-size: 12px;
      background: var(--accent-soft);
      padding: 2px 6px;
      border-radius: 8px;
    }}
    @media (max-width: 900px) {{
      body {{ padding: 16px; }}
      .hero-kpis, .mini-kpis {{
        grid-template-columns: 1fr;
      }}
      .mini-kpis {{
        min-width: 100%;
      }}
    }}
  </style>
</head>
<body>
  <main class="container">
    {hero}
    {build_summary_table(reports)}
    {detail_sections}
  </main>
</body>
</html>
"""


def main() -> None:
    reports = discover_reports()
    ANALYZE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_html(reports), encoding="utf-8")
    print(f"Saved HTML report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
