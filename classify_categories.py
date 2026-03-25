from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

try:
    import faiss
except ImportError:
    faiss = None


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "parsing"
EMBEDDING_DIR = ROOT_DIR / "pickle" / "embedding"
INDEX_DIR = ROOT_DIR / "index"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "classification"
DEFAULT_EMBEDDING_OUTPUT_DIR = ROOT_DIR / "outputs" / "classification_with_emb"
DEFAULT_FFNN_OUTPUT_DIR = ROOT_DIR / "outputs" / "classification_with_ffnn"
DEFAULT_SEARCH_OUTPUT_DIR = ROOT_DIR / "outputs" / "classification_with_search"
DEFAULT_RAG_OUTPUT_DIR = ROOT_DIR / "outputs" / "classification_with_rag"
DEFAULT_CKPT_ROOT = ROOT_DIR / "outputs" / "ckpt"


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    texts: list[str]
    labels: list[str]
    titles: list[str]
    descriptions: list[str]
    item_ids: list[str]


@dataclass(frozen=True)
class ExperimentResult:
    experiment_name: str
    feature_type: str
    model_name: str
    split_name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    artifact_dir: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "feature_type": self.feature_type,
            "model_name": self.model_name,
            "split_name": self.split_name,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
        }


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_input_text(title: str, description: str) -> str:
    title = normalize_text(title)
    description = normalize_text(description)
    input_text = title + " " + description
    return input_text.strip()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(data)!r}")
    return data


def load_split(name: str, path: Path) -> DatasetSplit:
    records = load_records(path)

    titles: list[str] = []
    descriptions: list[str] = []
    texts: list[str] = []
    labels: list[str] = []
    item_ids: list[str] = []

    for idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{path} row {idx} must be a dict.")

        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")
        label = str(row.get("label", "") or "").strip()
        item_id = str(row.get("item_id", "") or "").strip()

        if label == "":
            raise ValueError(f"{path} row {idx} has an empty label.")

        titles.append(title)
        descriptions.append(description)
        texts.append(build_input_text(title, description))
        labels.append(label)
        item_ids.append(item_id)

    return DatasetSplit(
        name=name,
        texts=texts,
        labels=labels,
        titles=titles,
        descriptions=descriptions,
        item_ids=item_ids,
    )


def build_splits(data_dir: Path) -> dict[str, DatasetSplit]:
    return {
        "train": load_split("train", data_dir / "train.json"),
        "valid": load_split("valid", data_dir / "valid.json"),
        "test": load_split("test", data_dir / "test.json"),
    }


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "default"


def get_model_basename(model_name: str) -> str:
    return Path(model_name).name or sanitize_name(model_name)


def discover_embedding_models(embedding_dir: Path = EMBEDDING_DIR) -> list[str]:
    if not embedding_dir.exists():
        return []

    split_requirements = {"train", "valid", "test"}
    grouped: dict[str, set[str]] = {}

    for path in embedding_dir.glob("*.pkl"):
        stem = path.stem
        for split_name in split_requirements:
            suffix = f"_{split_name}"
            if stem.endswith(suffix):
                prefix = stem[: -len(suffix)]
                grouped.setdefault(prefix, set()).add(split_name)
                break

    discovered_models: list[str] = []
    for prefix, splits_found in sorted(grouped.items()):
        if splits_found != split_requirements:
            continue

        train_cache_path = embedding_dir / f"{prefix}_train.pkl"
        try:
            with train_cache_path.open("rb") as f:
                payload = pickle.load(f)
        except Exception:
            print(f"[Embedding] Ignore {prefix} - failed to read cache payload")
            continue

        model_name = str(payload.get("model_name", "") or "").strip() if isinstance(payload, dict) else ""
        if model_name == "":
            print(f"[Embedding] Ignore {prefix} - missing model_name in cache")
            continue
        discovered_models.append(model_name)

    return discovered_models


def discover_ffnn_checkpoints(ckpt_root: Path = DEFAULT_CKPT_ROOT) -> list[Path]:
    if not ckpt_root.exists():
        return []
    checkpoint_paths = sorted(ckpt_root.glob("*/best_valid_macro_f1.pt"))
    return checkpoint_paths


def discover_search_models(
    index_dir: Path = INDEX_DIR,
    embedding_dir: Path = EMBEDDING_DIR,
) -> list[str]:
    if not index_dir.exists() or not embedding_dir.exists():
        return []

    discovered_models: list[str] = []
    seen_model_names: set[str] = set()

    for index_path in sorted(index_dir.glob("*_flat.index")):
        stem = index_path.stem
        if not stem.endswith("_flat"):
            continue
        prefix = stem[: -len("_flat")]
        item_ids_path = index_dir / f"{prefix}_flat_item_ids.json"
        test_cache_path = embedding_dir / f"{prefix}_test.pkl"
        if not item_ids_path.exists():
            continue
        if not test_cache_path.exists():
            continue

        try:
            with test_cache_path.open("rb") as f:
                payload = pickle.load(f)
        except Exception:
            print(f"[Search] Ignore {prefix} - failed to read test embedding cache")
            continue

        model_name = str(payload.get("model_name", "") or "").strip() if isinstance(payload, dict) else ""
        if model_name == "":
            print(f"[Search] Ignore {prefix} - missing model_name in test embedding cache")
            continue
        if model_name in seen_model_names:
            continue
        seen_model_names.add(model_name)
        discovered_models.append(model_name)

    return discovered_models


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    experiment_name: str,
    feature_type: str,
    model_name: str,
    split_name: str,
    artifact_dir: str = "",
) -> ExperimentResult:
    return ExperimentResult(
        experiment_name=experiment_name,
        feature_type=feature_type,
        model_name=model_name,
        split_name=split_name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted")),
        artifact_dir=artifact_dir,
    )


def make_logistic_regression() -> LogisticRegression:
    # return LogisticRegression(
    #     max_iter=3000,
    #     C=4.0,
    #     class_weight="balanced",
    #     solver="liblinear",
    #     multi_class="ovr",
    # )
    return LogisticRegression(max_iter=10000, C=1.0, class_weight="balanced", solver="saga")


def make_linear_svm() -> LinearSVC:
    return LinearSVC(max_iter=10000, C=1.0, class_weight="balanced")


def build_word_tfidf_vectorizer(ngram_range: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.98,
        max_features=200000,
        sublinear_tf=True,
    )


def build_char_tfidf_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.98,
        max_features=150000,
        sublinear_tf=True,
    )


def build_tfidf_experiments() -> list[tuple[str, str, Any, str]]:
    base_models = [
        # ("logreg", make_logistic_regression()),
        ("linear_svm", make_linear_svm()),
    ]

    feature_builders = [
        (
            "tfidf_word_unigram_bigram",
            build_word_tfidf_vectorizer((1, 2)),
            "word_unigram_bigram",
        ),
        (
            "tfidf_word_unigram_bigram_plus_char",
            FeatureUnion(
                [
                    ("word", build_word_tfidf_vectorizer((1, 2))),
                    ("char", build_char_tfidf_vectorizer()),
                ]
            ),
            "word_unigram_bigram_plus_char",
        ),
    ]

    experiments: list[tuple[str, str, Any, str]] = []
    for feature_name, feature_extractor, feature_type in feature_builders:
        for model_name, estimator in base_models:
            pipeline = Pipeline(
                [
                    ("features", feature_extractor),
                    ("classifier", clone(estimator)),
                ]
            )
            experiments.append(
                (
                    f"{feature_name}__{model_name}",
                    feature_type,
                    pipeline,
                    model_name,
                )
            )
    return experiments


def evaluate_predictions(
    experiment_name: str,
    feature_type: str,
    model_name: str,
    split_name: str,
    y_true: list[str],
    y_pred: list[str],
    output_dir: Path,
    process_time: float | None = None,
) -> ExperimentResult:
    result = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        experiment_name=experiment_name,
        feature_type=feature_type,
        model_name=model_name,
        split_name=split_name,
        artifact_dir=str(output_dir),
    )

    report_path = output_dir / f"{experiment_name}__{split_name}_classification_report.json"
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    if process_time is None:
        process_time = load_existing_process_time(report_path)
    if process_time is not None:
        report["process_time"] = process_time
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_path = output_dir / f"{experiment_name}__{split_name}_confusion_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())

    return result


def build_process_time(total_time_seconds: float, num_samples: int) -> float:
    safe_total_time = max(float(total_time_seconds), 0.0)
    _ = num_samples
    return round(safe_total_time, 4)


def load_existing_process_time(report_path: Path) -> float | None:
    if not report_path.exists():
        return None

    try:
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return None

    if not isinstance(report, dict):
        return None

    process_time = report.get("process_time")
    if isinstance(process_time, (int, float)):
        return float(process_time)
    if isinstance(process_time, dict):
        total_time = process_time.get("total_time")
        if isinstance(total_time, (int, float)):
            return float(total_time)
    return None


def save_predictions(
    output_dir: Path,
    experiment_name: str,
    split: DatasetSplit,
    predictions: list[str],
) -> None:
    path = output_dir / f"{experiment_name}__{split.name}_predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "description", "label", "prediction", "is_correct"])
        for title, description, label, pred in zip(
            split.titles,
            split.descriptions,
            split.labels,
            predictions,
        ):
            writer.writerow([title, description, label, pred, int(label == pred)])


def load_cached_predictions(
    output_dir: Path,
    experiment_name: str,
    split: DatasetSplit,
) -> list[str] | None:
    path = output_dir / f"{experiment_name}__{split.name}_predictions.csv"
    if not path.exists():
        return None

    predictions: list[str] = []
    cached_labels: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append((row.get("prediction") or "").strip())
            cached_labels.append((row.get("label") or "").strip())

    if len(predictions) != len(split.labels):
        print(
            f"[Cache] Ignore {experiment_name} - "
            f"row count mismatch ({len(predictions)} != {len(split.labels)})"
        )
        return None

    if cached_labels != split.labels:
        print(f"[Cache] Ignore {experiment_name} - label order mismatch")
        return None

    return predictions


def get_embedding_cache_path(
    model_name: str,
    split: DatasetSplit,
    embedding_dir: Path = EMBEDDING_DIR,
) -> Path:
    model_key = sanitize_name(get_model_basename(model_name))
    return embedding_dir / f"{model_key}_{split.name}.pkl"


def print_embedding_cache_help(model_name: str, normalize_embeddings: bool) -> None:
    command = f"python indexing.py --model-name {model_name}"
    if not normalize_embeddings:
        command += " --no-normalize-embeddings"
    print("[Embedding] Precomputed embedding cache not found.")
    print(f"[Embedding] Run this command first: {command}")


def load_precomputed_embeddings(
    model_name: str,
    split: DatasetSplit,
    normalize_embeddings: bool,
    embedding_dir: Path = EMBEDDING_DIR,
) -> np.ndarray:
    cache_path = get_embedding_cache_path(model_name, split, embedding_dir)
    if not cache_path.exists():
        print_embedding_cache_help(model_name, normalize_embeddings)
        raise SystemExit(1)

    with cache_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        print(f"[Embedding] Invalid cache payload: {cache_path}")
        raise SystemExit(1)

    cached_model_name = str(payload.get("model_name", "") or "")
    cached_num_texts = int(payload.get("num_texts", -1))
    cached_normalize = bool(payload.get("normalize_embeddings"))
    cached_item_ids = payload.get("item_ids")
    embeddings = payload.get("embeddings")

    if cached_model_name != model_name:
        print(f"[Embedding] Cache model mismatch: {cache_path.name}")
        print_embedding_cache_help(model_name, normalize_embeddings)
        raise SystemExit(1)

    if cached_num_texts != len(split.texts):
        print(
            f"[Embedding] Cache row count mismatch for {split.name}: "
            f"{cached_num_texts} != {len(split.texts)}"
        )
        print_embedding_cache_help(model_name, normalize_embeddings)
        raise SystemExit(1)

    if cached_normalize != normalize_embeddings:
        print(f"[Embedding] Cache normalize flag mismatch: {cache_path.name}")
        print_embedding_cache_help(model_name, normalize_embeddings)
        raise SystemExit(1)

    if split.item_ids and cached_item_ids is not None and cached_item_ids != split.item_ids:
        print(f"[Embedding] Cache item_id order mismatch: {cache_path.name}")
        print_embedding_cache_help(model_name, normalize_embeddings)
        raise SystemExit(1)

    if not isinstance(embeddings, np.ndarray):
        print(f"[Embedding] Invalid embeddings array in cache: {cache_path}")
        raise SystemExit(1)

    print(f"[Embedding] Load cache - {cache_path}")
    return embeddings.astype(np.float32, copy=False)


def get_display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def ljust_display(text: str, width: int) -> str:
    padding = max(width - get_display_width(text), 0)
    return text + (" " * padding)


def print_per_class_metrics(output_dir: Path, result: ExperimentResult) -> None:
    report_path = output_dir / f"{result.experiment_name}__{result.split_name}_classification_report.json"
    if not report_path.exists():
        print("[Per-class] classification report file not found.")
        return

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    excluded_keys = {"accuracy", "macro avg", "weighted avg", "process_time"}
    label_rows: list[tuple[str, float, float, float, int]] = []
    for label, metrics in report.items():
        if label in excluded_keys or not isinstance(metrics, dict):
            continue
        label_rows.append(
            (
                label,
                float(metrics.get("precision", 0.0)),
                float(metrics.get("recall", 0.0)),
                float(metrics.get("f1-score", 0.0)),
                int(metrics.get("support", 0)),
            )
        )

    label_rows.sort(key=lambda row: row[0])
    label_col_width = max(
        get_display_width("label"),
        get_display_width("TOTAL"),
        get_display_width("ACCURACY"),
        max((get_display_width(label) for label, *_ in label_rows), default=0),
        20,
    ) + 2

    print(f"\nPer-class metrics for {result.experiment_name} ({result.split_name})")
    print(
        f"{ljust_display('label', label_col_width)} "
        f"{'precision':>10} "
        f"{'recall':>10} "
        f"{'f1':>10} "
        f"{'샘플 수':>10}"
    )
    for label, precision, recall, f1, support in label_rows:
        print(
            f"{ljust_display(label, label_col_width)} "
            f"{precision:>10.4f} "
            f"{recall:>10.4f} "
            f"{f1:>10.4f} "
            f"{support:>10}"
        )

    weighted_avg = report.get("weighted avg", {})
    total_precision = float(weighted_avg.get("precision", 0.0))
    total_recall = float(weighted_avg.get("recall", 0.0))
    total_f1 = float(weighted_avg.get("f1-score", 0.0))
    total_support = int(weighted_avg.get("support", 0))
    total_accuracy = float(report.get("accuracy", 0.0))

    print("-" * (label_col_width + 46))
    print(
        f"{ljust_display('TOTAL', label_col_width)} "
        f"{total_precision:>10.4f} "
        f"{total_recall:>10.4f} "
        f"{total_f1:>10.4f} "
        f"{total_support:>10}"
    )
    print(f"{ljust_display('ACCURACY', label_col_width)} {total_accuracy:>10.4f}")


def run_tfidf_experiments(
    splits: dict[str, DatasetSplit],
    output_dir: Path,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    train_split = splits["train"]
    test_split = splits["test"]
    split = test_split

    for experiment_name, feature_type, model, model_name in build_tfidf_experiments():
        process_time: float | None = None
        predictions = load_cached_predictions(output_dir, experiment_name, split)
        if predictions is None:
            print(f"[TF-IDF] Training {experiment_name}")
            model.fit(train_split.texts, train_split.labels)
            start_time = time.perf_counter()
            predictions = model.predict(split.texts).tolist()
            process_time = build_process_time(time.perf_counter() - start_time, len(split.labels))
            save_predictions(output_dir, experiment_name, split, predictions)
        else:
            print(f"[TF-IDF] Use cached predictions for {experiment_name}")

        result = evaluate_predictions(
            experiment_name=experiment_name,
            feature_type=feature_type,
            model_name=model_name,
            split_name=split.name,
            y_true=split.labels,
            y_pred=predictions,
            output_dir=output_dir,
            process_time=process_time,
        )
        results.append(result)
        print(
            f"  - {split.name}: "
            f"accuracy={result.accuracy:.4f}, "
            f"macro_f1={result.macro_f1:.4f}, "
            f"weighted_f1={result.weighted_f1:.4f}"
        )

    return results


def run_embedding_experiments(
    splits: dict[str, DatasetSplit],
    output_dir: Path,
    normalize_embeddings: bool,
    embedding_dir: Path = EMBEDDING_DIR,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    embedding_model_names = discover_embedding_models(embedding_dir)
    if not embedding_model_names:
        print(f"[Embedding] No complete embedding cache sets found in {embedding_dir}")
        return results
    print("[Embedding] Discovered models:")
    for model_name in embedding_model_names:
        print(f"  - {model_name}")

    train_split = splits["train"]
    test_split = splits["test"]
    split = test_split

    for embedding_model_name in embedding_model_names:
        print(f"[Embedding] Run experiments for {embedding_model_name}")
        train_embeddings = load_precomputed_embeddings(
            model_name=embedding_model_name,
            split=train_split,
            normalize_embeddings=normalize_embeddings,
            embedding_dir=embedding_dir,
        )
        test_embeddings = load_precomputed_embeddings(
            model_name=embedding_model_name,
            split=test_split,
            normalize_embeddings=normalize_embeddings,
            embedding_dir=embedding_dir,
        )

        model_name = get_model_basename(embedding_model_name)
        dense_experiment_name = model_name + "_description_dense__linear_svm"
        process_time: float | None = None
        predictions = load_cached_predictions(output_dir, dense_experiment_name, split)
        if predictions is None:
            print(f"[Embedding] Training {dense_experiment_name}")
            dense_model = make_linear_svm()
            dense_model.fit(train_embeddings, train_split.labels)
            start_time = time.perf_counter()
            predictions = dense_model.predict(test_embeddings).tolist()
            process_time = build_process_time(time.perf_counter() - start_time, len(split.labels))
            save_predictions(output_dir, dense_experiment_name, split, predictions)
        else:
            print(f"[Embedding] Use cached predictions for {dense_experiment_name}")

        dense_result = evaluate_predictions(
            experiment_name=dense_experiment_name,
            feature_type="description_embedding_dense",
            model_name="linear_svm",
            split_name=split.name,
            y_true=split.labels,
            y_pred=predictions,
            output_dir=output_dir,
            process_time=process_time,
        )
        results.append(dense_result)
        print(
            f"  - {split.name}: "
            f"accuracy={dense_result.accuracy:.4f}, "
            f"macro_f1={dense_result.macro_f1:.4f}, "
            f"weighted_f1={dense_result.weighted_f1:.4f}"
        )

        concat_experiment_name = "tfidf_plus_" + model_name + "_description_dense__linear_svm"
        process_time = None
        predictions = load_cached_predictions(output_dir, concat_experiment_name, split)
        if predictions is None:
            print(f"[Embedding] Training {concat_experiment_name}")
            tfidf_features = FeatureUnion(
                [
                    ("word", build_word_tfidf_vectorizer((1, 2))),
                    ("char", build_char_tfidf_vectorizer()),
                ]
            )
            train_tfidf = tfidf_features.fit_transform(train_split.texts)
            train_concat = hstack([train_tfidf, csr_matrix(train_embeddings)], format="csr")

            concat_model = make_linear_svm()
            concat_model.fit(train_concat, train_split.labels)
            start_time = time.perf_counter()
            test_tfidf = tfidf_features.transform(test_split.texts)
            test_concat = hstack([test_tfidf, csr_matrix(test_embeddings)], format="csr")
            predictions = concat_model.predict(test_concat).tolist()
            process_time = build_process_time(time.perf_counter() - start_time, len(split.labels))
            save_predictions(output_dir, concat_experiment_name, split, predictions)
        else:
            print(f"[Embedding] Use cached predictions for {concat_experiment_name}")

        concat_result = evaluate_predictions(
            experiment_name=concat_experiment_name,
            feature_type="tfidf_plus_description_embedding",
            model_name="linear_svm",
            split_name=split.name,
            y_true=split.labels,
            y_pred=predictions,
            output_dir=output_dir,
            process_time=process_time,
        )
        results.append(concat_result)
        print(
            f"  - {split.name}: "
            f"accuracy={concat_result.accuracy:.4f}, "
            f"macro_f1={concat_result.macro_f1:.4f}, "
            f"weighted_f1={concat_result.weighted_f1:.4f}"
        )

    return results


def run_ffnn_experiments(
    splits: dict[str, DatasetSplit],
    output_dir: Path,
    ckpt_root: Path = DEFAULT_CKPT_ROOT,
    embedding_dir: Path = EMBEDDING_DIR,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    checkpoint_paths = discover_ffnn_checkpoints(ckpt_root)
    if not checkpoint_paths:
        print(f"[FFNN] No checkpoints found in {ckpt_root}")
        return results

    print("[FFNN] Discovered checkpoints:")
    for checkpoint_path in checkpoint_paths:
        print(f"  - {checkpoint_path}")

    train_split = splits["train"]
    test_split = splits["test"]
    split = test_split

    from run_train import EmbeddingMLP, evaluate_loader, ensure_torch_available, make_dataloader

    ensure_torch_available()
    import torch
    from sklearn.preprocessing import LabelEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_name = str(checkpoint.get("model_name", "") or "").strip()
        if model_name == "":
            print(f"[FFNN] Ignore {checkpoint_path} - missing model_name")
            continue

        input_dim = int(checkpoint["input_dim"])
        hidden_dims = tuple(checkpoint["hidden_dims"])
        num_classes = int(checkpoint["num_classes"])
        dropout = float(checkpoint.get("dropout", 0.0))
        label_classes = checkpoint.get("label_classes", [])
        if not label_classes:
            print(f"[FFNN] Ignore {checkpoint_path} - missing label_classes")
            continue

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.asarray(label_classes)
        y_test = label_encoder.transform(test_split.labels)

        test_embeddings = load_precomputed_embeddings(
            model_name=model_name,
            split=test_split,
            normalize_embeddings=True,
            embedding_dir=embedding_dir,
        )
        if test_embeddings.shape[1] != input_dim:
            print(
                f"[FFNN] Ignore {checkpoint_path} - "
                f"embedding dim mismatch ({test_embeddings.shape[1]} != {input_dim})"
            )
            continue

        test_loader = make_dataloader(test_embeddings, y_test, batch_size=128, shuffle=False)

        model = EmbeddingMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        experiment_name = f"ffnn_{checkpoint_path.parent.name}"
        process_time: float | None = None
        predictions = load_cached_predictions(output_dir, experiment_name, split)
        if predictions is None:
            print(f"[FFNN] Evaluating {experiment_name}")
            start_time = time.perf_counter()
            _, _, test_pred = evaluate_loader(
                model=model,
                data_loader=test_loader,
                criterion=criterion,
                device=device,
            )
            predictions = label_encoder.inverse_transform(test_pred).tolist()
            process_time = build_process_time(time.perf_counter() - start_time, len(split.labels))
            save_predictions(output_dir, experiment_name, split, predictions)
        else:
            print(f"[FFNN] Use cached predictions for {experiment_name}")

        result = evaluate_predictions(
            experiment_name=experiment_name,
            feature_type="ffnn_checkpoint",
            model_name=get_model_basename(model_name),
            split_name=split.name,
            y_true=split.labels,
            y_pred=predictions,
            output_dir=output_dir,
            process_time=process_time,
        )
        results.append(result)
        print(
            f"  - {split.name}: "
            f"accuracy={result.accuracy:.4f}, "
            f"macro_f1={result.macro_f1:.4f}, "
            f"weighted_f1={result.weighted_f1:.4f}"
        )

    return results


def ensure_faiss_available() -> None:
    if faiss is None:
        raise ImportError(
            "faiss is required for search evaluation. Install faiss-cpu or faiss-gpu first."
        )


def get_search_index_path(model_name: str, index_dir: Path = INDEX_DIR) -> Path:
    model_key = sanitize_name(get_model_basename(model_name))
    return index_dir / f"{model_key}_flat.index"


def get_search_item_ids_path(model_name: str, index_dir: Path = INDEX_DIR) -> Path:
    model_key = sanitize_name(get_model_basename(model_name))
    return index_dir / f"{model_key}_flat_item_ids.json"


def load_search_index(model_name: str, index_dir: Path = INDEX_DIR) -> Any:
    ensure_faiss_available()
    index_path = get_search_index_path(model_name, index_dir)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    print(f"[Search] Load index - {index_path}")
    return faiss.read_index(str(index_path))


def load_search_item_ids(model_name: str, index_dir: Path = INDEX_DIR) -> list[str]:
    item_ids_path = get_search_item_ids_path(model_name, index_dir)
    if not item_ids_path.exists():
        raise FileNotFoundError(f"FAISS item_id mapping not found: {item_ids_path}")

    with item_ids_path.open("r", encoding="utf-8") as f:
        item_ids = json.load(f)
    if not isinstance(item_ids, list):
        raise ValueError(f"Expected a list of item_ids in {item_ids_path}")

    normalized_item_ids = [str(item_id or "").strip() for item_id in item_ids]
    if any(item_id == "" for item_id in normalized_item_ids):
        raise ValueError(f"Empty item_id found in {item_ids_path}")
    return normalized_item_ids


def build_item_id_to_label_map(split: DatasetSplit) -> dict[str, str]:
    item_id_to_label: dict[str, str] = {}
    for idx, (item_id, label) in enumerate(zip(split.item_ids, split.labels)):
        if item_id == "":
            raise ValueError(f"{split.name}.json row {idx} is missing item_id.")
        existing_label = item_id_to_label.get(item_id)
        if existing_label is not None and existing_label != label:
            raise ValueError(
                f"{split.name}.json has conflicting labels for item_id={item_id}: "
                f"{existing_label!r} != {label!r}"
            )
        item_id_to_label[item_id] = label
    return item_id_to_label


def build_item_id_to_record_map(split: DatasetSplit) -> dict[str, dict[str, str]]:
    item_id_to_record: dict[str, dict[str, str]] = {}
    for idx, (item_id, title, description, label) in enumerate(
        zip(split.item_ids, split.titles, split.descriptions, split.labels)
    ):
        if item_id == "":
            raise ValueError(f"{split.name}.json row {idx} is missing item_id.")
        item_id_to_record[item_id] = {
            "title": title,
            "description": description,
            "label": label,
        }
    return item_id_to_record


def select_label_from_neighbors(
    row_indices: np.ndarray,
    index_item_ids: list[str],
    item_id_to_label: dict[str, str],
    fallback_label: str,
) -> str:
    label_scores: dict[str, float] = defaultdict(float)
    label_best_rank: dict[str, int] = {}
    label_hit_count: dict[str, int] = defaultdict(int)

    for zero_based_rank, raw_neighbor_idx in enumerate(row_indices):
        neighbor_idx = int(raw_neighbor_idx)
        if neighbor_idx < 0 or neighbor_idx >= len(index_item_ids):
            continue

        item_id = index_item_ids[neighbor_idx]
        label = item_id_to_label.get(item_id)
        if label is None:
            continue

        rank = zero_based_rank + 1
        reciprocal_rank_score = 1.0 / (1.0 + rank)
        label_scores[label] += reciprocal_rank_score
        label_hit_count[label] += 1
        label_best_rank[label] = min(label_best_rank.get(label, rank), rank)

    if not label_scores:
        return fallback_label

    best_label = sorted(
        label_scores,
        key=lambda label: (
            -label_scores[label],
            label_best_rank[label],
            -label_hit_count[label],
            label,
        ),
    )[0]
    return best_label


def predict_labels_from_search(
    neighbor_indices: np.ndarray,
    index_item_ids: list[str],
    item_id_to_label: dict[str, str],
    fallback_label: str,
) -> list[str]:
    return [
        select_label_from_neighbors(
            row_indices=row_indices,
            index_item_ids=index_item_ids,
            item_id_to_label=item_id_to_label,
            fallback_label=fallback_label,
        )
        for row_indices in neighbor_indices
    ]


def extract_text_from_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        )
    else:
        text = str(content)
    return re.sub(r"\s+", " ", text).strip()


def normalize_predicted_label(raw_text: str, candidate_labels: list[str]) -> str | None:
    cleaned = str(raw_text or "").strip()
    if cleaned == "":
        return None

    candidate_set = set(candidate_labels)
    if cleaned in candidate_set:
        return cleaned

    stripped_candidates = [cleaned]
    stripped_candidates.append(cleaned.splitlines()[0].strip())
    stripped_candidates.append(re.sub(r"^Selected category\s*:\s*", "", cleaned, flags=re.I).strip())
    stripped_candidates.append(re.sub(r"^category\s*:\s*", "", cleaned, flags=re.I).strip())
    stripped_candidates.append(cleaned.strip("`'\" "))

    for candidate in stripped_candidates:
        if candidate in candidate_set:
            return candidate

    normalized_target = re.sub(r"\s+", "", cleaned).lower()
    for candidate in candidate_labels:
        if re.sub(r"\s+", "", candidate).lower() == normalized_target:
            return candidate
    return None


def build_candidate_categories_text(labels: list[str]) -> str:
    unique_labels = sorted(set(labels))
    return "\n".join(f"- {label}" for label in unique_labels)


def build_rag_examples(
    row_indices: np.ndarray,
    index_item_ids: list[str],
    item_id_to_record: dict[str, dict[str, str]],
    max_examples: int = 3,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    seen_item_ids: set[str] = set()

    for raw_neighbor_idx in row_indices:
        neighbor_idx = int(raw_neighbor_idx)
        if neighbor_idx < 0 or neighbor_idx >= len(index_item_ids):
            continue
        item_id = index_item_ids[neighbor_idx]
        if item_id in seen_item_ids:
            continue
        seen_item_ids.add(item_id)
        record = item_id_to_record.get(item_id)
        if record is None:
            continue
        examples.append(record)
        if len(examples) >= max_examples:
            break

    while len(examples) < max_examples:
        examples.append({"title": "", "description": "", "label": ""})
    return examples


def run_rag_experiments(
    splits: dict[str, DatasetSplit],
    output_dir: Path,
    normalize_embeddings: bool,
    embedding_dir: Path = EMBEDDING_DIR,
    index_dir: Path = INDEX_DIR,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    rag_model_names = discover_search_models(index_dir=index_dir, embedding_dir=embedding_dir)
    if not rag_model_names:
        print(
            f"[RAG] No complete search assets found in {index_dir} "
            f"with matching test embedding caches in {embedding_dir}"
        )
        return results

    print("[RAG] Discovered models:")
    for model_name in rag_model_names:
        print(f"  - {model_name}")

    from prompt import build_llm_client
    from prompt.classify import PROMPT

    train_split = splits["train"]
    test_split = splits["test"]
    split = test_split
    item_id_to_label = build_item_id_to_label_map(train_split)
    item_id_to_record = build_item_id_to_record_map(train_split)
    fallback_label = train_split.labels[0]
    candidate_labels = sorted(set(train_split.labels))
    candidate_categories_text = build_candidate_categories_text(train_split.labels)
    llm_model = build_llm_client()

    for rag_model_name in rag_model_names:
        print(f"[RAG] Run experiments for {rag_model_name}")
        try:
            index = load_search_index(rag_model_name, index_dir=index_dir)
            index_item_ids = load_search_item_ids(rag_model_name, index_dir=index_dir)
        except Exception as exc:
            print(f"[RAG] Ignore {rag_model_name} - {exc}")
            continue

        if int(index.ntotal) != len(index_item_ids):
            print(
                f"[RAG] Ignore {rag_model_name} - "
                f"index size mismatch ({int(index.ntotal)} != {len(index_item_ids)})"
            )
            continue

        test_embeddings = load_precomputed_embeddings(
            model_name=rag_model_name,
            split=test_split,
            normalize_embeddings=normalize_embeddings,
            embedding_dir=embedding_dir,
        )
        if getattr(index, "d", None) is not None and int(index.d) != int(test_embeddings.shape[1]):
            print(
                f"[RAG] Ignore {rag_model_name} - "
                f"embedding dim mismatch ({test_embeddings.shape[1]} != {int(index.d)})"
            )
            continue

        effective_top_k = min(3, len(index_item_ids))
        if effective_top_k <= 0:
            print(f"[RAG] Ignore {rag_model_name} - no searchable vectors")
            continue

        experiment_name = f"{get_model_basename(rag_model_name)}_faiss_rag_top{effective_top_k}"
        process_time: float | None = None
        predictions = load_cached_predictions(output_dir, experiment_name, split)
        if predictions is None:
            print(f"[RAG] Retrieving neighbors for {experiment_name}")
            start_time = time.perf_counter()
            neighbor_indices = search_index_with_progress(
                index=index,
                query_embeddings=test_embeddings,
                top_k=effective_top_k,
                batch_size=1024,
                desc=f"[RAG-Search] {get_model_basename(rag_model_name)}",
            )

            predictions = []
            for row_idx in tqdm(range(len(split.labels)), desc=f"[RAG-LLM] {get_model_basename(rag_model_name)}", unit="item"):
                row_indices = neighbor_indices[row_idx]
                rag_examples = build_rag_examples(
                    row_indices=row_indices,
                    index_item_ids=index_item_ids,
                    item_id_to_record=item_id_to_record,
                    max_examples=3,
                )
                target_title = split.titles[row_idx]
                target_description = split.descriptions[row_idx]

                fallback_prediction = select_label_from_neighbors(
                    row_indices=row_indices,
                    index_item_ids=index_item_ids,
                    item_id_to_label=item_id_to_label,
                    fallback_label=fallback_label,
                )

                try:
                    prompt = PROMPT.format_messages(
                        candidate_categories=candidate_categories_text,
                        example1_title=rag_examples[0]["title"],
                        example1_description=rag_examples[0]["description"],
                        example1_category=rag_examples[0]["label"],
                        example2_title=rag_examples[1]["title"],
                        example2_description=rag_examples[1]["description"],
                        example2_category=rag_examples[1]["label"],
                        example3_title=rag_examples[2]["title"],
                        example3_description=rag_examples[2]["description"],
                        example3_category=rag_examples[2]["label"],
                        title=target_title,
                        description=target_description,
                    )
                    response = llm_model.invoke(prompt)
                    predicted_label = normalize_predicted_label(
                        extract_text_from_response(response),
                        candidate_labels=candidate_labels,
                    )
                except Exception:
                    predicted_label = None

                predictions.append(predicted_label or fallback_prediction)

            process_time = build_process_time(time.perf_counter() - start_time, len(split.labels))
            save_predictions(output_dir, experiment_name, split, predictions)
        else:
            print(f"[RAG] Use cached predictions for {experiment_name}")

        result = evaluate_predictions(
            experiment_name=experiment_name,
            feature_type="faiss_rag_llm",
            model_name=get_model_basename(rag_model_name),
            split_name=split.name,
            y_true=split.labels,
            y_pred=predictions,
            output_dir=output_dir,
            process_time=process_time,
        )
        results.append(result)
        print(
            f"  - {split.name}: "
            f"accuracy={result.accuracy:.4f}, "
            f"macro_f1={result.macro_f1:.4f}, "
            f"weighted_f1={result.weighted_f1:.4f}"
        )

    return results


def search_index_with_progress(
    index: Any,
    query_embeddings: np.ndarray,
    top_k: int,
    batch_size: int = 128,
    desc: str = "[Search] Querying index",
) -> np.ndarray:
    if query_embeddings.ndim != 2:
        raise ValueError(f"query_embeddings must be 2D, got shape={query_embeddings.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    all_neighbor_indices: list[np.ndarray] = []
    total_queries = int(query_embeddings.shape[0])

    for start in tqdm(range(0, total_queries, batch_size), desc=desc, unit="batch"):
        end = min(start + batch_size, total_queries)
        batch_queries = np.ascontiguousarray(query_embeddings[start:end], dtype=np.float32)
        _, batch_neighbor_indices = index.search(batch_queries, top_k)
        all_neighbor_indices.append(batch_neighbor_indices)

    if not all_neighbor_indices:
        return np.empty((0, top_k), dtype=np.int64)
    return np.vstack(all_neighbor_indices)


def run_search_experiments(
    splits: dict[str, DatasetSplit],
    output_dir: Path,
    normalize_embeddings: bool,
    embedding_dir: Path = EMBEDDING_DIR,
    index_dir: Path = INDEX_DIR,
    top_k: int = 10,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    search_model_names = discover_search_models(index_dir=index_dir, embedding_dir=embedding_dir)
    if not search_model_names:
        print(
            f"[Search] No complete search assets found in {index_dir} "
            f"with matching test embedding caches in {embedding_dir}"
        )
        return results

    print("[Search] Discovered models:")
    for model_name in search_model_names:
        print(f"  - {model_name}")

    train_split = splits["train"]
    test_split = splits["test"]
    split = test_split
    item_id_to_label = build_item_id_to_label_map(train_split)
    fallback_label = train_split.labels[0]

    for search_model_name in search_model_names:
        print(f"[Search] Run experiments for {search_model_name}")
        try:
            index = load_search_index(search_model_name, index_dir=index_dir)
            index_item_ids = load_search_item_ids(search_model_name, index_dir=index_dir)
        except Exception as exc:
            print(f"[Search] Ignore {search_model_name} - {exc}")
            continue

        if int(index.ntotal) != len(index_item_ids):
            print(
                f"[Search] Ignore {search_model_name} - "
                f"index size mismatch ({int(index.ntotal)} != {len(index_item_ids)})"
            )
            continue

        test_embeddings = load_precomputed_embeddings(
            model_name=search_model_name,
            split=test_split,
            normalize_embeddings=normalize_embeddings,
            embedding_dir=embedding_dir,
        )
        if getattr(index, "d", None) is not None and int(index.d) != int(test_embeddings.shape[1]):
            print(
                f"[Search] Ignore {search_model_name} - "
                f"embedding dim mismatch ({test_embeddings.shape[1]} != {int(index.d)})"
            )
            continue

        effective_top_k = min(top_k, len(index_item_ids))
        if effective_top_k <= 0:
            print(f"[Search] Ignore {search_model_name} - no searchable vectors")
            continue

        experiment_name = (
            f"{get_model_basename(search_model_name)}_faiss_search_rr_top{effective_top_k}"
        )
        process_time: float | None = None
        predictions = load_cached_predictions(output_dir, experiment_name, split)
        if predictions is None:
            print(f"[Search] Evaluating {experiment_name}")
            start_time = time.perf_counter()
            neighbor_indices = search_index_with_progress(
                index=index,
                query_embeddings=test_embeddings,
                top_k=effective_top_k,
                batch_size=1024,
                desc=f"[Search] {get_model_basename(search_model_name)}",
            )
            predictions = predict_labels_from_search(
                neighbor_indices=neighbor_indices,
                index_item_ids=index_item_ids,
                item_id_to_label=item_id_to_label,
                fallback_label=fallback_label,
            )
            process_time = build_process_time(time.perf_counter() - start_time, len(split.labels))
            save_predictions(output_dir, experiment_name, split, predictions)
        else:
            print(f"[Search] Use cached predictions for {experiment_name}")

        result = evaluate_predictions(
            experiment_name=experiment_name,
            feature_type="faiss_search_reciprocal_rank",
            model_name=get_model_basename(search_model_name),
            split_name=split.name,
            y_true=split.labels,
            y_pred=predictions,
            output_dir=output_dir,
            process_time=process_time,
        )
        results.append(result)
        print(
            f"  - {split.name}: "
            f"accuracy={result.accuracy:.4f}, "
            f"macro_f1={result.macro_f1:.4f}, "
            f"weighted_f1={result.weighted_f1:.4f}"
        )

    return results


def save_results_summary(results: list[ExperimentResult], output_dir: Path) -> None:
    summary_path = output_dir / "metrics_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_name",
                "feature_type",
                "model_name",
                "split_name",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_dict())

    grouped: dict[str, dict[str, float | str]] = {}
    for result in results:
        if result.split_name != "valid":
            continue
        grouped[result.experiment_name] = {
            "experiment_name": result.experiment_name,
            "feature_type": result.feature_type,
            "model_name": result.model_name,
            "valid_accuracy": result.accuracy,
            "valid_macro_f1": result.macro_f1,
            "valid_weighted_f1": result.weighted_f1,
        }

    for result in results:
        if result.split_name != "test":
            continue
        if result.experiment_name not in grouped:
            grouped[result.experiment_name] = {
                "experiment_name": result.experiment_name,
                "feature_type": result.feature_type,
                "model_name": result.model_name,
            }
        grouped[result.experiment_name]["test_accuracy"] = result.accuracy
        grouped[result.experiment_name]["test_macro_f1"] = result.macro_f1
        grouped[result.experiment_name]["test_weighted_f1"] = result.weighted_f1

    ranking = sorted(
        grouped.values(),
        key=lambda x: float(x.get("valid_macro_f1", -1.0)),
        reverse=True,
    )
    ranking_path = output_dir / "experiment_ranking.json"
    with ranking_path.open("w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate category classifiers on parsing/train.json, valid.json, and test.json."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing train.json, valid.json, and test.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for TF-IDF experiment outputs.",
    )
    parser.add_argument(
        "--embedding-output-dir",
        type=Path,
        default=DEFAULT_EMBEDDING_OUTPUT_DIR,
        help="Directory for embedding experiment outputs.",
    )
    parser.add_argument(
        "--use-tfidf",
        action="store_true",
        help="Run TF-IDF experiments only when this flag is selected, or by default when no use flags are given.",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Run embedding experiments only when this flag is selected, or by default when no use flags are given.",
    )
    parser.add_argument(
        "--use-ffnn",
        action="store_true",
        help="Evaluate saved FFNN checkpoints from outputs/ckpt on the test split.",
    )
    parser.add_argument(
        "--use-search",
        action="store_true",
        help="Evaluate FAISS retrieval-based label voting on the test split.",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Evaluate FAISS retrieval + LLM category selection on the test split.",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="models/KoE5",
        help="Deprecated. Embedding runs now auto-discover models from pickle/embedding.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable L2 normalization when generating sentence embeddings.",
    )
    parser.add_argument(
        "--ffnn-output-dir",
        type=Path,
        default=DEFAULT_FFNN_OUTPUT_DIR,
        help="Directory for FFNN checkpoint evaluation outputs.",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=DEFAULT_CKPT_ROOT,
        help="Root directory containing FFNN checkpoint folders.",
    )
    parser.add_argument(
        "--search-output-dir",
        type=Path,
        default=DEFAULT_SEARCH_OUTPUT_DIR,
        help="Directory for FAISS search evaluation outputs.",
    )
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=10,
        help="Top-k neighbors used for reciprocal-rank label voting.",
    )
    parser.add_argument(
        "--rag-output-dir",
        type=Path,
        default=DEFAULT_RAG_OUTPUT_DIR,
        help="Directory for FAISS RAG evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.embedding_output_dir.mkdir(parents=True, exist_ok=True)
    args.ffnn_output_dir.mkdir(parents=True, exist_ok=True)
    args.search_output_dir.mkdir(parents=True, exist_ok=True)
    args.rag_output_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits(args.data_dir)
    all_results: list[ExperimentResult] = []
    run_tfidf = args.use_tfidf
    run_embeddings = args.use_embeddings
    run_ffnn = args.use_ffnn
    run_search = args.use_search
    run_rag = args.use_rag
    if not run_tfidf and not run_embeddings and not run_ffnn and not run_search and not run_rag:
        run_tfidf = True
        run_embeddings = False
        run_ffnn = False
        run_search = False
        run_rag = False

    data_overview = {
        split_name: {
            "num_rows": len(split.labels),
            "num_classes": len(set(split.labels)),
            "empty_descriptions": int(sum(desc.strip() == "" for desc in split.descriptions)),
        }
        for split_name, split in splits.items()
    }
    common_data_overview_path = OUTPUTS_DIR / "data_overview.json"
    if not common_data_overview_path.exists():
        common_data_overview_path.parent.mkdir(parents=True, exist_ok=True)
        with common_data_overview_path.open("w", encoding="utf-8") as f:
            json.dump(data_overview, f, ensure_ascii=False, indent=2)

    if run_tfidf:
        all_results.extend(run_tfidf_experiments(splits=splits, output_dir=args.output_dir))

    if run_embeddings:
        all_results.extend(
            run_embedding_experiments(
                splits=splits,
                output_dir=args.embedding_output_dir,
                normalize_embeddings=not args.no_normalize_embeddings,
                embedding_dir=EMBEDDING_DIR,
            )
        )
    if run_ffnn:
        all_results.extend(
            run_ffnn_experiments(
                splits=splits,
                output_dir=args.ffnn_output_dir,
                ckpt_root=args.ckpt_root,
                embedding_dir=EMBEDDING_DIR,
            )
        )
    if run_search:
        all_results.extend(
            run_search_experiments(
                splits=splits,
                output_dir=args.search_output_dir,
                normalize_embeddings=not args.no_normalize_embeddings,
                embedding_dir=EMBEDDING_DIR,
                index_dir=INDEX_DIR,
                top_k=args.search_top_k,
            )
        )
    if run_rag:
        all_results.extend(
            run_rag_experiments(
                splits=splits,
                output_dir=args.rag_output_dir,
                normalize_embeddings=not args.no_normalize_embeddings,
                embedding_dir=EMBEDDING_DIR,
                index_dir=INDEX_DIR,
            )
        )

    tfidf_results = [result for result in all_results if result.artifact_dir == str(args.output_dir)]
    embedding_results = [result for result in all_results if result.artifact_dir == str(args.embedding_output_dir)]
    ffnn_results = [result for result in all_results if result.artifact_dir == str(args.ffnn_output_dir)]
    search_results = [result for result in all_results if result.artifact_dir == str(args.search_output_dir)]
    rag_results = [result for result in all_results if result.artifact_dir == str(args.rag_output_dir)]
    if tfidf_results:
        save_results_summary(tfidf_results, args.output_dir)
    if embedding_results:
        save_results_summary(embedding_results, args.embedding_output_dir)
    if ffnn_results:
        save_results_summary(ffnn_results, args.ffnn_output_dir)
    if search_results:
        save_results_summary(search_results, args.search_output_dir)
    if rag_results:
        save_results_summary(rag_results, args.rag_output_dir)

    if all_results:
        test_results = [result for result in all_results if result.split_name == "test"]
        best = max(test_results, key=lambda result: result.macro_f1)
        print(
            "\nBest test experiment: "
            f"{best.experiment_name} "
            f"(accuracy={best.accuracy:.4f}, macro_f1={best.macro_f1:.4f}, weighted_f1={best.weighted_f1:.4f})"
        )
        best_output_dir = Path(best.artifact_dir) if best.artifact_dir else args.output_dir
        print_per_class_metrics(best_output_dir, best)
        if tfidf_results:
            print(f"Saved TF-IDF outputs to: {args.output_dir}")
        if embedding_results:
            print(f"Saved embedding outputs to: {args.embedding_output_dir}")
        if ffnn_results:
            print(f"Saved FFNN outputs to: {args.ffnn_output_dir}")
        if search_results:
            print(f"Saved search outputs to: {args.search_output_dir}")
        if rag_results:
            print(f"Saved RAG outputs to: {args.rag_output_dir}")
    else:
        print("No experiments were executed. Check the CLI flags.")


if __name__ == "__main__":
    main()
