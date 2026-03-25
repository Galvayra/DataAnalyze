from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from classify_categories import build_input_text


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "parsing"
EMBEDDING_DIR = ROOT_DIR / "pickle" / "embedding"
INDEX_DIR = ROOT_DIR / "index"
DEFAULT_MODEL_NAME = "models/KoE5"
INDEX_TYPE = "flat"
MAX_SEQUENCE_LENGTH = 512


def ensure_dependencies() -> None:
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for indexing. "
            "Install dependencies in the target server environment first."
        )
    if faiss is None:
        raise ImportError(
            "faiss is required for indexing. "
            "Install faiss-cpu or faiss-gpu in the target server environment first."
        )


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "default"


def get_model_basename(model_name: str) -> str:
    return Path(model_name).name or sanitize_name(model_name)


def requires_e5_prompt(model_name: str) -> bool:
    return "e5" in model_name.lower()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(data)!r}")
    return data


def format_embedding_text(text: str, model_name: str) -> str:
    text = text.strip()
    if requires_e5_prompt(model_name):
        return f"passage: {text}"
    return text


def build_texts(records: list[dict[str, Any]], model_name: str) -> list[str]:
    texts: list[str] = []
    for row in records:
        title = str(row.get("title", "") or "")
        description = str(row.get("description", "") or "")
        text = build_input_text(title, description)
        texts.append(format_embedding_text(text, model_name))
    return texts


def build_item_ids(records: list[dict[str, Any]]) -> list[str]:
    item_ids: list[str] = []
    for idx, row in enumerate(records):
        item_id = str(row.get("item_id", "") or "").strip()
        if item_id == "":
            raise ValueError(f"Record {idx} is missing item_id.")
        item_ids.append(item_id)
    return item_ids


def get_embedding_cache_path(model_name: str, split_name: str) -> Path:
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    model_basename = sanitize_name(get_model_basename(model_name))
    return EMBEDDING_DIR / f"{model_basename}_{split_name}.pkl"


def load_cached_embeddings(
    cache_path: Path,
    model_name: str,
    texts: list[str],
    item_ids: list[str],
    normalize_embeddings: bool,
) -> np.ndarray | None:
    if not cache_path.exists():
        return None

    with cache_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        return None

    cached_model_name = payload.get("model_name")
    cached_num_texts = payload.get("num_texts")
    cached_normalize = payload.get("normalize_embeddings")
    cached_use_e5_prompt = payload.get("use_e5_prompt")
    cached_max_sequence_length = payload.get("max_sequence_length")
    cached_item_ids = payload.get("item_ids")
    embeddings = payload.get("embeddings")

    if cached_model_name != model_name:
        print(f"[Embedding] Ignore cache {cache_path.name} - model mismatch")
        return None
    if cached_num_texts != len(texts):
        print(f"[Embedding] Ignore cache {cache_path.name} - row count mismatch")
        return None
    if cached_item_ids != item_ids:
        print(f"[Embedding] Ignore cache {cache_path.name} - item_id order mismatch")
        return None
    if cached_normalize != normalize_embeddings:
        print(f"[Embedding] Ignore cache {cache_path.name} - normalize flag mismatch")
        return None
    if cached_use_e5_prompt != requires_e5_prompt(model_name):
        print(f"[Embedding] Ignore cache {cache_path.name} - prompt format mismatch")
        return None
    if cached_max_sequence_length != MAX_SEQUENCE_LENGTH:
        print(f"[Embedding] Ignore cache {cache_path.name} - max sequence length mismatch")
        return None
    if not isinstance(embeddings, np.ndarray):
        return None

    print(f"[Embedding] Load cache - {cache_path}")
    return embeddings.astype(np.float32, copy=False)


def save_embeddings_cache(
    cache_path: Path,
    model_name: str,
    texts: list[str],
    item_ids: list[str],
    embeddings: np.ndarray,
    normalize_embeddings: bool,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "num_texts": len(texts),
        "item_ids": item_ids,
        "normalize_embeddings": normalize_embeddings,
        "use_e5_prompt": requires_e5_prompt(model_name),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
    }
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Embedding] Save cache - {cache_path}")


def load_or_create_embeddings(
    model: SentenceTransformer,
    split_name: str,
    texts: list[str],
    item_ids: list[str],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    cache_path = get_embedding_cache_path(model_name, split_name)
    cached_embeddings = load_cached_embeddings(
        cache_path=cache_path,
        model_name=model_name,
        texts=texts,
        item_ids=item_ids,
        normalize_embeddings=normalize_embeddings,
    )
    if cached_embeddings is not None:
        return cached_embeddings

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    save_embeddings_cache(
        cache_path=cache_path,
        model_name=model_name,
        texts=texts,
        item_ids=item_ids,
        embeddings=embeddings,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings


def get_index_path(model_name: str, index_type: str = INDEX_TYPE) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_DIR / f"{sanitize_name(get_model_basename(model_name))}_{index_type}.index"


def get_index_meta_path(model_name: str, index_type: str = INDEX_TYPE) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_DIR / f"{sanitize_name(get_model_basename(model_name))}_{index_type}.json"


def get_index_item_ids_path(model_name: str, index_type: str = INDEX_TYPE) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_DIR / f"{sanitize_name(get_model_basename(model_name))}_{index_type}_item_ids.json"


def save_index_item_ids(model_name: str, item_ids: list[str], index_type: str = INDEX_TYPE) -> Path:
    item_ids_path = get_index_item_ids_path(model_name, index_type)
    with item_ids_path.open("w", encoding="utf-8") as f:
        json.dump(item_ids, f, ensure_ascii=False, indent=2)
    print(f"[FAISS] Save item_id mapping - {item_ids_path}")
    return item_ids_path


def build_or_load_flat_index(model_name: str, embeddings_train: np.ndarray, item_ids: list[str]) -> Any:
    ensure_dependencies()
    index_path = get_index_path(model_name)
    meta_path = get_index_meta_path(model_name)
    save_index_item_ids(model_name, item_ids)

    if len(item_ids) != int(embeddings_train.shape[0]):
        raise ValueError(
            f"item_id count mismatch: item_ids={len(item_ids)} != embeddings={embeddings_train.shape[0]}"
        )

    if index_path.exists():
        print(f"[FAISS] Load index - {index_path}")
        return faiss.read_index(str(index_path))

    dimension = int(embeddings_train.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(embeddings_train, dtype=np.float32))
    faiss.write_index(index, str(index_path))
    print(f"[FAISS] Save index - {index_path}")

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "index_type": INDEX_TYPE,
                "metric": "inner_product",
                "num_vectors": int(embeddings_train.shape[0]),
                "dimension": dimension,
                "item_id_mapping_path": str(get_index_item_ids_path(model_name)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[FAISS] Save index meta - {meta_path}")
    return index


def print_model_info(model: SentenceTransformer, model_name: str, batch_size: int) -> None:
    try:
        dimension = model.get_sentence_embedding_dimension()
    except Exception:
        dimension = "unknown"

    print("Model info -")
    print(f"  model_name: {model_name}")
    print(f"  max_seq_length: {getattr(model, 'max_seq_length', 'unknown')}")
    print(f"  dimension: {dimension}")
    print(f"  use_passage_prompt: {requires_e5_prompt(model_name)}")
    print(f"  batch_size: {batch_size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SentenceTransformer embeddings and a FAISS flat index from parsing splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing train.json, valid.json, and test.json.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model path or model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable embedding normalization before indexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dependencies()

    train_records = load_records(args.data_dir / "train.json")
    valid_records = load_records(args.data_dir / "valid.json")
    test_records = load_records(args.data_dir / "test.json")

    train_texts = build_texts(train_records, args.model_name)
    valid_texts = build_texts(valid_records, args.model_name)
    test_texts = build_texts(test_records, args.model_name)
    train_item_ids = build_item_ids(train_records)
    valid_item_ids = build_item_ids(valid_records)
    test_item_ids = build_item_ids(test_records)

    model = SentenceTransformer(args.model_name)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    normalize_embeddings = not args.no_normalize_embeddings
    print_model_info(model, args.model_name, args.batch_size)

    embeddings_train = load_or_create_embeddings(
        model=model,
        split_name="train",
        texts=train_texts,
        item_ids=train_item_ids,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings_valid = load_or_create_embeddings(
        model=model,
        split_name="valid",
        texts=valid_texts,
        item_ids=valid_item_ids,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings_test = load_or_create_embeddings(
        model=model,
        split_name="test",
        texts=test_texts,
        item_ids=test_item_ids,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    build_or_load_flat_index(args.model_name, embeddings_train, train_item_ids)

    print(
        "Embedding shapes -",
        f"train: {embeddings_train.shape}",
        f"valid: {embeddings_valid.shape}",
        f"test: {embeddings_test.shape}",
    )


if __name__ == "__main__":
    main()
