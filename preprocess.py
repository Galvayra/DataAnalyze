from __future__ import annotations

import argparse
from html import unescape
import json
import pickle
from pathlib import Path
import random
import re
from typing import Any
from prompt import build_llm_client
import pandas as pd


ITEMS_CSV_PATH = Path("datasets/category/items.csv")
ITEMS_PICKLE_PATH = Path("pickle/items.pkl")
REPRESENT_ITEMS_PICKLE_PATH = Path("pickle/represent_items.pkl")
CATEGORIES_JSON_PATH = Path("parsing/categories.json")
TRAIN_JSON_PATH = Path("parsing/train.json")
VALID_JSON_PATH = Path("parsing/valid.json")
TEST_JSON_PATH = Path("parsing/test.json")
RANDOM_SEED = 42
REPRESENT_SHOW_INTERVAL = 100
MIN_LENGTH_DESCRIPTION = 10


def remove_duplicate_categories(
    item_dict: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], bool]:
    """Remove duplicate lower-level categories in hierarchical order."""
    changed = False

    for item in item_dict.values():
        cat_1 = (item.get("cat_1") or "").strip()
        cat_2 = (item.get("cat_2") or "").strip()
        cat_3 = (item.get("cat_3") or "").strip()

        item["cat_1"] = cat_1
        item["cat_2"] = cat_2
        item["cat_3"] = cat_3

        if cat_2 and cat_2 == cat_3:
            item["cat_3"] = ""
            cat_3 = ""
            changed = True

        if cat_1 and cat_1 == cat_2:
            item["cat_2"] = ""
            changed = True

    return item_dict, changed


def build_item_dict(csv_path: str | Path = ITEMS_CSV_PATH) -> dict[str, dict[str, str]]:
    """Build an item metadata dictionary keyed by item_id."""
    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        dtype=str,
        usecols=["item_id", "name", "description", "cat_1", "cat_2", "cat_3"],
        keep_default_na=False,
    )
    df["item_id"] = df["item_id"].str.strip()
    df = df[df["item_id"] != ""]

    item_dict = df.set_index("item_id").to_dict(orient="index")
    item_dict, _ = remove_duplicate_categories(item_dict)
    return item_dict


def extract_text_from_response(response: Any) -> str:
    """Extract plain text content from a LangChain chat model response."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                if isinstance(chunk.get("text"), str):
                    parts.append(chunk["text"])
                elif chunk.get("type") == "text" and isinstance(chunk.get("value"), str):
                    parts.append(chunk["value"])
        text = " ".join(parts)
    else:
        text = str(content)

    text = text.strip()
    text = re.sub(r"^new_description\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def generate_represent_text(llm_model: Any, title: str, description: str) -> str:
    """Generate a rewritten description from the raw title and description."""
    from prompt.represent import PROMPT

    response = llm_model.invoke(
        PROMPT.format_messages(
            title=(title or "").strip(),
            description=(description or "").strip(),
        )
    )
    return extract_text_from_response(response)


def append_item_represent(
    item_dict: dict[str, dict[str, str]],
    pickle_path: str | Path,
    show_interval: int = REPRESENT_SHOW_INTERVAL,
) -> dict[str, dict[str, str]]:
    """Populate `represent` for items with non-empty descriptions and cache progress."""
    from tqdm import tqdm

    llm_model = build_llm_client()
    represent_count = 0
    no_description_count = 0
    pickle_path = Path(pickle_path)

    if pickle_path.exists():
        item_dict = load_item_dict(pickle_path)
        # item_dict, changed = remove_duplicate_categories(item_dict)
        # if changed:
        #     save_item_dict(item_dict, pickle_path)
        return item_dict

    represent_list = []
    for item_id, item in item_dict.items():
        description = (item.get("description") or "").strip()
        item_dict[item_id]["represent"] = ""
        if description and len(description) > MIN_LENGTH_DESCRIPTION:
            represent_list.append(item_id)

    for item_id in tqdm(represent_list, desc="generate represent"):
        item = item_dict[item_id]
        description = (item.get("description") or "").strip()

        try:
            represent = generate_represent_text(
                llm_model=llm_model,
                title=item.get("name", ""),
                description=description,
            )
        except:
            represent = clean_description(description)
        else:
            represent_count += 1
        item["represent"] = represent
        # if show_interval % (index + 1) == 0:
        #     print(represent_count, index + 1))
        # print(represent, represent_count)

    print(
        "Saved representation -",
        f"represent count: {represent_count}",
        f" no description: {no_description_count}",
    )
    save_item_dict(item_dict, pickle_path)

    return item_dict


def save_item_dict(
    item_dict: dict[str, dict[str, str]],
    output_path: str | Path = ITEMS_PICKLE_PATH,
) -> Path:
    """Save item metadata dictionary to a pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(item_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save pickle -", output_path)

    return output_path


def load_item_dict(pickle_path: str | Path = ITEMS_PICKLE_PATH) -> dict[str, dict[str, str]]:
    """Load item metadata dictionary from a pickle file."""
    pickle_path = Path(pickle_path)

    with pickle_path.open("rb") as f:
        print("Load pickle -", pickle_path)
        return pickle.load(f)


def get_item_dict(
    csv_path: str | Path = ITEMS_CSV_PATH,
    pickle_path: str | Path = ITEMS_PICKLE_PATH,
) -> dict[str, dict[str, str]]:
    """Load from pickle when available, otherwise build and save it first."""
    pickle_path = Path(pickle_path)

    if pickle_path.exists():
        item_dict = load_item_dict(pickle_path)
        # item_dict, changed = remove_duplicate_categories(item_dict)
        # if changed:
        #     save_item_dict(item_dict, pickle_path)
        return item_dict

    item_dict = build_item_dict(csv_path)
    save_item_dict(item_dict, pickle_path)
    return item_dict


def build_category_dict(
    item_dict: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """Build sorted unique category lists for each hierarchy level."""
    return {
        "cat_1": sorted(
            {
                (item.get("cat_1") or "").strip()
                for item in item_dict.values()
                if (item.get("cat_1") or "").strip()
            }
        ),
        "cat_2": sorted(
            {
                (item.get("cat_2") or "").strip()
                for item in item_dict.values()
                if (item.get("cat_2") or "").strip()
            }
        ),
        "cat_3": sorted(
            {
                (item.get("cat_3") or "").strip()
                for item in item_dict.values()
                if (item.get("cat_3") or "").strip()
            }
        ),
    }


def save_category_dict(
    category_dict: dict[str, list[str]],
    output_path: str | Path = CATEGORIES_JSON_PATH,
) -> Path:
    """Save category lists as UTF-8 JSON if the file does not exist."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print("Skip json -", output_path)
        return output_path

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(category_dict, f, ensure_ascii=False, indent=2)
        print("Save json -", output_path)

    return output_path


def normalize_category_term(text: str) -> str:
    """Normalize category text for matching."""
    return re.sub(r"\s+", "", (text or "").strip().lower())


def build_category_term_set(category_dict: dict[str, list[str]]) -> set[str]:
    """Flatten all category labels into a normalized lookup set."""
    return {
        normalize_category_term(category)
        for categories in category_dict.values()
        for category in categories
        if normalize_category_term(category)
    }


def remove_non_category_brackets(name: str, category_terms: set[str]) -> str:
    """Keep bracket contents only when they contain a known category term."""

    def replace_parentheses(match: re.Match[str]) -> str:
        inner = match.group(1)
        normalized_inner = normalize_category_term(inner)
        if any(term in normalized_inner for term in category_terms):
            return f" {inner} "
        return " "

    def replace_square_brackets(match: re.Match[str]) -> str:
        inner = match.group(1)
        normalized_inner = normalize_category_term(inner)
        if any(term in normalized_inner for term in category_terms):
            return f" {inner} "
        return " "

    name = re.sub(r"\(([^)]*)\)", replace_parentheses, name)
    name = re.sub(r"\[([^\]]*)\]", replace_square_brackets, name)
    return name


def clean_title(name: str, category_terms: set[str]) -> str:
    """Normalize product names for classification."""
    title = remove_non_category_brackets((name or "").strip(), category_terms)
    title = title.lower()
    title = re.sub(r"\b[a-z]{1,4}\d{3,}[a-z]*\b", " ", title)
    title = re.sub(r"[/&\\\-_+#!]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def clean_description(description: str) -> str:
    """Strip HTML and normalize whitespace in descriptions."""
    text = (description or "").lower()
    text = unescape(text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", " ", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<?img\s+src\s*=\s*['\"][^'\"]*['\"][^>]*>?", " ", text, flags=re.I)
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<(meta|link)[^>]*>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[♥♡❤🖤🤍●■◆▲▼★☆!]", " ", text)
    text = re.sub(r"\b(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:/[^\s]*)?\b", " ", text)
    text = re.sub(r"([^\d\s])\1{1,}", " ", text)
    text = re.sub(r"[\[\]\(\)]", " ", text)
    text = re.sub(r"[:\)#/\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_parsed_records(
    item_dict: dict[str, dict[str, str]],
    category_dict: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Build parsed records using cat_1 as the label."""
    records: list[dict[str, str]] = []
    category_terms = build_category_term_set(category_dict)

    for item_id, item in item_dict.items():
        title = clean_title(item.get("name", ""), category_terms)
        label = (item.get("cat_1") or "").strip()
        assert label != "", "label(cat_1) is not Null"

        description = item.get("description", "").strip()
        represent = item.get("represent", "").strip()
        records.append(
            {
                "item_id": item_id,
                "title": title,
                "description": clean_description(description),
                "represent": represent,
                "label": label,
            }
        )

    return records


def split_records(
    records: list[dict[str, str]],
    seed: int = RANDOM_SEED,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split all products globally into train/valid/test with 8:1:1 ratio."""
    rng = random.Random(seed)
    shuffled_records = sorted(records, key=lambda record: record["item_id"])
    rng.shuffle(shuffled_records)

    total_count = len(shuffled_records)
    train_count = int(total_count * 0.8)
    valid_count = int(total_count * 0.1)

    train_end = train_count
    valid_end = train_end + valid_count

    train_records = shuffled_records[:train_end]
    valid_records = shuffled_records[train_end:valid_end]
    test_records = shuffled_records[valid_end:]
    return train_records, valid_records, test_records


def save_json_records(
    records: list[dict[str, str]],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Save parsed records as UTF-8 JSON if the file does not exist."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print("Skip json -", output_path)
        return output_path

    export_records = [
        {
            "item_id": record["item_id"],
            "title": record["title"],
            "description": record["description"],
            "represent": record.get("represent", ""),
            "label": record["label"],
        }
        for record in records
    ]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(export_records, f, ensure_ascii=False, indent=2)
        print("Save json -", output_path)

    return output_path


def load_json_records(input_path: str | Path) -> list[dict[str, str]]:
    """Load parsed records from a saved UTF-8 JSON file."""
    input_path = Path(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    print("Load json -", input_path)
    return records


def attach_represent_to_records(
    records: list[dict[str, str]],
    represent_lookup: dict[tuple[str, str, str], str],
) -> list[dict[str, str]]:
    """Attach cached represent text to already-saved split records."""
    enriched_records: list[dict[str, str]] = []

    for record in records:
        key = (
            record.get("title", ""),
            record.get("description", ""),
            record.get("label", ""),
        )
        enriched = dict(record)
        enriched["represent"] = represent_lookup.get(key, record.get("represent", ""))
        enriched_records.append(enriched)

    return enriched_records


def print_tfidf_vocabulary_sizes(records: list[dict[str, str]]) -> None:
    """Print actual TF-IDF vocabulary sizes from title + description texts."""
    from classify_categories import (
        build_char_tfidf_vectorizer,
        build_input_text,
        build_word_tfidf_vectorizer,
    )

    texts = [
        build_input_text(record.get("title", ""), record.get("description", ""))
        for record in records
    ]

    actual_word_vectorizer = build_word_tfidf_vectorizer((1, 2))
    actual_word_vectorizer.set_params(max_features=None)
    actual_word_vectorizer.fit(texts)

    actual_char_vectorizer = build_char_tfidf_vectorizer()
    actual_char_vectorizer.set_params(max_features=None)
    actual_char_vectorizer.fit(texts)

    capped_word_vectorizer = build_word_tfidf_vectorizer((1, 2))
    capped_word_vectorizer.fit(texts)

    capped_char_vectorizer = build_char_tfidf_vectorizer()
    capped_char_vectorizer.fit(texts)

    print(
        "TF-IDF vocabulary sizes -",
        f"actual word(1,2): {len(actual_word_vectorizer.vocabulary_)}",
        f"actual char(3,5): {len(actual_char_vectorizer.vocabulary_)}",
        f"capped word(1,2): {len(capped_word_vectorizer.vocabulary_)}",
        f"capped char(3,5): {len(capped_char_vectorizer.vocabulary_)}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess item data into train/valid/test JSON splits.")
    parser.add_argument(
        "--add_represent",
        action="store_true",
        help="Generate `represent` with the vLLM server and cache item metadata in pickle/represent_items.pkl.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    item_dict = get_item_dict(pickle_path=ITEMS_PICKLE_PATH)
    if args.add_represent:
        item_dict = append_item_represent(item_dict, pickle_path=REPRESENT_ITEMS_PICKLE_PATH)
        print(f"loaded {len(item_dict)} items added represent")
    else:
        print(f"loaded {len(item_dict)} items")

    category_dict = build_category_dict(item_dict)
    save_category_dict(category_dict)

    if TRAIN_JSON_PATH.exists() and VALID_JSON_PATH.exists() and TEST_JSON_PATH.exists():
        train_records = load_json_records(TRAIN_JSON_PATH)
        valid_records = load_json_records(VALID_JSON_PATH)
        test_records = load_json_records(TEST_JSON_PATH)
    else:
        records = build_parsed_records(item_dict, category_dict)
        train_records, valid_records, test_records = split_records(records)
        save_json_records(train_records, TRAIN_JSON_PATH)
        save_json_records(valid_records, VALID_JSON_PATH)
        save_json_records(test_records, TEST_JSON_PATH)

    print(
        "split sizes -",
        f"train: {len(train_records)}",
        f"valid: {len(valid_records)}",
        f"test: {len(test_records)}",
    )
    # print_tfidf_vocabulary_sizes(train_records)


if __name__ == "__main__":
    main()

