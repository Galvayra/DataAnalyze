from __future__ import annotations

try:
    from build.action_db import DB_PATH, PATTERN_QUERIES, fetch_query_results
except ModuleNotFoundError:
    from action_db import DB_PATH, PATTERN_QUERIES, fetch_query_results


def print_result(title: str, headers: list[str], rows: list[object]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return

    print(" | ".join(headers))
    print("-" * (sum(len(header) for header in headers) + len(headers) * 3))
    for row in rows:
        print(" | ".join(str(row[header]) for header in headers))


def main() -> None:
    print(f"Using SQLite database: {DB_PATH}")
    for title, sql in PATTERN_QUERIES:
        result_title, headers, rows = fetch_query_results(title, sql)
        print_result(result_title, headers, rows)


if __name__ == "__main__":
    main()
