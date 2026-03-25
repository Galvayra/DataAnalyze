from __future__ import annotations

from build.action_db import DB_PATH, bootstrap_database


def main() -> None:
    db_path = bootstrap_database()
    print(f"SQLite database created at: {db_path}")
    if db_path != DB_PATH:
        print(f"Resolved database path: {DB_PATH}")


if __name__ == "__main__":
    main()
