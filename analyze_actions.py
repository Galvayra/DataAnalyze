from __future__ import annotations

import argparse
from pathlib import Path

from analyze.age_order_trends import main as run_age_order_trends
from analyze.category_order_trends import main as run_category_order_trends
from analyze.search_keyword_conversion import main as run_search_keyword_conversion
from analyze.shop_target_age_comparison import main as run_shop_target_age_comparison
from analyze.time_of_day_orders import main as run_time_of_day_orders


ANALYSIS_HELP = {
    "1": "시간대별 주문 추이 시각화",
    "2": "카테고리별 주문 추이 시각화",
    "3": "연령대별 주문 추이 시각화",
    "4": "검색 키워드와 구매 전환 시각화",
    "5": "쇼핑몰 타깃 연령 비교 시각화",
    "time_of_day_orders": "시간대별 주문 추이 시각화",
    "category_order_trends": "카테고리별 주문 추이 시각화",
    "age_order_trends": "연령대별 주문 추이 시각화",
    "search_keyword_conversion": "검색 키워드와 구매 전환 시각화",
    "shop_target_age_comparison": "쇼핑몰 타깃 연령 비교 시각화",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Action 데이터 분석 스크립트를 선택적으로 실행합니다."
    )
    parser.add_argument(
        "--analysis",
        "-a",
        choices=sorted(ANALYSIS_HELP.keys()),
        help="실행할 분석 번호를 선택합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = args.analysis

    if not selection:
        print("실행할 분석 번호를 선택하세요.")
        for key, description in ANALYSIS_HELP.items():
            if key.isdigit():
                print(f"{key}. {description}")
        print("\n예시: python3 analyze_actions.py --analysis 1")
        print("또는: python3 analyze_actions.py --analysis category_order_trends")
        return

    if selection in {"1", "time_of_day_orders"}:
        output_path: Path = run_time_of_day_orders()
        print(f"완료: {output_path}")
        return

    if selection in {"2", "category_order_trends"}:
        output_path = run_category_order_trends()
        print(f"완료: {output_path}")
        return

    if selection in {"3", "age_order_trends"}:
        output_path = run_age_order_trends()
        print(f"완료: {output_path}")
        return

    if selection in {"4", "search_keyword_conversion"}:
        output_path = run_search_keyword_conversion()
        print(f"완료: {output_path}")
        return

    if selection in {"5", "shop_target_age_comparison"}:
        output_path = run_shop_target_age_comparison()
        print(f"완료: {output_path}")
        return

    print(f"{selection} 분석은 아직 구현되지 않았습니다.")


if __name__ == "__main__":
    main()
