"""Запуск полного A/B/C эксперимента на CSV без Jupyter."""
from pathlib import Path
import argparse
from src.hotel_project import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Модуль 1 ML: проект 4, уровень C")
    parser.add_argument("--data", type=Path, default=Path("data/hotel_bookings.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/full"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Нет файла {args.data}. Скачайте CSV: python download_data.py или запустите демо: bash run_demo.sh")
    summary, cv, groups = run_experiment(args.data, args.out, args.folds)
    print("\nCROSS-VALIDATION:\n", cv.to_string(index=False))
    print("\nОценка на отложенном тесте:", summary["test_at_selected_threshold"])
    print("\nЦелевая precision достигнута на validation:", summary["goal_met_on_validation"])
    print("\nОтчёты:", args.out)
