"""Модуль 1 ML, проект 4: отмена бронирования отеля (A/B/C)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, StratifiedGroupKFold, cross_validate, train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_FEATURES = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal",
    "country", "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled",
    "reserved_room_type", "deposit_type", "customer_type",
]
POST_BOOKING = [
    "reservation_status", "reservation_status_date", "assigned_room_type",
    "booking_changes", "days_in_waiting_list", "total_of_special_requests",
]
DROPPED_EXTRAS = [
    "company", "agent", "arrival_date_week_number", "adr",
    "required_car_parking_spaces",
]
TARGET = "is_canceled"
SEED = 42

def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(MODEL_FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Отсутствуют необходимые столбцы: " + ", ".join(sorted(missing)))
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not df[TARGET].isin([0, 1]).all() or df[TARGET].nunique() != 2:
        raise ValueError("В is_canceled должны быть классы 0 и 1")
    return df

def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    def as_number(name):
        return pd.to_numeric(df[name], errors="coerce")
    guests = sum(as_number(c).fillna(0) for c in ("adults", "children", "babies"))
    adr = as_number("adr") if "adr" in df else pd.Series(dtype=float)
    values = {
        "строк": len(df), "столбцов": len(df.columns),
        "доля_отмен": float(df[TARGET].mean()),
        "дубликаты_полных_строк": int(df.duplicated().sum()),
        "брони_без_гостей": int((guests == 0).sum()),
        "отрицательная_цена_adr": int((adr < 0).sum()),
        "adr_больше_1000": int((adr > 1000).sum()),
    }
    return pd.DataFrame([{"проверка": k, "значение": v} for k, v in values.items()])

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "пропуски": df.isna().sum(), "процент": 100 * df.isna().mean(),
    })
    return out[out["пропуски"] > 0].sort_values("пропуски", ascending=False)

def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, MODEL_FEATURES].copy()

def split_data(df: pd.DataFrame):
    groups = pd.util.hash_pandas_object(feature_frame(df), index=False)
    if groups.duplicated().any():
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_indices = [test for _, test in cv.split(df, df[TARGET], groups=groups)]
        train_idx = np.concatenate(fold_indices[2:])
        return (
            df.iloc[train_idx].copy(), df.iloc[fold_indices[1]].copy(),
            df.iloc[fold_indices[0]].copy(),
        )
    train_val, test = train_test_split(
        df, test_size=.2, stratify=df[TARGET], random_state=SEED,
    )
    train, val = train_test_split(
        train_val, test_size=.25, stratify=train_val[TARGET], random_state=SEED,
    )
    return train.copy(), val.copy(), test.copy()

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categoric = [name for name in X.columns if name not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]), categoric),
    ], remainder="drop", sparse_threshold=.3)

def make_pipeline(X: pd.DataFrame, variant: str = "baseline") -> Pipeline:
    if variant not in ("baseline", "tree", "weighted", "oversample"):
        raise ValueError(f"Неизвестная модель {variant}")
    if variant == "tree":
        estimator = DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=40, random_state=SEED,
        )
    else:
        estimator = LogisticRegression(
            solver="liblinear", max_iter=350, random_state=SEED,
            class_weight="balanced" if variant == "weighted" else None,
        )
    steps: list[tuple[str, Any]] = [("prep", make_preprocessor(X))]
    if variant == "oversample":
        steps.append(("oversample", RandomOverSampler(random_state=SEED)))
    steps.append(("classifier", estimator))
    return Pipeline(steps)

def compare_models(
    train: pd.DataFrame,
    variants=("baseline", "tree", "weighted", "oversample"),
    n_splits=5,
):
    X = feature_frame(train)
    y = train[TARGET]
    groups = pd.util.hash_pandas_object(X, index=False)
    if groups.duplicated().any():
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED,
        )
        cv_groups = groups
    else:
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED,
        )
        cv_groups = None
    rows = []
    for variant in variants:
        scores = cross_validate(
            make_pipeline(X, variant), X, y, cv=cv, groups=cv_groups,
            scoring={
                "pr_auc": "average_precision", "roc_auc": "roc_auc",
                "precision_at_half": "precision", "recall_at_half": "recall",
            },
            n_jobs=1,
        )
        rows.append({
            "model": variant,
            "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
            "cv_pr_auc_std": float(scores["test_pr_auc"].std()),
            "cv_roc_auc": float(scores["test_roc_auc"].mean()),
            "cv_precision_0_5": float(scores["test_precision_at_half"].mean()),
            "cv_recall_0_5": float(scores["test_recall_at_half"].mean()),
        })
    return pd.DataFrame(rows).sort_values(
        "cv_pr_auc_mean", ascending=False,
    ).reset_index(drop=True)

def threshold_table(y_true, probabilities, min_alerts: int = 20) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows = []
    for threshold in np.round(np.arange(.5, .996, .005), 3):
        pred = (probabilities >= threshold)
        alerts = int(pred.sum())
        if alerts < min_alerts:
            continue
        tp = int(((y_true == 1) & pred).sum())
        fp = int(((y_true == 0) & pred).sum())
        rows.append({
            "threshold": float(threshold), "alerts": alerts,
            "TP": tp, "FP": fp, "precision": tp / alerts,
            "recall": tp / int(y_true.sum()) if y_true.sum() else 0.,
        })
    if not rows:
        raise ValueError("Не найден порог с достаточным количеством прогнозов")
    return pd.DataFrame(rows)

def choose_threshold(table: pd.DataFrame, target_precision=.95):
    good = table[table["precision"] >= target_precision]
    if len(good):
        chosen = good.sort_values(
            ["recall", "precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        return float(chosen["threshold"]), True
    chosen = table.sort_values(
        ["precision", "recall", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(chosen["threshold"]), False

def evaluate(y_true, prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "N": int(len(y_true)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "predicted_cancellations": int(pred.sum()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }

def group_errors(test: pd.DataFrame, prob, threshold,
                 columns=("hotel", "market_segment", "deposit_type")):
    frame = test.copy()
    frame["pred"] = (np.asarray(prob) >= threshold).astype(int)
    rows = []
    for column in columns:
        if column not in frame:
            continue
        for value, group in frame.groupby(column, dropna=False):
            y = group[TARGET].to_numpy(dtype=int)
            p = group["pred"].to_numpy(dtype=int)
            tp = int(((y == 1) & (p == 1)).sum())
            fp = int(((y == 0) & (p == 1)).sum())
            fn = int(((y == 1) & (p == 0)).sum())
            tn = int(((y == 0) & (p == 0)).sum())
            rows.append({
                "feature": column, "group": str(value), "n": len(group),
                "FP": fp, "FN": fn,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                "false_negative_rate": fn / (fn + tp) if fn + tp else np.nan,
                "alerts": tp + fp,
            })
    return pd.DataFrame(rows).sort_values(
        ["feature", "n"], ascending=[True, False],
    )

def create_eda_figures(train: pd.DataFrame, folder: str | Path):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    train[TARGET].value_counts().sort_index().plot.bar(
        color=["#347db0", "#d07d34"],
    )
    plt.xticks([0, 1], ["Не отменена", "Отменена"], rotation=0)
    plt.ylabel("Бронирований (train)")
    plt.title("Баланс классов")
    plt.tight_layout()
    plt.savefig(folder / "class_balance.png", dpi=130)
    plt.close()
    plt.figure(figsize=(7, 4))
    for cls, label in [(0, "Не отменена"), (1, "Отменена")]:
        plt.hist(
            train.loc[train[TARGET] == cls, "lead_time"].dropna(),
            bins=45, alpha=.5, label=label,
        )
    plt.xlabel("Дней до заезда (lead_time)")
    plt.ylabel("Количество")
    plt.title("Срок до заезда и отмена (train)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "lead_time_hist.png", dpi=130)
    plt.close()

def save_threshold_figure(table: pd.DataFrame, choice: float, folder: str | Path):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(table["threshold"], table["precision"], label="Precision")
    plt.plot(table["threshold"], table["recall"], label="Recall")
    plt.axvline(choice, linestyle="--", label=f"Выбранный порог: {choice:.3f}")
    plt.axhline(.95, linestyle=":", color="gray", label="Цель precision 0.95")
    plt.xlabel("Порог")
    plt.ylabel("Метрика на validation")
    plt.title("Выбор порога на validation, не на test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "precision_recall_threshold.png", dpi=130)
    plt.close()

def run_experiment(data_path: str | Path, output_path: str | Path,
                   n_splits=5, precision_goal=.95):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    audit_data(df).to_csv(output_path / "audit.csv", index=False)
    missing_report(df).to_csv(output_path / "missing.csv")
    train, val, test = split_data(df)
    create_eda_figures(train, output_path)
    cv = compare_models(train, n_splits=n_splits)
    cv.to_csv(output_path / "cross_validation.csv", index=False)
    winner = str(cv.iloc[0]["model"])
    model = make_pipeline(feature_frame(train), winner)
    model.fit(feature_frame(train), train[TARGET])
    prob_val = model.predict_proba(feature_frame(val))[:, 1]
    table = threshold_table(val[TARGET], prob_val,
                            min_alerts=max(20, int(.01 * len(val))))
    table.to_csv(output_path / "thresholds_validation.csv", index=False)
    threshold, safe = choose_threshold(table, precision_goal)
    save_threshold_figure(table, threshold, output_path)
    prob_test = model.predict_proba(feature_frame(test))[:, 1]
    default = evaluate(test[TARGET], prob_test, .5)
    final = evaluate(test[TARGET], prob_test, threshold)
    groups = group_errors(test, prob_test, threshold)
    groups.to_csv(output_path / "group_errors_test.csv", index=False)
    summary = {
        "data_path": str(data_path), "source_rows": len(df),
        "train_size": len(train), "val_size": len(val), "test_size": len(test),
        "winner_from_cv": winner, "precision_goal": precision_goal,
        "goal_met_on_validation": safe,
        "test_at_0_5": default, "test_at_selected_threshold": final,
        "message": "Исследовательский прототип. Нужна временная валидация, калибровка и аудит рисков.",
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return summary, cv, groups
