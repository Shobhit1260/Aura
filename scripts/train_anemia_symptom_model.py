from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a symptom-based anemia logistic regression model")
    parser.add_argument("--csv", type=str, required=True, help="CSV with symptom columns and target")
    parser.add_argument("--output", type=str, required=True, help="Path to output .joblib model")
    parser.add_argument(
        "--features",
        type=str,
        default="fatigue,pale_skin,dizziness,shortness_of_breath,headache,cold_hands_feet",
        help="Comma-separated feature column names",
    )
    parser.add_argument("--target", type=str, default="anemia", help="Target column name (0/1)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = [f.strip() for f in args.features.split(",") if f.strip()]

    df = pd.read_csv(csv_path)
    missing = [c for c in features + [args.target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    x = df[features]
    y = df[args.target].astype(int)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )

    model.fit(x_train, y_train)

    val_probs = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)

    joblib.dump(model, output_path)

    metadata = {
        "features": features,
        "target": args.target,
        "val_auc": float(auc),
        "n_samples": int(len(df)),
    }
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Validation AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
