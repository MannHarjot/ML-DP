import argparse

import numpy as np

from src.dp_logistic import DPLogisticRegression
from src.ml_utils import (
    build_preprocessor,
    evaluate_binary,
    infer_target_column,
    load_dataframe,
    metrics_to_json,
    save_bundle,
    split_features_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DP-SGD binary risk model")
    parser.add_argument("--data", required=True, help="Path to CSV/XLSX dataset")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--model-out", default="models/dp_bundle.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--l2-reg", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataframe(args.data)
    target_col = infer_target_column(df, args.target)

    X_train, X_test, y_train, y_test = split_features_target(
        df, target_col, args.test_size, args.random_state
    )

    classes = np.sort(y_train.unique())
    if len(classes) != 2:
        raise ValueError("This MVP DP trainer expects a binary target")

    positive_label = classes[1]
    y_train_bin = (y_train == positive_label).astype(int).to_numpy()
    y_test_bin = (y_test == positive_label).astype(int).to_numpy()

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = DPLogisticRegression(
        epsilon=args.epsilon,
        delta=args.delta,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
        l2_reg=args.l2_reg,
        random_state=args.random_state,
    )
    model.fit(X_train_t, y_train_bin)

    y_prob = model.predict_proba(X_test_t)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluate_binary(y_test_bin, y_pred, y_prob)

    bundle = {
        "model_type": "dp_logreg",
        "model": model,
        "preprocessor": preprocessor,
        "target_col": target_col,
        "feature_cols": X_train.columns.tolist(),
        "positive_label": str(positive_label),
        "classes": [str(c) for c in classes.tolist()],
        "privacy": {
            "epsilon": args.epsilon,
            "delta": args.delta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_grad_norm": args.max_grad_norm,
        },
        "metrics": metrics,
    }
    save_bundle(args.model_out, bundle)

    print("Saved DP bundle:", args.model_out)
    print(metrics_to_json(metrics))


if __name__ == "__main__":
    main()
