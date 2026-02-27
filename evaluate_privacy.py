import argparse
import json

import numpy as np

from src.ml_utils import infer_target_column, load_bundle, load_dataframe, split_features_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate membership-inference risk for baseline vs DP models"
    )
    parser.add_argument("--data", required=True, help="Path to CSV/XLSX dataset")
    parser.add_argument("--target", default=None, help="Target column")
    parser.add_argument("--baseline", default="models/baseline_bundle.joblib")
    parser.add_argument("--dp", default="models/dp_bundle.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def membership_attack(conf_member: np.ndarray, conf_non_member: np.ndarray) -> dict:
    scores = np.concatenate([conf_member, conf_non_member])
    labels = np.concatenate([
        np.ones_like(conf_member, dtype=int),
        np.zeros_like(conf_non_member, dtype=int),
    ])

    candidate_thresholds = np.quantile(scores, q=np.linspace(0.05, 0.95, 37))

    best = None
    for th in candidate_thresholds:
        pred = (scores >= th).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())

        acc = (tp + tn) / max(len(labels), 1)
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        precision = tp / max(tp + fp, 1)

        record = {
            "threshold": float(th),
            "attack_accuracy": float(acc),
            "attack_precision": float(precision),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "advantage": float(abs(tpr - fpr)),
        }

        if best is None or record["attack_accuracy"] > best["attack_accuracy"]:
            best = record

    return best


def get_confidences(bundle: dict, X_member, X_non_member):
    pre = bundle["preprocessor"]
    model = bundle["model"]

    X_mem_t = pre.transform(X_member)
    X_non_t = pre.transform(X_non_member)

    conf_member = model.predict_proba(X_mem_t).max(axis=1)
    conf_non_member = model.predict_proba(X_non_t).max(axis=1)

    return conf_member, conf_non_member


def main() -> None:
    args = parse_args()

    df = load_dataframe(args.data)
    target_col = infer_target_column(df, args.target)
    X_train, X_test, _, _ = split_features_target(df, target_col, args.test_size, args.random_state)

    baseline_bundle = load_bundle(args.baseline)
    dp_bundle = load_bundle(args.dp)

    b_mem, b_non = get_confidences(baseline_bundle, X_train, X_test)
    d_mem, d_non = get_confidences(dp_bundle, X_train, X_test)

    result = {
        "baseline": membership_attack(b_mem, b_non),
        "dp": membership_attack(d_mem, d_non),
        "note": "Lower attack_accuracy and lower advantage are better.",
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
