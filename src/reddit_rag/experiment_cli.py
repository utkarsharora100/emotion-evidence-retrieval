from __future__ import annotations

import argparse
import json

from .experiment import run_experiment
from .pipeline import load_csv, load_kaggle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Reddit sentiment retrieval experiment.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv")
    source.add_argument("--kaggle-dataset")
    parser.add_argument("--output", default="outputs/reddit_experiment")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--retrieval-limit", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--skip-retrieval", action="store_true", help="Run the full labeled classification benchmark without SPLADE retrieval encoding.")
    args = parser.parse_args()
    frame = load_csv(args.csv, reddit_only=True) if args.csv else load_kaggle(args.kaggle_dataset, reddit_only=True, limit=None)
    result = run_experiment(frame, args.output, args.epochs, args.retrieval_limit, args.queries, not args.skip_retrieval)
    print(json.dumps({"dataset_rows": result["dataset_rows"], "classification": {k: result["classification"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")}, "retrieval": result["retrieval"]}, indent=2))


if __name__ == "__main__":
    main()
