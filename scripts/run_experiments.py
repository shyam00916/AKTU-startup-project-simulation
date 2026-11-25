"""CLI entrypoint to run reproducible ReceivablesOS ML experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from receivables_ml.config import load_config
from receivables_ml.pipeline import run_full_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReceivablesOS ML Research Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default_experiment.json"),
        help="Path to JSON or YAML experiment config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--n-customers", type=int, default=None, help="Override simulated customer count.")
    parser.add_argument("--epochs", type=int, default=None, help="Override sequence model epochs.")
    parser.add_argument("--output-root", type=str, default=None, help="Override artifacts output root.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_customers is not None:
        cfg.data.n_customers = args.n_customers
    if args.epochs is not None:
        cfg.sequence.epochs = args.epochs
    if args.output_root is not None:
        cfg.output_root = args.output_root

    result = run_full_experiment(cfg, base_dir=ROOT)
    print(f"Experiment completed: run_id={result['run_id']}")
    print(f"Artifacts saved to: {result['run_dir']}")
    print("\nHoldout Test Metrics Summary:")
    for model_name, m in result["metrics"]["model_comparison"]["test_metrics"].items():
        print(
            f"  {model_name:22s} | ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} "
            f"| F1={m['f1']:.4f} | ECE={m['calibration_ece']:.4f} | MAE={m['mae']:.2f}d | RMSE={m['rmse']:.2f}d"
        )


if __name__ == "__main__":
    main()
