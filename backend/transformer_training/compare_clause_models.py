"""Compare transformer results with established TF-IDF and hybrid baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_BASELINES = {
    "post_audit_baseline": {
        "accuracy": 0.4121,
        "macro_f1": 0.3649,
        "weighted_f1": 0.4489,
    },
    "character_tfidf": {
        "accuracy": 0.5000,
        "macro_f1": 0.4467,
        "weighted_f1": 0.4974,
    },
    "current_hybrid": {
        "accuracy": 0.5333,
        "macro_f1": 0.4686,
        "weighted_f1": 0.5313,
    },
}


def recommendation(
    transformer: dict[str, float],
) -> str:
    """Return a conservative model-adoption recommendation."""

    hybrid = DEFAULT_BASELINES["current_hybrid"]

    accuracy_gain = (
        transformer["accuracy"]
        - hybrid["accuracy"]
    )
    macro_gain = (
        transformer["macro_f1"]
        - hybrid["macro_f1"]
    )

    if accuracy_gain >= 0.03 and macro_gain >= 0.02:
        return (
            "Strong candidate for integration. Test a transformer-plus-rules "
            "hybrid and verify runtime requirements before replacing the "
            "current classifier."
        )

    if accuracy_gain > 0 and macro_gain > 0:
        return (
            "Transformer improves both accuracy and Macro F1, but the gain "
            "is modest. Keep the current model until latency, model size, "
            "and class-level changes are reviewed."
        )

    if (
        transformer["accuracy"] > hybrid["accuracy"]
        or transformer["macro_f1"] > hybrid["macro_f1"]
    ):
        return (
            "Mixed result. Consider using the transformer as an ensemble "
            "signal for selected clause classes rather than replacing the "
            "current hybrid."
        )

    return (
        "Do not replace the current hybrid classifier. Retain this as a "
        "documented transformer experiment or investigate a legal-domain "
        "pretrained model."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer-summary",
        type=Path,
        default=Path(
            "transformer_training/results/"
            "transformer_external_summary.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "transformer_training/results/"
            "model_comparison.json"
        ),
    )
    args = parser.parse_args()

    transformer = json.loads(
        args.transformer_summary.read_text(
            encoding="utf-8"
        )
    )

    comparison = {
        "baselines": DEFAULT_BASELINES,
        "transformer": {
            "accuracy": transformer["accuracy"],
            "macro_f1": transformer["macro_f1"],
            "weighted_f1": transformer["weighted_f1"],
            "milliseconds_per_clause": transformer.get(
                "milliseconds_per_clause"
            ),
            "model_size_megabytes": transformer.get(
                "model_size_megabytes"
            ),
        },
        "difference_vs_current_hybrid": {
            "accuracy": (
                transformer["accuracy"]
                - DEFAULT_BASELINES[
                    "current_hybrid"
                ]["accuracy"]
            ),
            "macro_f1": (
                transformer["macro_f1"]
                - DEFAULT_BASELINES[
                    "current_hybrid"
                ]["macro_f1"]
            ),
            "weighted_f1": (
                transformer["weighted_f1"]
                - DEFAULT_BASELINES[
                    "current_hybrid"
                ]["weighted_f1"]
            ),
        },
        "recommendation": recommendation(
            transformer
        ),
    }

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_json.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
