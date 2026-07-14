from pathlib import Path
import re

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml_model"

DATA_FILE = MODEL_DIR / "external_test_clauses_final.csv"
MODEL_FILE = MODEL_DIR / "clause_type_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "clause_type_vectorizer.pkl"

RESULTS_FILE = MODEL_DIR / "phase6_hybrid_results.csv"
PREDICTIONS_FILE = MODEL_DIR / "phase6_hybrid_predictions.csv"
REPORT_FILE = MODEL_DIR / "phase6_hybrid_report.txt"

LOW_CONFIDENCE_THRESHOLD = 0.40


ALLOWED_LABELS = {
    "Alterations / Improvements",
    "Assignment / Subletting",
    "Confidentiality",
    "Dispute Resolution",
    "General Obligation",
    "Governing Law",
    "Insurance",
    "Lease Grant",
    "Liability / Indemnity",
    "Notice",
    "Other / Unknown",
    "Payment / Rent",
    "Possession / Surrender",
    "Property / Premises",
    "Quiet Enjoyment",
    "Repairs / Maintenance",
    "Signature / Execution",
    "Taxes / Utilities",
    "Term and Renewal",
    "Termination / Default",
    "Use of Premises",
    "Warranties / Representations",
}


# High-precision legal rules.
# These are intentionally narrower than the broad rules used in main.py.
RULES = [
    (
        "Signature / Execution",
        [
            r"\bin witness whereof\b",
            r"\bsigned and delivered\b",
            r"\bexecuted by\b",
            r"\belectronic signatures?\b",
            r"\bcounterparts?\b",
            r"\bwitnessed by\b",
            r"\bin the presence of\b",
        ],
    ),
    (
        "Quiet Enjoyment",
        [
            r"\bquiet enjoyment\b",
            r"\bpeaceably and quietly\b",
            r"\bquietly hold\b",
            r"\bquietly enjoy\b",
            r"\bwithout interruption\b",
            r"\bwithout disturbance\b",
        ],
    ),
    (
        "Assignment / Subletting",
        [
            r"\bshall not assign\b",
            r"\bnot to assign\b",
            r"\bassignment\b",
            r"\bsublet(?:ting)?\b",
            r"\bunderlet\b",
            r"\bpart with possession\b",
        ],
    ),
    (
        "Insurance",
        [
            r"\bcommercial general liability insurance\b",
            r"\brenter'?s insurance\b",
            r"\bproperty insurance\b",
            r"\badditional insured\b",
            r"\binsured risks?\b",
            r"\binsurance premium\b",
        ],
    ),
    (
        "Liability / Indemnity",
        [
            r"\bindemnif(?:y|ied|ication)\b",
            r"\bhold harmless\b",
            r"\blimitation of liability\b",
            r"\bunlimited liability\b",
        ],
    ),
    (
        "Termination / Default",
        [
            r"\bevent of default\b",
            r"\bright of re-entry\b",
            r"\bright of reentry\b",
            r"\bre-entry\b",
            r"\breentry\b",
            r"\bearlier determination\b",
            r"\bterminate this agreement\b",
            r"\btermination of this lease\b",
        ],
    ),
    (
        "Possession / Surrender",
        [
            r"\bsurrender the premises\b",
            r"\byield up the premises\b",
            r"\bvacant possession\b",
            r"\bdeliver possession\b",
            r"\breturn possession\b",
            r"\bholding over\b",
        ],
    ),
    (
        "Alterations / Improvements",
        [
            r"\bstructural alteration\b",
            r"\btenant improvements?\b",
            r"\badditions or alterations\b",
            r"\binstall fixtures?\b",
            r"\bapproved plans\b",
        ],
    ),
    (
        "Repairs / Maintenance",
        [
            r"\bgood and substantial repair\b",
            r"\btenantable repair\b",
            r"\bstructural repairs?\b",
            r"\bkeep the premises clean\b",
            r"\bmaintain the premises\b",
        ],
    ),
    (
        "Taxes / Utilities",
        [
            r"\bproperty taxes?\b",
            r"\breal estate taxes?\b",
            r"\brates and taxes\b",
            r"\butility charges?\b",
            r"\bwater charges?\b",
            r"\belectricity charges?\b",
            r"\bgas charges?\b",
        ],
    ),
    (
        "Governing Law",
        [
            r"\bgoverned by the laws?\b",
            r"\bgoverning law\b",
            r"\bchoice of law\b",
        ],
    ),
    (
        "Dispute Resolution",
        [
            r"\barbitration\b",
            r"\bmediation\b",
            r"\bwaiver of jury trial\b",
        ],
    ),
    (
        "Confidentiality",
        [
            r"\bconfidential information\b",
            r"\bnon-disclosure\b",
            r"\bnondisclosure\b",
            r"\bproprietary information\b",
        ],
    ),
    (
        "Notice",
        [
            r"\bnotice shall be in writing\b",
            r"\bcertified mail\b",
            r"\bregistered post\b",
            r"\bformal notice\b",
        ],
    ),
    (
        "Lease Grant",
        [
            r"\bhereby leases\b",
            r"\bhereby lets\b",
            r"\bhereby demises\b",
            r"\bto hold the demised premises\b",
            r"\baccepts the lease\b",
        ],
    ),
    (
        "Term and Renewal",
        [
            r"\boption to renew\b",
            r"\brenewal term\b",
            r"\binitial term\b",
            r"\bfixed term\b",
        ],
    ),
    (
        "Payment / Rent",
        [
            r"\bmonthly rent\b",
            r"\bannual rent\b",
            r"\bbase rent\b",
            r"\bsecurity deposit\b",
            r"\badditional rent\b",
        ],
    ),
    (
        "Use of Premises",
        [
            r"\bused only for\b",
            r"\bpermitted use\b",
            r"\bresidential purposes only\b",
            r"\bbusiness purposes only\b",
        ],
    ),
]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def load_external_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Frozen external test file not found:\n{DATA_FILE}"
        )

    dataframe = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    required = {"clause_text", "clause_type"}
    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"External test file missing columns: {sorted(missing)}"
        )

    dataframe = dataframe.copy()
    dataframe["clause_text"] = dataframe["clause_text"].apply(clean_text)
    dataframe["clause_type"] = dataframe["clause_type"].apply(clean_text)

    if "corrected_clause_type" in dataframe.columns:
        corrected = (
            dataframe["corrected_clause_type"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        dataframe["clause_type"] = corrected.where(
            corrected != "",
            dataframe["clause_type"],
        )

    if "include_in_evaluation" in dataframe.columns:
        include_mask = (
            dataframe["include_in_evaluation"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        )

        dataframe = dataframe[include_mask].copy()

    dataframe = dataframe[
        (dataframe["clause_text"] != "")
        & (dataframe["clause_type"] != "")
    ].copy()

    return dataframe.reset_index(drop=True)


def detect_rule_label(text: str) -> tuple[str | None, list[str]]:
    matches = []

    for label, patterns in RULES:
        if any(
            re.search(pattern, text, flags=re.IGNORECASE)
            for pattern in patterns
        ):
            matches.append(label)

    unique_matches = list(dict.fromkeys(matches))

    if len(unique_matches) == 1:
        return unique_matches[0], unique_matches

    return None, unique_matches


def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> dict:
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        ),
    }


def main() -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    if not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            f"Vectorizer not found:\n{VECTORIZER_FILE}"
        )

    dataframe = load_external_data()

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    features = vectorizer.transform(dataframe["clause_text"])
    ml_predictions = model.predict(features)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        ml_confidence = probabilities.max(axis=1)
    else:
        ml_confidence = [0.0] * len(dataframe)

    rule_predictions = []
    rule_match_counts = []
    hybrid_predictions = []
    hybrid_sources = []

    for index, text in enumerate(dataframe["clause_text"]):
        rule_label, all_rule_matches = detect_rule_label(text)

        ml_label = str(ml_predictions[index]).strip()
        confidence = float(ml_confidence[index])

        # Rules-only mode:
        # when no single high-precision rule applies, use Other / Unknown.
        rule_prediction = (
            rule_label
            if rule_label in ALLOWED_LABELS
            else "Other / Unknown"
        )

        # Hybrid mode:
        # - use a unique high-precision rule when ML confidence is low
        # - also use a rule for several highly distinctive categories
        distinctive_rule_labels = {
            "Signature / Execution",
            "Quiet Enjoyment",
            "Assignment / Subletting",
            "Insurance",
            "Liability / Indemnity",
            "Governing Law",
            "Dispute Resolution",
            "Confidentiality",
        }

        use_rule = (
            rule_label in ALLOWED_LABELS
            and (
                confidence < LOW_CONFIDENCE_THRESHOLD
                or rule_label in distinctive_rule_labels
            )
        )

        if use_rule:
            hybrid_prediction = rule_label
            hybrid_source = "rule"
        else:
            hybrid_prediction = ml_label
            hybrid_source = "ml"

        rule_predictions.append(rule_prediction)
        rule_match_counts.append(len(all_rule_matches))
        hybrid_predictions.append(hybrid_prediction)
        hybrid_sources.append(hybrid_source)

    dataframe["ml_prediction"] = ml_predictions
    dataframe["ml_confidence"] = ml_confidence
    dataframe["rule_prediction"] = rule_predictions
    dataframe["rule_match_count"] = rule_match_counts
    dataframe["hybrid_prediction"] = hybrid_predictions
    dataframe["hybrid_source"] = hybrid_sources

    ml_metrics = calculate_metrics(
        dataframe["clause_type"],
        dataframe["ml_prediction"],
    )

    rule_metrics = calculate_metrics(
        dataframe["clause_type"],
        dataframe["rule_prediction"],
    )

    hybrid_metrics = calculate_metrics(
        dataframe["clause_type"],
        dataframe["hybrid_prediction"],
    )

    results = pd.DataFrame(
        [
            {
                "approach": "ML only",
                **ml_metrics,
            },
            {
                "approach": "Rules only",
                **rule_metrics,
            },
            {
                "approach": "Hybrid",
                **hybrid_metrics,
            },
        ]
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    dataframe.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    best_row = results.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]

    report_parts = [
        "PHASE 6 — HYBRID VALIDATION",
        "=" * 80,
        "",
        f"Frozen external examples: {len(dataframe)}",
        f"Hybrid confidence threshold: {LOW_CONFIDENCE_THRESHOLD:.2f}",
        "",
        "SUMMARY",
        "-" * 80,
        results.to_string(index=False),
        "",
    ]

    for approach, prediction_column in [
        ("ML ONLY", "ml_prediction"),
        ("RULES ONLY", "rule_prediction"),
        ("HYBRID", "hybrid_prediction"),
    ]:
        report_parts.extend(
            [
                "=" * 80,
                approach,
                "=" * 80,
                classification_report(
                    dataframe["clause_type"],
                    dataframe[prediction_column],
                    digits=4,
                    zero_division=0,
                ),
                "",
            ]
        )

    report_parts.extend(
        [
            "=" * 80,
            "BEST APPROACH",
            "=" * 80,
            f"Approach: {best_row['approach']}",
            f"Accuracy: {best_row['accuracy']:.4f}",
            f"Macro F1: {best_row['macro_f1']:.4f}",
            f"Weighted F1: {best_row['weighted_f1']:.4f}",
            "",
            f"Hybrid predictions using rules: "
            f"{(dataframe['hybrid_source'] == 'rule').sum()}",
            f"Hybrid predictions using ML: "
            f"{(dataframe['hybrid_source'] == 'ml').sum()}",
            "",
            "Files generated:",
            str(RESULTS_FILE),
            str(PREDICTIONS_FILE),
            str(REPORT_FILE),
        ]
    )

    REPORT_FILE.write_text(
        "\n".join(report_parts),
        encoding="utf-8",
    )

    print("PHASE 6 — HYBRID VALIDATION")
    print("=" * 60)
    print(f"Frozen external examples: {len(dataframe)}")

    print("\nResults:")
    print(results.to_string(index=False))

    print("\nHybrid source usage:")
    print(
        dataframe["hybrid_source"]
        .value_counts()
        .to_string()
    )

    print("\nBest approach:")
    print(f"  {best_row['approach']}")
    print(f"  Accuracy: {best_row['accuracy']:.4f}")
    print(f"  Macro F1: {best_row['macro_f1']:.4f}")
    print(f"  Weighted F1: {best_row['weighted_f1']:.4f}")

    print(f"\nReport saved to:\n{REPORT_FILE}")
    print(f"\nResults saved to:\n{RESULTS_FILE}")
    print(f"\nPredictions saved to:\n{PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()