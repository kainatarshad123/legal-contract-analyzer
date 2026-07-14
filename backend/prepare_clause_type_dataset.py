from pathlib import Path
import re

import pandas as pd
from datasets import load_dataset


# ---------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml_model"
OUTPUT_FILE = MODEL_DIR / "clause_type_dataset.csv"

# Limit each category so one large category does not dominate training.
MAX_EXAMPLES_PER_CATEGORY = 500

RANDOM_STATE = 42


# ---------------------------------------------------------
# Map LEDGAR labels to the categories used by our SaaS
# ---------------------------------------------------------

LABEL_RULES = {
    "Payment / Rent": [
        "payment",
        "payments",
        "fees",
        "purchase price",
        "compensation",
        "salary",
        "royalties",
    ],

    "Term and Renewal": [
    "term of the agreement",
    "agreement term",
    "initial term",
    "renewal",
    "renewals",
    "duration",
    "expiration",
    "extension",
],

    "Termination / Default": [
        "termination",
        "terminations",
        "default",
        "defaults",
        "events of default",
        "remedies",
    ],

    "Assignment / Subletting": [
        "assignment",
        "assignments",
        "assigns",
        "successors and assigns",
    ],

    "Liability / Indemnity": [
        "indemnification",
        "indemnifications",
        "indemnity",
        "liability",
        "liabilities",
        "limitation of liability",
        "hold harmless",
    ],

    "Confidentiality": [
        "confidentiality",
        "confidential information",
        "non-disclosure",
        "nondisclosure",
        "secrecy",
    ],

    "Dispute Resolution": [
        "arbitration",
        "dispute resolution",
        "jurisdiction",
        "consent to jurisdiction",
        "venue",
        "waiver of jury trial",
    ],

    "Governing Law": [
        "governing law",
        "governing laws",
        "applicable law",
        "applicable laws",
        "choice of law",
    ],

    "Notice": [
        "notice",
        "notices",
    ],

    "Property / Premises": [
    "premises",
    "real property",
    "leased premises",
    "leased property",
    "property condition",
],

    "Insurance": [
        "insurance",
        "insurance policies",
        "coverage",
    ],

    "Force Majeure": [
        "force majeure",
        "acts of god",
    ],

    "Warranties / Representations": [
        "warranty",
        "warranties",
        "representation",
        "representations",
        "representations and warranties",
    ],

    "Signature / Execution": [
        "signature",
        "signatures",
        "execution",
        "counterpart",
        "counterparts",
        "electronic signatures",
    ],

    "General Obligation": [
        "obligation",
        "obligations",
        "covenant",
        "covenants",
        "cooperation",
        "compliance with laws",
        "further assurances",
    ],
}


def normalize_label(label: str) -> str:
    """Normalize a dataset label before matching it."""
    label = label.lower().strip()
    label = label.replace("_", " ")
    label = label.replace("-", " ")
    label = re.sub(r"\s+", " ", label)
    return label


def map_ledgar_label(original_label: str) -> str | None:
    """
    Convert an original LEDGAR category into one of our categories.

    Returns None when the LEDGAR category is not currently useful for
    our clause-type classifier.
    """
    normalized = normalize_label(original_label)

    excluded_labels = {
        "intellectual property",
        "defined terms",
        "definitions",
    }

    if normalized in excluded_labels:
        return None

    # Prefer exact matches before partial matches.
    for target_category, keywords in LABEL_RULES.items():
        for keyword in keywords:
            if normalized == normalize_label(keyword):
                return target_category

    # Then allow partial matching.
    for target_category, keywords in LABEL_RULES.items():
        for keyword in keywords:
            normalized_keyword = normalize_label(keyword)

            if normalized_keyword in normalized:
                return target_category

    return None


def clean_clause_text(text: str) -> str:
    """Remove unnecessary whitespace and reject unusable text."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_clause_text(text: str) -> str:
    """Remove unnecessary whitespace and reject unusable text."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading the LEDGAR dataset...")
    print("This requires an internet connection.")

    dataset = load_dataset(
        "coastalcph/lex_glue",
        "ledgar",
    )

    print("\nDataset downloaded successfully.")
    print(f"Available splits: {list(dataset.keys())}")

    # Obtain the original class names from the dataset metadata.
    first_split_name = next(iter(dataset.keys()))
    label_feature = dataset[first_split_name].features["label"]
    original_label_names = label_feature.names

    print(f"Original LEDGAR categories: {len(original_label_names)}")

    rows = []

    for split_name, split_data in dataset.items():
        print(f"Processing split: {split_name}")

        for record in split_data:
            clause_text = clean_clause_text(record.get("text", ""))
            label_id = record.get("label")

            if not clause_text:
                continue

            if label_id is None:
                continue

            if not isinstance(label_id, int):
                continue

            if label_id < 0 or label_id >= len(original_label_names):
                continue

            original_label = original_label_names[label_id]
            clause_type = map_ledgar_label(original_label)

            # Ignore labels that are outside our initial categories.
            if clause_type is None:
                continue

            # Very short text is usually a heading or extraction error.
            if len(clause_text.split()) < 8:
                continue

            rows.append(
                {
                    "clause_text": clause_text,
                    "clause_type": clause_type,
                    "original_label": original_label,
                    "source": "LEDGAR",
                    "source_split": split_name,
                }
            )

    if not rows:
        raise RuntimeError(
            "No clauses were mapped. The dataset format or labels may "
            "have changed."
        )

    dataframe = pd.DataFrame(rows)

    print(f"\nMapped examples before cleaning: {len(dataframe)}")

    # Remove exact duplicates.
    dataframe = dataframe.drop_duplicates(
        subset=["clause_text", "clause_type"]
    ).reset_index(drop=True)

    # Balance the categories by limiting each category.
    balanced_parts = []

    for category, category_data in dataframe.groupby("clause_type"):
        sample_size = min(
            len(category_data),
            MAX_EXAMPLES_PER_CATEGORY,
        )

        sampled = category_data.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )

        balanced_parts.append(sampled)

    dataframe = pd.concat(
        balanced_parts,
        ignore_index=True,
    )

    dataframe = dataframe.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\nDataset creation completed.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Final number of examples: {len(dataframe)}")

    print("\nExamples per category:")
    print(
        dataframe["clause_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOriginal LEDGAR labels used:")
    used_labels = (
        dataframe.groupby(["clause_type", "original_label"])
        .size()
        .reset_index(name="examples")
        .sort_values(["clause_type", "examples"], ascending=[True, False])
    )

    print(used_labels.to_string(index=False))


if __name__ == "__main__":
    main()