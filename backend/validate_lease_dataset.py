from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "ml_model" / "lease_clause_type_dataset.csv"

EXPECTED_COLUMNS = {
    "clause_text",
    "clause_type",
    "source",
    "contract_type",
}

ALLOWED_TYPES = {
    "Lease Grant",
    "Term and Renewal",
    "Property / Premises",
    "Repairs / Maintenance",
    "Use of Premises",
    "Taxes / Utilities",
    "Alterations / Improvements",
    "Quiet Enjoyment",
    "Possession / Surrender",
    "Other / Unknown",
}


def main():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)

    missing_columns = EXPECTED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df.dropna(subset=list(EXPECTED_COLUMNS)).copy()

    for column in EXPECTED_COLUMNS:
        df[column] = df[column].astype(str).str.strip()

    empty_rows = df[
        (df["clause_text"] == "")
        | (df["clause_type"] == "")
        | (df["source"] == "")
        | (df["contract_type"] == "")
    ]

    invalid_types = sorted(
        set(df["clause_type"]) - ALLOWED_TYPES
    )

    duplicate_rows = df[
        df.duplicated(
            subset=["clause_text", "clause_type"],
            keep=False,
        )
    ]

    short_rows = df[
        df["clause_text"].str.split().str.len() < 6
    ]

    print(f"Total rows: {len(df)}")
    print(f"Categories: {df['clause_type'].nunique()}")

    print("\nExamples per category:")
    print(
        df["clause_type"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\nEmpty rows: {len(empty_rows)}")
    print(f"Duplicate rows: {len(duplicate_rows)}")
    print(f"Very short clauses: {len(short_rows)}")

    if invalid_types:
        print("\nInvalid clause types:")
        for clause_type in invalid_types:
            print(f"- {clause_type}")
    else:
        print("\nAll clause types are valid.")

    if not duplicate_rows.empty:
        print("\nDuplicate examples:")
        print(
            duplicate_rows[
                ["clause_text", "clause_type", "source"]
            ].to_string(index=False)
        )

    if not short_rows.empty:
        print("\nVery short examples to review:")
        print(
            short_rows[
                ["clause_text", "clause_type", "source"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()