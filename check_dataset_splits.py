from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/model_input")

for language in ["tamil", "malayalam"]:
    print("\n" + "=" * 60)
    print("LANGUAGE:", language.upper())

    datasets = {}

    for split in ["train", "dev", "test"]:
        path = DATA_DIR / f"{language}_{split}.csv"

        if not path.exists():
            print(f"\nMissing file: {path}")
            continue

        df = pd.read_csv(path)

        print(f"\n{split.upper()} rows: {len(df)}")
        print("\nLabel counts:")

        if "label" not in df.columns or "text" not in df.columns:
            print("ERROR: Expected 'text' and 'label' columns.")
            print("Available columns:", list(df.columns))
            continue

        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(str)

        print(df["label"].value_counts().to_string())

        print("\nLabel percentages:")
        print(
            (df["label"].value_counts(normalize=True) * 100)
            .round(2)
            .to_string()
        )

        df["normalized_text"] = (
            df["text"].str.lower().str.split().str.join(" ")
        )

        datasets[split] = df

    for first, second in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        if first in datasets and second in datasets:
            a = set(datasets[first]["normalized_text"])
            b = set(datasets[second]["normalized_text"])

            overlap = (a & b) - {""}

            print(
                f"\nDuplicate comments in {first} and {second}: "
                f"{len(overlap)}"
            )

print("\nDataset diagnostic complete. No dataset files modified.")