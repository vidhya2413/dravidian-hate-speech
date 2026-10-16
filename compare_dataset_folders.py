from pathlib import Path
import pandas as pd

ROOT = Path("data")

for language in ["tamil", "malayalam"]:
    print("\n" + "=" * 60)
    print(language.upper())

    for split in ["train", "dev", "test"]:
        cleaned_path = ROOT / "cleaned" / f"{language}_{split}.csv"
        input_path = ROOT / "model_input" / f"{language}_{split}.csv"

        cleaned = pd.read_csv(cleaned_path)
        model_input = pd.read_csv(input_path)

        print(f"\n{split.upper()}")
        print("Cleaned rows:", len(cleaned))
        print("Model-input rows:", len(model_input))
        print("Cleaned columns:", list(cleaned.columns))
        print("Model-input columns:", list(model_input.columns))

        if {"text", "label"}.issubset(cleaned.columns) and {
            "text", "label"
        }.issubset(model_input.columns):
            def normalize(df):
                return pd.DataFrame({
                    "text": (
                        df["text"].fillna("").astype(str)
                        .str.lower().str.split().str.join(" ")
                    ),
                    "label": df["label"].astype(str),
                })

            a = normalize(cleaned)
            b = normalize(model_input)

            print("Rows with text and label matching between folders:",
                  len(a.merge(b.drop_duplicates(), on=["text", "label"])))

print("\nComparison complete. No files modified.")