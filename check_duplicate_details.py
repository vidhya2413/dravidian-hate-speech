from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/model_input")

for language in ["tamil", "malayalam"]:
    print("\n" + "=" * 65)
    print("LANGUAGE:", language.upper())

    datasets = {}

    for split in ["train", "dev", "test"]:
        path = DATA_DIR / f"{language}_{split}.csv"
        df = pd.read_csv(path)

        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(str)

        df["normalized_text"] = (
            df["text"].str.lower().str.split().str.join(" ")
        )

        # Ignore empty comments
        df = df[df["normalized_text"] != ""].copy()
        datasets[split] = df

    pairs = [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]

    for first, second in pairs:
        a = datasets[first][
            ["normalized_text", "label"]
        ].drop_duplicates()

        b = datasets[second][
            ["normalized_text", "label"]
        ].drop_duplicates()

        overlap = a.merge(
            b,
            on="normalized_text",
            how="inner",
            suffixes=("_first", "_second"),
        )

        same_label = (
            overlap["label_first"] == overlap["label_second"]
        )

        print(f"\n{first.upper()} vs {second.upper()}")
        print("Overlapping text-label combinations:", len(overlap))
        print("Matching labels:", int(same_label.sum()))
        print("Different labels:", int((~same_label).sum()))

        if (~same_label).any():
            print("\nExamples with different labels:")
            print(
                overlap.loc[
                    ~same_label,
                    [
                        "normalized_text",
                        "label_first",
                        "label_second",
                    ],
                ].head(5).to_string(index=False)
            )

print("\nDuplicate-label diagnostic complete.")
print("No dataset files modified.")