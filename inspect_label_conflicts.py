
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/model_input")

for language in ["tamil", "malayalam"]:
    print("\n" + "=" * 70)
    print(f"{language.upper()} LABEL CONFLICT EXAMPLES")
    print("=" * 70)

    for split in ["train", "dev", "test"]:
        df = pd.read_csv(DATA_DIR / f"{language}_{split}.csv")

        text_col = next(
            c for c in ["text", "comment", "sentence", "Text", "Comment"]
            if c in df.columns
        )
        label_col = next(
            c for c in ["label", "labels", "task_1", "Label", "category"]
            if c in df.columns
        )

        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df["label"] = df["label"].fillna("").astype(str).str.strip()
        df = df[df["text"] != ""]

        conflicts = (
            df.groupby("text")["label"]
            .agg(lambda labels: sorted(set(labels)))
        )
        conflicts = conflicts[conflicts.map(len) > 1]

        print(f"\n{split.upper()}: {len(conflicts)} conflicting texts")

        for text, labels in conflicts.head(10).items():
            print(f"Text: {text[:250]}")
            print(f"Labels: {labels}")
            print("-" * 40)
