
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/model_input")

for language in ["tamil", "malayalam"]:
    print("\n" + "=" * 60)
    print(f"{language.upper()} DUPLICATE AND LABEL CHECK")
    print("=" * 60)

    frames = {}

    for split in ["train", "dev", "test"]:
        path = DATA_DIR / f"{language}_{split}.csv"
        df = pd.read_csv(path)

        text_col = next(
            (c for c in ["text", "comment", "sentence", "Text", "Comment"]
             if c in df.columns),
            None
        )
        label_col = next(
            (c for c in ["label", "labels", "task_1", "Label", "category"]
             if c in df.columns),
            None
        )

        if text_col is None or label_col is None:
            raise ValueError(
                f"{path.name}: columns found: {list(df.columns)}"
            )

        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "label"]
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df["label"] = df["label"].fillna("").astype(str).str.strip()
        df = df[df["text"] != ""]

        frames[split] = df

        conflicting = (
            df.groupby("text")["label"].nunique() > 1
        ).sum()

        print(
            f"{split.upper()}: repeated rows = "
            f"{len(df) - df['text'].nunique()}, "
            f"within-split texts with conflicting labels = {conflicting}"
        )

    for first, second in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        a = frames[first].drop_duplicates(["text", "label"])
        b = frames[second].drop_duplicates(["text", "label"])

        merged = a.merge(
            b, on="text", suffixes=("_first", "_second")
        )

        same_label = (
            merged["label_first"] == merged["label_second"]
        )
        conflicts = merged.loc[~same_label, "text"].nunique()

        print(
            f"{first.upper()} vs {second.upper()}: "
            f"shared texts = {merged['text'].nunique()}, "
            f"conflicting-label texts = {conflicts}"
        )
