
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/model_input")
SPLITS = ["train", "dev", "test"]

for language in ["tamil", "malayalam"]:
    print("\n" + "=" * 65)
    print(f"{language.upper()} DUPLICATE-AWARE AUDIT")
    print("=" * 65)

    data = {}

    for split in SPLITS:
        path = DATA_DIR / f"{language}_{split}.csv"
        df = pd.read_csv(path)

        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df["label"] = df["label"].fillna("").astype(str).str.strip()
        df = df[df["text"] != ""].copy()

        data[split] = df

        print(
            f"{split.upper()}: rows={len(df)}, "
            f"unique_texts={df['text'].nunique()}"
        )

    for first, second in [
        ("train", "dev"),
        ("train", "test"),
        ("dev", "test"),
    ]:
        a = data[first]
        b = data[second]

        shared_texts = set(a["text"]) & set(b["text"])

        same_label_count = 0
        conflict_count = 0

        for text in shared_texts:
            labels_a = set(a.loc[a["text"] == text, "label"])
            labels_b = set(b.loc[b["text"] == text, "label"])

            if labels_a & labels_b:
                same_label_count += 1

            if labels_a != labels_b:
                conflict_count += 1

        print(f"\n{first.upper()} vs {second.upper()}")
        print(f"Shared unique texts: {len(shared_texts)}")
        print(f"Texts with at least one matching label: {same_label_count}")
        print(f"Texts with different label sets: {conflict_count}")

print("\nAudit complete. Original dataset files were not modified.")
