import pandas as pd
from pathlib import Path

data_dir = Path("data") / "model_input"

for language in ["tamil", "malayalam"]:
    print(f"\n{'=' * 45}")
    print(f"LANGUAGE: {language.upper()}")
    print("=" * 45)

    for split in ["train", "dev", "test"]:
        file_path = data_dir / f"{language}_{split}.csv"
        df = pd.read_csv(file_path)

        print(f"\n{split.upper()} — {len(df)} rows")
        print(df["label"].value_counts().to_string())