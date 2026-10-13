import pandas as pd
from datasets import Dataset, load_dataset

DATASET_NAME = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_artificial"
SUBSET_SIZE = 100
TEXT_COLUMNS = ["question", "long_answer"]


def load_pubmedqa() -> Dataset:
    """Load the full pqa_artificial split (cached on disk after the first run)."""
    # pqa_artificial only ships a "train" split.
    return load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")


def load_subset(dataset: Dataset, size: int = SUBSET_SIZE) -> pd.DataFrame:
    """Take the first `size` instances and convert them to a DataFrame."""
    df = dataset.select(range(size)).to_pandas()
    # to_pandas() is typed as DataFrame | Iterator[DataFrame]; it only returns the
    # iterator when called with batched=True, so this narrows the type for the checker.
    assert isinstance(df, pd.DataFrame)
    return df


def build_combined_text(row: pd.Series) -> str:
    """Step 4: join the labelled context paragraphs and the conclusion into one document."""
    context = row["context"]
    # The question is deliberately left out: it is what users search with, so
    # including it would let every document trivially match its own query.
    parts = [
        f"[{label}] {paragraph}"
        for label, paragraph in zip(context["labels"], context["contexts"])
    ]
    parts.append(f"[CONCLUSION] {row['long_answer']}")
    return "\n".join(parts)


def add_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4: add a combined_text column built from each row."""
    # axis=1 calls the function once per row (the default, axis=0, is once per column).
    df["combined_text"] = df.apply(build_combined_text, axis=1)
    return df


def explore_dataset(dataset: Dataset) -> None:
    """Step 2: print total instances and feature names."""
    print(f"Total instances: {dataset.num_rows:,}")
    print(f"Features: {dataset.column_names}")


def explore_subset(df: pd.DataFrame) -> None:
    """Step 3: report duplicates, missing/empty text, and the label distribution."""
    print(f"\nSubset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    duplicate_count = df["pubid"].duplicated().sum()
    print(f"\nDuplicate pubid values: {duplicate_count}")

    print("\nMissing / empty values:")
    for column in TEXT_COLUMNS:
        missing = df[column].isna().sum()
        # Whitespace-only counts as empty; fillna keeps missing values out of this count.
        empty = (df[column].fillna("x").str.strip() == "").sum()
        print(f"  {column}: {missing} missing, {empty} empty")

    print("\nfinal_decision distribution:")
    counts = df["final_decision"].value_counts()
    for decision, count in counts.items():
        print(f"  {decision}: {count} ({count / len(df):.0%})")
