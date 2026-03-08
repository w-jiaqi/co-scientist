import pandas as pd
from datasets import load_dataset


def load(fold_id: str) -> dict[str, pd.DataFrame]:
    """Load the translation efficiency Muscle dataset for a specific fold.

    Args:
        fold_id (str): The fold ID of the test set. 0-9.

    Returns:
        dict[str, pd.DataFrame]: A dictionary keys 'train' and 'test', each containing DataFrames with columns 'sequence', 'labels', and 'fold_id'.
    """
    # Download dataset from HuggingFace
    dataset = load_dataset(
        "genbio-ai/rna-downstream-tasks",
        "translation_efficiency_Muscle",
        split="train"
    )

    # Convert to pandas DataFrame
    df = dataset.to_pandas()

    # Filter to the specified fold
    fold_id = int(fold_id)
    train_df = df[df['fold_id'] != fold_id]
    test_df = df[df['fold_id'] == fold_id]

    return {
        "train": train_df,
        "test": test_df,
    }