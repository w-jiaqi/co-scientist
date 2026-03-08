import scanpy as sc
from pathlib import Path
import urllib.request


TRAIN_URL = "https://huggingface.co/datasets/genbio-ai/cell-downstream-tasks/resolve/main/Segerstolpe/Segerstolpe_train.h5ad"
TEST_URL = "https://huggingface.co/datasets/genbio-ai/cell-downstream-tasks/resolve/main/Segerstolpe/Segerstolpe_test.h5ad"
CACHE_DIR = Path.home() / ".cache" / "genbio_leaderboard" / "Segerstolpe"


def load(fold_id: str) -> dict[str, sc.AnnData]:
    """Load the Segerstolpe pancreatic cell type classification dataset.

    The Segerstolpe dataset contains single-cell RNA-seq data from human pancreatic islets
    with 13 annotated cell types. This is a single train/test split (fold "0" only).

    Args:
        fold_id (str): Fold identifier. Must be "0" (only one split available).

    Returns:
        dict[str, sc.AnnData]: Dictionary with keys 'train' and 'test':
            - train: AnnData with shape (1279, 19264) - 1,279 cells, 19,264 genes
            - test: AnnData with shape (427, 19264) - 427 cells, 19,264 genes

            Each AnnData object contains:
            - X: Gene expression matrix (cells × genes)
            - obs['cell_type_label']: Integer labels 0-12 representing cell types

    Raises:
        ValueError: If fold_id is not "0"
    """
    if fold_id != "0":
        raise ValueError(f"Segerstolpe only supports fold '0', got '{fold_id}'")

    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_path = CACHE_DIR / "Segerstolpe_train.h5ad"
    test_path = CACHE_DIR / "Segerstolpe_test.h5ad"

    # Download train data if not cached
    if not train_path.exists():
        print(f"Downloading train data from {TRAIN_URL}...")
        urllib.request.urlretrieve(TRAIN_URL, train_path)
        print(f"Train data cached at {train_path}")

    # Download test data if not cached
    if not test_path.exists():
        print(f"Downloading test data from {TEST_URL}...")
        urllib.request.urlretrieve(TEST_URL, test_path)
        print(f"Test data cached at {test_path}")

    # Load data
    train_adata = sc.read_h5ad(train_path)
    test_adata = sc.read_h5ad(test_path)

    return {
        "train": train_adata,
        "test": test_adata,
    }
