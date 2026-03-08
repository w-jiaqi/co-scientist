import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate(preds: sc.AnnData, targets: sc.AnnData) -> dict[str, float]:
    """Evaluate cell type classification predictions for Segerstolpe dataset.

    Note:
        Primary metric is macro F1.

    Args:
        preds (sc.AnnData): AnnData object containing predicted cell type labels
            in preds.obs['cell_type_label']. Must contain integer labels 0-12.
        targets (sc.AnnData): AnnData object containing true cell type labels
            in targets.obs['cell_type_label']. Contains integer labels 0-12.

    Returns:
        dict[str, float]: A dictionary containing the following keys:
            - primary_metric: 'f1_macro'
            - f1_macro: Macro-averaged F1 score across all 13 cell types
            - f1_weighted: Weighted F1 score
            - accuracy: Overall classification accuracy
            - precision_macro: Macro-averaged precision
            - recall_macro: Macro-averaged recall
    """
    # Extract predictions and targets from the specific field used by Segerstolpe
    y_pred = preds.obs['cell_type_label'].values
    y_true = targets.obs['cell_type_label'].values

    # Validate same length
    assert len(y_pred) == len(y_true), f"Predictions and targets must have the same length. Got {len(y_pred)} and {len(y_true)}"

    # Calculate metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'primary_metric': 'f1_macro',
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
    }
