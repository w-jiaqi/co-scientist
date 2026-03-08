import pandas as pd
import torch
from torchmetrics.regression import (
    MeanSquaredError,
    MeanAbsoluteError,
    PearsonCorrCoef,
    SpearmanCorrCoef,
    R2Score,
)


def evaluate(preds: pd.DataFrame, targets: pd.DataFrame) -> dict[str, float]:
    """Evaluate predictions against labels using multiple regression metrics.

    Note: 
        Primary metric is Spearman correlation.
    
    Args:
        preds (pd.DataFrame): DataFrame containing model predictions with a 'labels' column in the same order as the targets.
        targets (pd.DataFrame): DataFrame containing true labels with a 'labels' column.

    Returns:
        dict[str, float]: A dictionary containing the following keys:
            - primary_metric: 'spearman'
            - spearman: Spearman correlation
            - pearson: Pearson correlation
            - mse: Mean Squared Error
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - r2: R-squared score
    """
    preds = torch.tensor(preds['labels'].to_numpy())
    targets = torch.tensor(targets['labels'].to_numpy())
    assert len(preds) == len(targets), "Predictions and targets must have the same length."

    MSE = MeanSquaredError()
    MAE = MeanAbsoluteError()
    Pearson = PearsonCorrCoef()
    Spearman = SpearmanCorrCoef()
    R2 = R2Score()

    mse = MSE(preds, targets).item()
    mae = MAE(preds, targets).item()
    pearson = Pearson(preds, targets).item()
    spearman = Spearman(preds, targets).item()
    r2 = R2(preds, targets).item()
    rmse = mse ** 0.5

    return {
        'primary_metric': 'spearman',
        'spearman': spearman,
        'pearson': pearson,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
    }