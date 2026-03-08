"""Agent workspace scaffolding: prepares data, instructions, and starter code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from genbio.coscientist.adapters.base import DatasetAdapter
from genbio.coscientist.config import DataProfile, DataType, RunConfig


class WorkspaceScaffolder:
    """Creates a self-contained workspace for an AI coding agent."""

    def __init__(
        self,
        config: RunConfig,
        adapter: DatasetAdapter,
        profile: DataProfile,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.profile = profile

    def scaffold(
        self,
        train_data: Any,
        test_data: Any,
        baseline_score: float | None = None,
    ) -> Path:
        ws = self.config.run_dir / "agent_workspace"
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "data").mkdir(exist_ok=True)
        (ws / "output").mkdir(exist_ok=True)

        self.adapter.save_train_data(train_data, ws / "data" / "train.pkl")
        self.adapter.save_test_features(test_data, ws / "data" / "test_features.pkl")

        profile_path = ws / "data" / "profile.json"
        profile_path.write_text(json.dumps(self.profile.to_dict(), indent=2))

        if baseline_score is not None:
            (ws / "baseline_score.json").write_text(
                json.dumps({"baseline_score": baseline_score, "metric": self._primary_metric()}, indent=2)
            )

        self._write_instructions(ws, baseline_score)
        self._write_starter(ws)
        self._write_evaluate_locally(ws)

        return ws

    def _primary_metric(self) -> str:
        from genbio.coscientist.config import TaskType
        if self.profile.task_type == TaskType.REGRESSION:
            return "spearman"
        return "f1_macro"

    def _write_instructions(self, ws: Path, baseline_score: float | None) -> None:
        p = self.profile
        metric = self._primary_metric()

        instructions = f"""# Co-Scientist Agent Instructions

## Task
Build the best possible ML model for this dataset.

## Dataset Details
- **Data type**: {p.data_type.value}
- **Task type**: {p.task_type.value}
- **Train samples**: {p.train_samples}
- **Test samples**: {p.test_samples}
- **Primary metric**: {metric} (higher is better)
"""
        if p.n_features:
            instructions += f"- **Features**: {p.n_features}\n"
        if p.n_classes:
            instructions += f"- **Classes**: {p.n_classes}\n"
        if p.sequence_column:
            instructions += f"- **Input**: DNA/RNA sequences in column '{p.sequence_column}'\n"
        if p.label_column:
            instructions += f"- **Label column**: {p.label_column}\n"

        if baseline_score is not None:
            instructions += f"""
## Baseline to Beat
The built-in pipeline achieved **{metric} = {baseline_score:.4f}**.
Your goal is to beat this score.
"""

        instructions += f"""
## Data Location
- Training data: `data/train.pkl` (or `data/train.h5ad` for AnnData)
- Test features: `data/test_features.pkl` (or `data/test_features.h5ad`)
- Data profile: `data/profile.json`

## Required Output
Save these files to the `output/` directory:
1. `output/model.joblib` - trained model
2. `output/predictions.pkl` - predictions on test features
3. `output/train.py` - training script (for reproducibility)
4. `output/predict.py` - inference script

## Evaluation
Run `python evaluate_locally.py` to check your model's performance on a validation split.

## Constraints
- Do NOT look at or use test labels (they are not provided)
- Save all artifacts to `output/`
- Use `starter.py` as a reference for loading data and making predictions
- Focus on maximizing {metric}
"""

        (ws / "INSTRUCTIONS.md").write_text(instructions)

    def _write_starter(self, ws: Path) -> None:
        p = self.profile

        if p.data_type == DataType.ANNDATA:
            starter = '''"""Starter script: loads data, trains a baseline, evaluates."""
import numpy as np
import joblib
import scanpy as sc
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Load data
train = sc.read_h5ad("data/train.h5ad")
X = np.asarray(train.X.toarray() if hasattr(train.X, "toarray") else train.X, dtype=np.float32)
y = train.obs["cell_type_label"].values

# Preprocess
X_norm = normalize(X, norm="l1", axis=1) * 1e4
X_log = np.log1p(X_norm)

# Build pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=50, random_state=42)),
    ("model", LogisticRegression(C=1.0, max_iter=2000, multi_class="multinomial",
                                  solver="saga", class_weight="balanced", random_state=42)),
])

# Evaluate
scores = cross_val_score(pipe, X_log, y, cv=3, scoring="f1_macro")
print(f"CV macro F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Train on full data
pipe.fit(X_log, y)

# Save
joblib.dump(pipe, "output/model.joblib")

# Predict test
test = sc.read_h5ad("data/test_features.h5ad")
X_test = np.asarray(test.X.toarray() if hasattr(test.X, "toarray") else test.X, dtype=np.float32)
X_test_norm = normalize(X_test, norm="l1", axis=1) * 1e4
X_test_log = np.log1p(X_test_norm)
preds = pipe.predict(X_test_log)

import pickle
with open("output/predictions.pkl", "wb") as f:
    pickle.dump(preds, f)

print(f"Predictions saved. Shape: {preds.shape}")
'''
        elif p.sequence_column:
            starter = '''"""Starter script: loads data, trains a baseline, evaluates."""
import numpy as np
import joblib
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_pickle("data/train.pkl")
X_train = train_df["sequence"]
y_train = train_df["labels"].to_numpy()

# Build pipeline
pipe = Pipeline([
    ("kmer", CountVectorizer(analyzer="char", ngram_range=(3, 6))),
    ("scale", StandardScaler(with_mean=False)),
    ("model", Ridge(alpha=1.0)),
])

# Evaluate (use negative MSE as proxy; we compute Spearman separately)
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)
spearman_scores = []
for train_idx, val_idx in kf.split(X_train):
    Xt, Xv = X_train.iloc[train_idx], X_train.iloc[val_idx]
    yt, yv = y_train[train_idx], y_train[val_idx]
    pipe.fit(Xt, yt)
    preds = pipe.predict(Xv)
    corr, _ = spearmanr(preds, yv)
    spearman_scores.append(corr)
print(f"CV Spearman: {np.mean(spearman_scores):.4f} (+/- {np.std(spearman_scores):.4f})")

# Train on full data
pipe.fit(X_train, y_train)
joblib.dump(pipe, "output/model.joblib")

# Predict test
test_df = pd.read_pickle("data/test_features.pkl")
preds = pipe.predict(test_df["sequence"])
with open("output/predictions.pkl", "wb") as f:
    pickle.dump(preds, f)

print(f"Predictions saved. Shape: {preds.shape}")
'''
        else:
            starter = '''"""Starter script: loads data, trains a baseline, evaluates."""
import numpy as np
import joblib
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_pickle("data/train.pkl")
label_col = "labels"
feature_cols = [c for c in train_df.columns if c not in (label_col, "fold_id")]
X_train = train_df[feature_cols].to_numpy(dtype="float32")
y_train = train_df[label_col].to_numpy()

# Build pipeline
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])

# Evaluate
scores = cross_val_score(pipe, X_train, y_train, cv=3)
print(f"CV R2: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Train on full data
pipe.fit(X_train, y_train)
joblib.dump(pipe, "output/model.joblib")

# Predict test
test_df = pd.read_pickle("data/test_features.pkl")
preds = pipe.predict(test_df[feature_cols].to_numpy(dtype="float32"))
with open("output/predictions.pkl", "wb") as f:
    pickle.dump(preds, f)

print(f"Predictions saved. Shape: {preds.shape}")
'''

        (ws / "starter.py").write_text(starter)

    def _write_evaluate_locally(self, ws: Path) -> None:
        p = self.profile

        script = '''"""Local evaluation: run on a validation split to check performance."""
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

'''
        if p.data_type == DataType.ANNDATA:
            script += '''
import scanpy as sc
train = sc.read_h5ad("data/train.h5ad")
X = np.asarray(train.X.toarray() if hasattr(train.X, "toarray") else train.X, dtype=np.float32)
y = train.obs["cell_type_label"].values
'''
        elif p.sequence_column:
            script += '''
train_df = pd.read_pickle("data/train.pkl")
X = train_df["sequence"]
y = train_df["labels"].to_numpy()
'''
        else:
            script += '''
train_df = pd.read_pickle("data/train.pkl")
label_col = "labels"
feature_cols = [c for c in train_df.columns if c not in (label_col, "fold_id")]
X = train_df[feature_cols].to_numpy(dtype="float32")
y = train_df[label_col].to_numpy()
'''

        script += '''
import joblib
try:
    model = joblib.load("output/model.joblib")
except FileNotFoundError:
    print("No model found at output/model.joblib. Train first.")
    exit(1)

# Quick validation split
'''
        if p.task_type.value == "classification":
            script += '''
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(X if isinstance(X, np.ndarray) else np.zeros(len(X)), y):
    if isinstance(X, np.ndarray):
        Xv = X[val_idx]
    else:
        Xv = X.iloc[val_idx]
    yv = y[val_idx]
    preds = model.predict(Xv)
    scores.append(f1_score(yv, preds, average="macro", zero_division=0))
print(f"Validation macro F1: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
'''
        else:
            script += '''
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(X if isinstance(X, np.ndarray) else np.zeros(len(X))):
    if isinstance(X, np.ndarray):
        Xv = X[val_idx]
    else:
        Xv = X.iloc[val_idx]
    yv = y[val_idx]
    preds = model.predict(Xv)
    corr, _ = spearmanr(preds, yv)
    scores.append(corr)
print(f"Validation Spearman: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
'''

        (ws / "evaluate_locally.py").write_text(script)
