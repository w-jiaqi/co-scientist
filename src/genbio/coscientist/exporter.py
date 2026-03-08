"""Portable project exporter: generates standalone train.py, predict.py, etc."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from genbio.coscientist.config import DataProfile, DataType, ExperimentResult, RunConfig, TaskType
from genbio.coscientist.planner import ExperimentPlan


class ProjectExporter:
    """Exports a self-contained project from a completed run."""

    def __init__(
        self,
        config: RunConfig,
        profile: DataProfile,
        best: ExperimentResult,
        plan: ExperimentPlan | None = None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.best = best
        self.plan = plan

    def export(self) -> Path:
        export_dir = self.config.run_dir / "exported_project"
        export_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = export_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        if self.best.model_path and self.best.model_path.exists():
            shutil.copy2(self.best.model_path, artifacts_dir / "model.joblib")

        config_path = artifacts_dir / "pipeline_config.json"
        config_path.write_text(json.dumps(self.best.config, indent=2))

        self._write_train_script(export_dir)
        self._write_predict_script(export_dir)
        self._write_requirements(export_dir)
        self._write_readme(export_dir)
        self._write_config(export_dir)

        return export_dir

    def _write_train_script(self, export_dir: Path) -> None:
        p = self.profile
        pipeline_name = self.best.pipeline_name

        if p.data_type == DataType.DATAFRAME_SEQUENCE:
            script = _TRAIN_SEQUENCE.format(
                pipeline_name=pipeline_name,
                config_repr=repr(self.best.config),
            )
        elif p.data_type == DataType.ANNDATA:
            script = _TRAIN_ANNDATA.format(
                pipeline_name=pipeline_name,
                config_repr=repr(self.best.config),
            )
        else:
            script = _TRAIN_TABULAR.format(
                pipeline_name=pipeline_name,
                config_repr=repr(self.best.config),
            )

        (export_dir / "train.py").write_text(script)

    def _write_predict_script(self, export_dir: Path) -> None:
        if self.profile.data_type == DataType.ANNDATA:
            script = _PREDICT_ANNDATA
        else:
            script = _PREDICT_DATAFRAME.format(label_col=self.profile.label_column)

        (export_dir / "predict.py").write_text(script)

    def _write_requirements(self, export_dir: Path) -> None:
        reqs = [
            "numpy",
            "pandas",
            "scikit-learn>=1.3",
            "joblib",
        ]
        if self.profile.data_type == DataType.ANNDATA:
            reqs.extend(["anndata", "scanpy", "scipy"])
        if self.profile.data_type == DataType.DATAFRAME_SEQUENCE:
            reqs.append("scipy")

        (export_dir / "requirements.txt").write_text("\n".join(reqs) + "\n")

    def _write_readme(self, export_dir: Path) -> None:
        readme = f"""# Co-Scientist Exported Model

## Quick Start

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py --dataset <dataset_name> --fold <fold_id>
```

### Inference

```bash
python predict.py --model artifacts/model.joblib --input <input_file> --output predictions.csv
```

## Model Details

- **Pipeline**: {self.best.pipeline_name}
- **Best CV {self.best.primary_metric_name}**: {self.best.primary_metric_value:.4f}
- **Data type**: {self.profile.data_type.value}
- **Task type**: {self.profile.task_type.value}
"""
        (export_dir / "README.md").write_text(readme)

    def _write_config(self, export_dir: Path) -> None:
        self.config.save(export_dir / "config.yaml")


def run_inference(run_dir: Path, input_path: Path, output_path: Path) -> None:
    """Run inference using a previously exported model."""
    import joblib

    model_path = run_dir / "exported_project" / "artifacts" / "model.joblib"
    if not model_path.exists():
        model_path = run_dir / "best_model" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {run_dir}")

    model = joblib.load(model_path)

    suffix = input_path.suffix.lower()
    if suffix in (".h5ad",):
        import scanpy as sc
        data = sc.read_h5ad(input_path)
        import numpy as np
        from sklearn.preprocessing import normalize
        X = np.asarray(data.X.toarray() if hasattr(data.X, "toarray") else data.X, dtype=np.float32)
        X_norm = normalize(X, norm="l1", axis=1) * 1e4
        X_log = np.log1p(X_norm)
        preds = model.predict(X_log)
        data.obs["cell_type_label"] = preds
        data.write_h5ad(output_path.with_suffix(".h5ad"))
    else:
        import pandas as pd
        df = pd.read_csv(input_path)
        if "sequence" in df.columns:
            preds = model.predict(df["sequence"])
        else:
            feature_cols = [c for c in df.columns if c not in ("labels", "label", "fold_id")]
            preds = model.predict(df[feature_cols].to_numpy(dtype="float32"))
        df["predictions"] = preds
        df.to_csv(output_path, index=False)


# --- Script templates ---

_TRAIN_SEQUENCE = '''"""Train script for sequence regression model."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def gc_content(sequences):
    vals = []
    for seq in sequences:
        s = seq.upper()
        gc = (s.count("G") + s.count("C")) / max(len(s), 1)
        vals.append([gc])
    return np.array(vals, dtype=np.float32)


def seq_length(sequences):
    return np.array([[len(s)] for s in sequences], dtype=np.float32)


def build_pipeline(config):
    ngram_lo = config.get("ngram_lo", 3)
    ngram_hi = config.get("ngram_hi", 6)
    alpha = config.get("alpha", 1.0)

    transformers = [
        ("kmer", CountVectorizer(analyzer="char", ngram_range=(ngram_lo, ngram_hi))),
        ("gc", FunctionTransformer(gc_content, validate=False)),
        ("seqlen", FunctionTransformer(seq_length, validate=False)),
    ]

    return Pipeline([
        ("features", FeatureUnion(transformers)),
        ("scale", StandardScaler(with_mean=False)),
        ("model", Ridge(alpha=alpha)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--output", default="artifacts/model.joblib")
    args = parser.parse_args()

    from datasets import load_dataset
    dataset = load_dataset("genbio-ai/rna-downstream-tasks", args.dataset, split="train")
    df = dataset.to_pandas()
    fold_id = int(args.fold)
    train_df = df[df["fold_id"] != fold_id]

    config = {config_repr}
    pipe = build_pipeline(config)
    pipe.fit(train_df["sequence"], train_df["labels"].to_numpy())

    joblib.dump(pipe, args.output)
    print(f"Model saved to {{args.output}}")


if __name__ == "__main__":
    main()
'''

_TRAIN_ANNDATA = '''"""Train script for expression classification model."""
import argparse
import json
import joblib
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def preprocess(X, target_sum=1e4):
    X_norm = normalize(X, norm="l1", axis=1) * target_sum
    return np.log1p(X_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.h5ad")
    parser.add_argument("--output", default="artifacts/model.joblib")
    args = parser.parse_args()

    import scanpy as sc
    train = sc.read_h5ad(args.train)
    X = np.asarray(train.X.toarray() if hasattr(train.X, "toarray") else train.X, dtype=np.float32)
    y = train.obs["cell_type_label"].values

    config = {config_repr}
    X_proc = preprocess(X)

    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True)),
        ("pca", PCA(n_components=config.get("n_components", 50), random_state=42)),
        ("model", LogisticRegression(
            C=config.get("C", 1.0), max_iter=2000,
            solver="saga", class_weight="balanced", random_state=42,
        )),
    ])
    pipe.fit(X_proc, y)

    joblib.dump(pipe, args.output)
    print(f"Model saved to {{args.output}}")


if __name__ == "__main__":
    main()
'''

_TRAIN_TABULAR = '''"""Train script for tabular model."""
import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to training CSV")
    parser.add_argument("--label-col", default="labels")
    parser.add_argument("--output", default="artifacts/model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    feature_cols = [c for c in df.columns if c not in (args.label_col, "fold_id")]
    X = df[feature_cols].to_numpy(dtype="float32")
    y = df[args.label_col].to_numpy()

    config = {config_repr}

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=config.get("alpha", 1.0))),
    ])
    pipe.fit(X, y)

    joblib.dump(pipe, args.output)
    print(f"Model saved to {{args.output}}")


if __name__ == "__main__":
    main()
'''

_PREDICT_DATAFRAME = '''"""Inference script for DataFrame-based models."""
import argparse
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)

    if "sequence" in df.columns:
        preds = model.predict(df["sequence"])
    else:
        feature_cols = [c for c in df.columns if c not in ("{label_col}", "fold_id")]
        preds = model.predict(df[feature_cols].to_numpy(dtype="float32"))

    df["predictions"] = preds
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {{args.output}}")


if __name__ == "__main__":
    main()
'''

_PREDICT_ANNDATA = '''"""Inference script for AnnData-based models."""
import argparse
import joblib
import numpy as np
from sklearn.preprocessing import normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--input", required=True, help="Input h5ad file")
    parser.add_argument("--output", default="predictions.h5ad")
    args = parser.parse_args()

    import scanpy as sc
    model = joblib.load(args.model)
    data = sc.read_h5ad(args.input)

    X = np.asarray(data.X.toarray() if hasattr(data.X, "toarray") else data.X, dtype=np.float32)
    X_norm = normalize(X, norm="l1", axis=1) * 1e4
    X_log = np.log1p(X_norm)

    preds = model.predict(X_log)
    data.obs["cell_type_label"] = preds
    data.write_h5ad(args.output)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
'''
