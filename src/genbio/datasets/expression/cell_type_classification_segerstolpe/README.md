# Cell Type Classification Segerstolpe

Unique Identifier: `cell-type-classification-segerstolpe`
Input: Single-cell gene expression data (h5ad format)
Target: Cell type classification (13 classes)
Primary Metric: Macro F1 Score

## Dataset Description

The Segerstolpe dataset contains single-cell RNA sequencing data from human pancreatic islets in health and type 2 diabetes. The task is to classify cells into one of 13 different cell types based on their gene expression profiles.

**Source:** Segerstolpe et al. 2016
**Splits:** Following Ho et al. 2024

## Data Format

- **Input:** AnnData object containing:
  - `X`: Gene expression matrix (cells × genes), shape (1279, 19264) for train, (427, 19264) for test
  - `obs['cell_type_label']`: Integer labels 0-12 representing cell types

- **Output:** Predicted cell type labels (integers 0-12) in `preds.obs['cell_type_label']`

- **Classes:** 13 pancreatic cell types (encoded as integers 0-12)

## Folds

| Fold ID | Train Size | Test Size | Total Genes |
|---------|------------|-----------|-------------|
| 0       | 1,279      | 427       | 19,264      |

## Usage

```python
import genbio.leaderboard as gl
import numpy as np

# Select a dataset and fold, and provide a username
task = gl.BenchmarkTask(name='expression/cell-type-classification-segerstolpe', fold='0', user='your_name')

# Load the training and test data
train, test = task.setup()
print(train.shape)
print(test.shape)

# Build your model, make predictions
preds_train = train.copy()
preds_train.obs['cell_type_label'] = np.random.permutation(preds_train.obs['cell_type_label'].values)

preds_test = test.copy()
preds_test.obs['cell_type_label'] = np.random.permutation(preds_test.obs['cell_type_label'].values)

# Compute intermediate train metrics
task.evaluate(preds_train, train)

# Make a submission
task.submit(preds_test, 'Dummy submission')
```

## Citation

```
@article{segerstolpe2016,
  title={Single-cell transcriptome profiling of human pancreatic islets in health and type 2 diabetes},
  author={Segerstolpe, {\AA}sa and Palasantza, Athanasia and Eliasson, Pernilla and Andersson, Eva-Marie and Andr{\'e}asson, Anne-Christine and Sun, Xiaoyan and Picelli, Simone and Sabirsh, Alan and Clausen, Maryam and Bjursell, Magnus K and others},
  journal={Cell metabolism},
  volume={24},
  number={4},
  pages={593--607},
  year={2016},
  publisher={Elsevier}
}
```
