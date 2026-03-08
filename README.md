# GenBio Leaderboard

## Quickstart

### Installation
```bash
git clone https://github.com/cnellington/genbio-leaderboard.git
cd genbio-leaderboard
pip install .
```

### Usage

#### Python SDK
The main functions are:
- `setup(name, fold, user)`: Select a dataset and fold, and provide a username.
- `describe()`: Print dataset README and documentation for load, evaluate, and submit functions.
- `load()`: Load the training and test data for the selected dataset and fold.
- `evaluate(preds, targets)`: Compute metrics given predictions and ground truth.
- `submit(preds)`: Submit predictions to the leaderboard, get final test metrics.

Code example:
```python
import genbio.leaderboard as gl

# Select a dataset and fold, and provide a username
task = gl.BenchmarkTask(name='RNA/translation-efficiency-muscle', fold='0', user='caleb.ellington')

# Print dataset description and documentation
task.describe()

# Load the training and test data
train_df, test_df = task.setup()

# Build your model, make predictions
mean_pred = train_df['labels'].mean()
train_pred_df = train_df.copy()
train_pred_df['labels'] = mean_pred
test_pred_df = test_df.copy()
test_pred_df['labels'] = mean_pred

# Compute intermediate train metrics
task.evaluate(train_pred_df, train_df)

# Make a submission
task.submit(test_pred_df, name='mean_predictor_v0.2', description='Dummy submission using mean prediction')
```

#### CLI Tools
View leaderboard and submission history:

```bash
# Display leaderboard for a dataset and fold
genbio-leaderboard leaderboard --dataset RNA/translation-efficiency-muscle --fold 0

# Display submission history for a specific user
genbio-leaderboard history --dataset RNA/translation-efficiency-muscle --fold 0 --user caleb.ellington
```

## Adding New Datasets
Add new datasets under `src/genbio/datasets/`. Each entry should include
- `README.md` ([example](datasets/RNA/translation_efficiency_muscle/README.md)): Description of the dataset (source, inputs, targets, size, folds, etc.)

- `load.py` ([example](datasets/RNA/translation_efficiency_muscle/load.py)): Takes a fold and returns a train and test set. Downloads and prepares the dataset for the first time if necessary.

- `evaluate.py` ([example](datasets/RNA/translation_efficiency_muscle/evaluate.py)): Implements a function `evaluate(...)` which computes metrics given predictions and ground truth. Must return a dictionary with keys for each metric and a `primary_metric`.

- `__init__.py` (can leave empty)
