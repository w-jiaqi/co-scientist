# Co-Scientist Architecture Guide

This document explains every component, strategy tier, workflow, and configuration option in the co-scientist system in detail.

---

## Table of Contents

1. [Strategy Tiers Explained](#strategy-tiers-explained)
   - [Tier 1: Built-in (`--strategy builtin`)](#tier-1-built-in---strategy-builtin)
   - [Tier 2: LLM-Advised (automatic when API key present)](#tier-2-llm-advised)
   - [Tier 3: Agentic (`--strategy agentic` or `--strategy hybrid`)](#tier-3-agentic---strategy-agentic-or---strategy-hybrid)
2. [Budget Tiers](#budget-tiers)
3. [What is Optuna](#what-is-optuna)
4. [Pipeline Registry: All Models and Hyperparameters](#pipeline-registry)
5. [File-by-File Reference](#file-by-file-reference)

---

## Strategy Tiers Explained

### Tier 1: Built-in (`--strategy builtin`)

**What it is:** The default strategy. Uses pre-built, deterministic scikit-learn pipelines. No API keys, no external tools, no internet access needed at runtime. Everything runs locally.

**When to use:** Always works. Best when you want fast, reliable, reproducible results.

**Complete workflow:**

```
Step 1: LOAD DATASET
   The orchestrator creates a BenchmarkTask from the genbio leaderboard API
   and calls task.setup() to get (train_data, test_data).

Step 2: ANALYZE DATASET
   The analyzer auto-detects the data format:
   - If train_data is a pd.DataFrame → DataFrameAdapter
     - If it has a "sequence"/"sequences" column → DataType.DATAFRAME_SEQUENCE
     - Otherwise → DataType.DATAFRAME_TABULAR
   - If train_data is an sc.AnnData → AnnDataAdapter → DataType.ANNDATA

   It then infers the task type:
   - Float labels with >20 unique values → REGRESSION
   - Otherwise → CLASSIFICATION

   It produces a DataProfile with: sample counts, feature counts, class
   distributions, label statistics, sequence length stats, etc.

Step 3: PLAN EXPERIMENTS
   The planner looks up the PipelineRegistry for pipelines matching
   (data_type, task_type) and creates an ExperimentPlan:

   1. One BASELINE experiment using the first pipeline with default config
      (no tuning, just to establish a floor).
   2. N TUNED experiments (N = budget.max_models), each using a different
      pipeline. Hyperparameters are tuned with Optuna for
      budget.optuna_trials trials.
   3. If budget allows, an ENSEMBLE of the top-2 models.

Step 4: EXECUTE EXPERIMENTS
   For each planned experiment:
   a. Set up cross-validation (uses fold_id if available, else StratifiedKFold
      or KFold with 2-3 splits).
   b. If tuning: run Optuna to find the best hyperparameters by maximizing
      the primary metric (Spearman for regression, macro F1 for classification)
      on cross-validation.
   c. If not tuning: evaluate default config via cross-validation.
   d. Train the final model on ALL training data with the best config.
   e. Save model.joblib, config.json, metrics.json to the run directory.

Step 5: SELECT BEST MODEL
   Pick the experiment with the highest cross-validation primary metric.

Step 6: GENERATE TEST PREDICTIONS
   Load the best model and run predictions on the test set.

Step 7: EVALUATE
   Call task.evaluate(predictions, test_data) using the official
   leaderboard evaluation function. This returns the real test metrics.

Step 8: EXPORT + REPORT
   - Generate report.md with dataset profile, methodology, results, analysis
   - Generate figures (label distribution, model comparison bar chart)
   - Export a portable project: train.py, predict.py, model.joblib,
     requirements.txt, README.md
```

**What the planner produces for each dataset × budget combination:**

#### RNA/translation-efficiency-muscle (DATAFRAME_SEQUENCE, REGRESSION)

Available pipeline pool: `kmer_ridge`, `kmer_elasticnet`, `kmer_hgbr` (3 pipelines)

| | `--budget fast` | `--budget standard` | `--budget thorough` |
|--|-----------------|---------------------|---------------------|
| **Experiments** | 3 | 5 | 6 |
| **Models explored** | 2 distinct pipelines | 3 distinct pipelines | 3 distinct pipelines |
| **Baseline (no tuning)** | 1× kmer_ridge (4 HP at defaults) | 1× kmer_ridge (4 HP at defaults) | 1× kmer_ridge (4 HP at defaults) |
| **Tuned experiments** | kmer_ridge (10 trials, 4 HP), kmer_elasticnet (10 trials, 4 HP) | kmer_ridge (30 trials, 4 HP), kmer_elasticnet (30 trials, 4 HP), kmer_hgbr (30 trials, 7 HP) | kmer_ridge (60 trials, 4 HP), kmer_elasticnet (60 trials, 4 HP), kmer_hgbr (60 trials, 7 HP) |
| **Total Optuna trials** | 20 | 90 | 180 |
| **Total HP configs evaluated** | 21 (1 default + 20 tuned) | 91 (1 default + 90 tuned) | 181 (1 default + 180 tuned) |
| **Ensemble** | No | Yes (avg of top-2) | Yes (avg of top-2) |
| **Typical wall time** | ~1 min | ~5-10 min | ~20-40 min |

Hyperparameters searched per pipeline:
- `kmer_ridge`: `ngram_lo` (2-4), `ngram_hi` (5-8), `alpha` (0.001-1000), `use_engineered` (T/F) → **4 HP**
- `kmer_elasticnet`: `ngram_lo` (2-4), `ngram_hi` (5-8), `alpha` (0.0001-10), `l1_ratio` (0.01-0.99) → **4 HP**
- `kmer_hgbr`: `ngram_lo` (2-4), `ngram_hi` (5-8), `max_iter` (100-500), `learning_rate` (0.01-0.3), `max_depth` (3-10), `min_samples_leaf` (5-50), `l2_regularization` (0.0001-10) → **7 HP**

#### expression/cell-type-classification-segerstolpe (ANNDATA, CLASSIFICATION)

Available pipeline pool: `expr_logreg`, `expr_svm`, `expr_mlp` (3 pipelines)

| | `--budget fast` | `--budget standard` | `--budget thorough` |
|--|-----------------|---------------------|---------------------|
| **Experiments** | 3 | 5 | 6 |
| **Models explored** | 2 distinct pipelines | 3 distinct pipelines | 3 distinct pipelines |
| **Baseline (no tuning)** | 1× expr_logreg (4 HP at defaults) | 1× expr_logreg (4 HP at defaults) | 1× expr_logreg (4 HP at defaults) |
| **Tuned experiments** | expr_logreg (10 trials, 2 HP), expr_svm (10 trials, 2 HP) | expr_logreg (30 trials, 2 HP), expr_svm (30 trials, 2 HP), expr_mlp (30 trials, 5 HP) | expr_logreg (60 trials, 2 HP), expr_svm (60 trials, 2 HP), expr_mlp (60 trials, 5 HP) |
| **Total Optuna trials** | 20 | 90 | 180 |
| **Total HP configs evaluated** | 21 (1 default + 20 tuned) | 91 (1 default + 90 tuned) | 181 (1 default + 180 tuned) |
| **Ensemble** | No | Yes (majority vote of top-2) | Yes (majority vote of top-2) |
| **Typical wall time** | ~7 min | ~20-30 min | ~60+ min |

Hyperparameters searched per pipeline:
- `expr_logreg`: `n_components` (20-200), `C` (0.001-100) → **2 HP**
- `expr_svm`: `n_components` (20-200), `C` (0.001-100) → **2 HP**
- `expr_mlp`: `n_components` (20-200), `hidden1` (128/256/512), `hidden2` (64/128/256), `alpha` (0.00001-0.1), `lr` (0.0001-0.01) → **5 HP**

#### Unknown tabular dataset (DATAFRAME_TABULAR, REGRESSION)

Available pipeline pool: `tabular_ridge`, `tabular_hgbr` (2 regression pipelines)

| | `--budget fast` | `--budget standard` | `--budget thorough` |
|--|-----------------|---------------------|---------------------|
| **Experiments** | 3 | 4 | 4 |
| **Models explored** | 2 distinct pipelines | 2 distinct pipelines | 2 distinct pipelines |
| **Tuned experiments** | ridge (10 trials, 1 HP), hgbr (10 trials, 3 HP) | ridge (30 trials, 1 HP), hgbr (30 trials, 3 HP) | ridge (60 trials, 1 HP), hgbr (60 trials, 3 HP) |
| **Ensemble** | No | Yes | Yes |

#### Unknown tabular dataset (DATAFRAME_TABULAR, CLASSIFICATION)

Available pipeline pool: `tabular_logreg`, `tabular_hgbc`, `tabular_rf` (3 classification pipelines)

| | `--budget fast` | `--budget standard` | `--budget thorough` |
|--|-----------------|---------------------|---------------------|
| **Experiments** | 3 | 5 | 6 |
| **Models explored** | 2 distinct pipelines | 3 distinct pipelines | 3 distinct pipelines |
| **Tuned experiments** | logreg (10 trials, 1 HP), hgbc (10 trials, 3 HP) | logreg (30 trials, 1 HP), hgbc (30 trials, 3 HP), rf (30 trials, 2 HP) | logreg (60 trials, 1 HP), hgbc (60 trials, 3 HP), rf (60 trials, 2 HP) |
| **Ensemble** | No | Yes | Yes |

**Example runs:**

```bash
# Fast: 3 experiments, 20 Optuna trials total, ~1 min
cosci run "RNA/translation-efficiency-muscle" --fold 0 --budget fast

# Standard: 5 experiments + ensemble, 90 Optuna trials total, ~5-10 min
cosci run "RNA/translation-efficiency-muscle" --fold 0 --budget standard

# Thorough: 4 experiments + ensemble, 180 Optuna trials total, ~20-40 min
cosci run "RNA/translation-efficiency-muscle" --fold 0 --budget thorough
```

Example fast run output:
```
Experiments: baseline_kmer_ridge → 0.668 (0.8s)
             tuned_kmer_ridge   → 0.751 (12s)   ← 10 Optuna trials
             tuned_kmer_elnet   → 0.754 (51s)   ← 10 Optuna trials
Best: tuned_kmer_elasticnet  Test Spearman: 0.735
```

---

### Tier 2: LLM-Advised

**What it is:** Not a separate `--strategy` flag. It activates automatically whenever an LLM API key is available in the environment (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`). It enhances the Tier 1 workflow with natural language explanations, suggestions, and report polishing.

**When to use:** Provides richer explanations and more helpful interactive mode. Especially useful with `--mode interactive`.

**What the LLM does (layered on top of Tier 1):**

| Capability | With LLM | Without LLM (fallback) |
|------------|----------|----------------------|
| Dataset interpretation | LLM writes a 3-4 sentence natural language summary of what the dataset is and what challenges to expect | Template-based summary from DataProfile fields |
| Approach suggestion | LLM reasons about which models to try and why | Heuristic: just lists the matching pipelines |
| Results interpretation | LLM analyzes experiment outcomes and explains why one model beat another | Static message: "See the metrics table above" |
| Report polishing | LLM rewrites the report with better prose and scientific rigor | Template report used as-is (still good, just more formulaic) |
| Interactive chat | Free-form Q&A: "why did SVM perform poorly?", "should I try more PCA components?" | Not available — falls back to menu-driven prompts |

**What the LLM does NOT do:**

- It never writes training code. All ML pipelines are pre-built and tested.
- It never makes model selection decisions. The planner uses deterministic heuristics.
- It never touches the evaluation. Official task.evaluate() is always used.

**How it works in interactive mode with LLM:**

```
cosci run "expression/cell-type-classification-segerstolpe" --fold 0 --mode interactive

[After dataset analysis]
> The Advisor says: "This is a 13-class single-cell classification task with
> 1,279 cells and 19,264 genes. Classes are highly imbalanced — class 3
> dominates with 523 cells while classes 0, 11, 12 have only 3 cells each.
> Balanced class weights will be critical for macro F1."

[After planning]
What would you like to do? [proceed/ask/quit]
> ask
Your question: Why not try a neural network?
> Advisor: "With only 1,279 training samples and 13 classes (some with <5
> examples), a neural network is likely to overfit. The logistic regression
> with PCA is a better fit here because it has strong regularization and
> handles small samples well. The MLP pipeline is available with --budget
> thorough if you want to try it anyway."

[After results]
What would you like to do? [accept/ask/quit]
> accept
```

**How it works without LLM (menu-driven fallback):**

```
cosci run "RNA/translation-efficiency-muscle" --fold 0 --mode interactive

[After planning]
What would you like to do? [proceed/ask/quit]
> ask
Your question: should I try a different model?
> LLM not available. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY
> for chat functionality.

> proceed
```

**Triggering Tier 2:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cosci run "RNA/translation-efficiency-muscle" --fold 0 --mode interactive
# Now the interactive prompts use LLM-powered responses
```

Or with OpenAI:

```bash
export OPENAI_API_KEY=sk-...
cosci run "RNA/translation-efficiency-muscle" --fold 0 --mode interactive
```

The system checks for `ANTHROPIC_API_KEY` first (prefers Claude), then `OPENAI_API_KEY`.

---

### Tier 3: Agentic (`--strategy agentic` or `--strategy hybrid`)

**What it is:** Delegates model development to an external AI coding agent (Claude Code CLI or OpenAI Codex CLI). The co-scientist becomes the "lab infrastructure" — it loads data, sets up evaluation, and validates results — while the AI agent acts as the "researcher" who writes and runs custom training code.

**When to use:** When you want potentially better models than the built-in pipelines can achieve. The agent can try deep learning, creative feature engineering, custom architectures — things our pre-built pipelines don't cover.

**Prerequisites:** One of these CLIs must be installed and authenticated:
- `claude` (Claude Code) — install from https://code.claude.com
- `codex` (OpenAI Codex) — install from https://github.com/openai/codex

Agent detection is automatic: `shutil.which("claude")` / `shutil.which("codex")`.

#### Agentic-only mode (`--strategy agentic`)

The entire model development is delegated. No built-in pipelines run.

**Workflow:**

```
Step 1-2: Same as Tier 1 (load dataset, analyze)

Step 3: SCAFFOLD AGENT WORKSPACE
   Creates a directory structure:

   agent_workspace/
     INSTRUCTIONS.md        ← Detailed prompt for the agent
     starter.py             ← Working baseline script the agent can modify
     evaluate_locally.py    ← Script to check validation performance
     data/
       train.pkl            ← Training data (or train.h5ad for AnnData)
       test_features.pkl    ← Test features WITHOUT labels (prevents leakage)
       profile.json         ← Dataset statistics
     output/                ← Where the agent saves its artifacts
       (model.joblib)       ← Agent saves trained model here
       (predictions.pkl)    ← Agent saves test predictions here

   INSTRUCTIONS.md tells the agent:
   - What the task is (regression/classification, primary metric)
   - Data format and column names
   - What output format is required
   - Where to save everything
   - That it must NOT access test labels

Step 4: INVOKE AGENT
   Calls the agent CLI as a subprocess:

   Claude Code:
     claude -p "$(cat INSTRUCTIONS.md)" \
       --allowedTools "Read,Edit,Write,Bash" \
       --output-format text

   Codex:
     codex --full-auto -C agent_workspace "$(cat INSTRUCTIONS.md)"

   The agent has full freedom to:
   - Read the starter script and data
   - Write custom training code
   - Install packages
   - Train models (PyTorch, XGBoost, anything)
   - Run evaluate_locally.py to check performance
   - Iterate on its approach
   - Save final model + predictions to output/

   Timeout: 5 min (fast), 10 min (standard), 20 min (thorough)

Step 5: VALIDATE RESULTS
   After the agent finishes (or times out):
   a. Check if output/predictions.pkl exists
   b. Load predictions with pickle
   c. Format via adapter.format_predictions() to match expected format
   d. Run task.evaluate(predictions, test_data) — the official evaluation
   e. If anything fails → report error, no fallback in agentic-only mode

Step 6: Report results
```

**Example run:**

```bash
cosci run "RNA/translation-efficiency-muscle" --fold 0 --strategy agentic --agent claude
```

**Risk:** If the agent fails or produces invalid output, there is no fallback. Use `--strategy hybrid` for safety.

#### Hybrid mode (`--strategy hybrid`)

Best of both worlds. Runs built-in pipelines FIRST, then invokes the agent with the baseline score as a target.

**Workflow:**

```
Steps 1-5: Exactly like Tier 1 (builtin)
   → Produces a baseline result, e.g. Spearman = 0.75

Step 6: SCAFFOLD + INVOKE AGENT
   Same as agentic, but INSTRUCTIONS.md also includes:
   "The built-in pipeline achieved Spearman = 0.7500.
    Your goal is to beat this score."

   This gives the agent a concrete target and motivation.

Step 7: VALIDATE + COMPARE
   If agent produces valid predictions:
     - Evaluate with task.evaluate()
     - If agent score > baseline score → use agent model
     - If agent score <= baseline score → keep baseline model

   If agent fails or produces invalid output:
     - Log warning, keep baseline model
     - The user always gets at least the baseline result

Step 8-9: Export + Report (using whichever model won)
```

**Example run:**

```bash
cosci run "expression/cell-type-classification-segerstolpe" --fold 0 --strategy hybrid
```

This will:
1. Run built-in pipelines → get baseline macro F1 ~0.85
2. Scaffold workspace with baseline_score.json saying "beat 0.85"
3. Invoke Claude Code or Codex
4. If agent achieves macro F1 > 0.85 → use agent model
5. Otherwise → keep built-in model
6. User always gets a result

---

## Budget Tiers

The `--budget` flag controls how much compute to spend.

| Parameter | `fast` | `standard` | `thorough` |
|-----------|--------|-----------|------------|
| Max models tried | 2 | 3 | 5 |
| Optuna trials per model | 10 | 30 | 60 |
| Ensemble | No | Yes (top-2) | Yes (top-2) |
| Agent timeout | 5 min | 10 min | 20 min |
| Typical RNA task time | ~1 min | ~5 min | ~15 min |
| Typical expression task time | ~7 min | ~20 min | ~60 min |

**What "max models" means concretely:**

For the RNA/translation-efficiency-muscle task with `--budget standard`:
- Experiment 1: baseline_kmer_ridge (default config, no tuning)
- Experiment 2: tuned_kmer_ridge (30 Optuna trials)
- Experiment 3: tuned_kmer_elasticnet (30 Optuna trials)
- Experiment 4: tuned_kmer_hgbr (30 Optuna trials)
- Experiment 5: ensemble (average of top-2)

With `--budget fast`:
- Experiment 1: baseline_kmer_ridge (default config)
- Experiment 2: tuned_kmer_ridge (10 Optuna trials)
- Experiment 3: tuned_kmer_elasticnet (10 Optuna trials)
- No ensemble

---

## What is Optuna

**Optuna** is a hyperparameter optimization framework. Instead of manually choosing values like `alpha=0.1` or `n_components=50`, Optuna automatically searches for the best values.

**How we use it:**

1. Each pipeline defines a `get_search_space(trial)` method that declares what hyperparameters to search and their ranges.
2. Optuna creates a "study" that will try `N` different combinations (N = optuna_trials from budget).
3. For each trial, Optuna suggests a set of hyperparameter values.
4. We train and cross-validate the model with those values.
5. Optuna observes the primary metric score and uses Bayesian optimization (TPE sampler) to decide which values to try next.
6. After all trials, we take the best-performing set of hyperparameters.

**Example:** For KmerRidgePipeline with 10 Optuna trials:

```
Trial 1:  ngram_lo=3, ngram_hi=7, alpha=0.5,   engineered=True  → Spearman=0.71
Trial 2:  ngram_lo=2, ngram_hi=6, alpha=10.0,  engineered=False → Spearman=0.65
Trial 3:  ngram_lo=3, ngram_hi=5, alpha=0.01,  engineered=True  → Spearman=0.74
Trial 4:  ngram_lo=4, ngram_hi=8, alpha=0.003, engineered=True  → Spearman=0.73
...
Trial 10: ngram_lo=3, ngram_hi=6, alpha=0.008, engineered=True  → Spearman=0.76
Best: Trial 10 config → used for final model
```

Optuna is smarter than random search — it learns from previous trials which regions of the search space are promising.

---

## Pipeline Registry

All available models, what data they support, and exactly what hyperparameters get tuned.

### Sequence Regression Pipelines (for DNA/RNA → scalar)

#### `kmer_ridge`
- **What it does:** Counts character n-grams (k-mers) in the DNA sequence using sklearn CountVectorizer, adds GC content and sequence length as engineered features, scales features, then fits a Ridge regression.
- **Default config:** `ngram_lo=3, ngram_hi=6, alpha=1.0, use_engineered=True`
- **Tuned hyperparameters (4):**
  | Parameter | Range | Type |
  |-----------|-------|------|
  | `ngram_lo` | 2-4 | int |
  | `ngram_hi` | 5-8 | int |
  | `alpha` | 0.001-1000 | float (log scale) |
  | `use_engineered` | True/False | categorical |

#### `kmer_elasticnet`
- **What it does:** Same k-mer featurization as Ridge, but uses ElasticNet which combines L1 (Lasso) and L2 (Ridge) regularization. Better at feature selection.
- **Default config:** `ngram_lo=3, ngram_hi=6, alpha=0.1, l1_ratio=0.5`
- **Tuned hyperparameters (4):**
  | Parameter | Range | Type |
  |-----------|-------|------|
  | `ngram_lo` | 2-4 | int |
  | `ngram_hi` | 5-8 | int |
  | `alpha` | 0.0001-10 | float (log scale) |
  | `l1_ratio` | 0.01-0.99 | float |

#### `kmer_hgbr`
- **What it does:** K-mer features fed into HistGradientBoostingRegressor (a fast gradient boosting tree). Can capture non-linear relationships between k-mers and the target.
- **Default config:** `ngram_lo=3, ngram_hi=6, max_iter=200, learning_rate=0.1, max_depth=6, min_samples_leaf=10, l2_regularization=0.1`
- **Tuned hyperparameters (7):**
  | Parameter | Range | Type |
  |-----------|-------|------|
  | `ngram_lo` | 2-4 | int |
  | `ngram_hi` | 5-8 | int |
  | `max_iter` | 100-500 | int |
  | `learning_rate` | 0.01-0.3 | float (log scale) |
  | `max_depth` | 3-10 | int |
  | `min_samples_leaf` | 5-50 | int |
  | `l2_regularization` | 0.0001-10 | float (log scale) |

### Expression Classification Pipelines (for AnnData → cell type)

All three share a preprocessing step: normalize total counts to 10k → log1p transform → then the pipeline does StandardScaler → PCA → classifier.

#### `expr_logreg`
- **What it does:** PCA dimensionality reduction followed by multinomial logistic regression with balanced class weights. Strong baseline for expression data.
- **Default config:** `n_components=50, C=1.0, max_iter=2000, target_sum=1e4`
- **Tuned hyperparameters (2):**
  | Parameter | Range | Type |
  |-----------|-------|------|
  | `n_components` | 20-200 | int (PCA dimensions) |
  | `C` | 0.001-100 | float (log scale, regularization) |

#### `expr_svm`
- **What it does:** PCA followed by a calibrated linear SVM (LinearSVC wrapped in CalibratedClassifierCV for probability estimates). Balanced class weights.
- **Default config:** `n_components=50, C=1.0, target_sum=1e4`
- **Tuned hyperparameters (2):**
  | Parameter | Range | Type |
  |-----------|-------|------|
  | `n_components` | 20-200 | int |
  | `C` | 0.001-100 | float (log scale) |

#### `expr_mlp`
- **What it does:** PCA followed by a 2-layer neural network (MLPClassifier). Uses early stopping to prevent overfitting.
- **Default config:** `n_components=50, hidden1=256, hidden2=128, alpha=0.001, lr=0.001, max_iter=500`
- **Tuned hyperparameters (5):**
  | Parameter | Range | Type |
  |-----------|-------|------|
  | `n_components` | 20-200 | int |
  | `hidden1` | 128/256/512 | categorical |
  | `hidden2` | 64/128/256 | categorical |
  | `alpha` | 0.00001-0.1 | float (log scale, L2 penalty) |
  | `lr` | 0.0001-0.01 | float (log scale) |

### Generic Tabular Pipelines (fallback for unknown datasets)

These activate when a DataFrame dataset has no sequence column — just numeric features.

#### `tabular_ridge` (regression)
- StandardScaler → Ridge regression
- **Tuned:** `alpha` (0.001-1000, log)

#### `tabular_hgbr` (regression)
- HistGradientBoostingRegressor
- **Tuned:** `max_iter`, `learning_rate`, `max_depth`

#### `tabular_logreg` (classification)
- StandardScaler → Logistic Regression (balanced)
- **Tuned:** `C` (0.001-100, log)

#### `tabular_hgbc` (classification)
- HistGradientBoostingClassifier (balanced)
- **Tuned:** `max_iter`, `learning_rate`, `max_depth`

#### `tabular_rf` (classification)
- Random Forest (balanced)
- **Tuned:** `n_estimators`, `max_depth`

### Ensemble

When budget allows (standard and thorough), after all individual models finish, the top-2 by CV score are combined:
- **Regression:** simple average of predictions
- **Classification:** majority vote

The ensemble is cross-validated the same way to get a fair comparison.

---

## File-by-File Reference

### Top-Level Modules

#### `cli.py`
**Purpose:** Typer-based CLI entry point — the `cosci` command.

**What it handles:**
- Defines all CLI commands: `run`, `list-datasets`, `describe`, `predict`, `report`
- Parses command-line arguments and builds a `RunConfig` dataclass
- Delegates to the `Orchestrator` for the actual work
- Uses rich markup for polished help text

**Commands defined:**
- `cosci run <dataset>` — main command, runs the full workflow
- `cosci list-datasets` — scans `genbio.datasets` and lists available tasks
- `cosci describe <dataset>` — prints dataset README and docstrings
- `cosci predict --run-dir ... --input ... --output ...` — inference with saved model
- `cosci report --run-dir ...` — regenerate report from saved artifacts

---

#### `config.py`
**Purpose:** All configuration dataclasses and enums.

**What it handles:**
- `BudgetTier` enum: fast/standard/thorough with properties for optuna_trials, max_models, do_ensemble
- `TaskType` enum: regression/classification
- `DataType` enum: dataframe_sequence/dataframe_tabular/anndata
- `DataProfile` dataclass: complete statistical profile of a dataset (shape, types, distributions, sequence stats)
- `ExperimentResult` dataclass: outcome of one experiment (name, pipeline, config, metrics, model path, duration)
- `RunConfig` dataclass: complete run specification (dataset, fold, mode, strategy, budget, seed, etc.)
  - Auto-generates a unique `run_id` with timestamp + dataset slug + random hex
  - Serializes/deserializes to YAML for reproducibility

---

#### `orchestrator.py`
**Purpose:** The "brain" — coordinates the entire workflow end-to-end.

**What it handles:**
- Strategy dispatch: routes to builtin, agentic, or hybrid workflow
- Calls analyzer → planner → runner → reporter → exporter in sequence
- Interactive mode: pauses at plan review and results review, supports LLM chat
- Suppresses noisy sklearn/optuna warnings for clean terminal output
- Saves all artifacts (config, data profile, test metrics) to the run directory
- Handles errors gracefully in agentic/hybrid mode with fallbacks

**Key methods:**
- `run()` — main entry point
- `_run_builtin()` — Tier 1 workflow
- `_run_agentic()` — Tier 3 agentic-only workflow
- `_run_hybrid_agent()` — Tier 3 hybrid agent phase
- `_interactive_plan_review()` — interactive plan approval with LLM chat
- `_interactive_results_review()` — interactive results approval with LLM chat

---

#### `analyzer.py`
**Purpose:** Dataset auto-detection and profiling.

**What it handles:**
- `discover_datasets()` — scans the filesystem for available datasets by looking for `load.py` files under `genbio/datasets/`
- `analyze_dataset(train, test)` — auto-detects the adapter (DataFrame vs AnnData) and generates a complete `DataProfile`

---

#### `planner.py`
**Purpose:** Decides which experiments to run based on the dataset profile and budget.

**What it handles:**
- Looks up the `PipelineRegistry` for pipelines matching the detected (data_type, task_type)
- Creates an `ExperimentPlan` with a baseline experiment + N tuned experiments
- Falls back to generic tabular pipelines if no specialized pipeline matches
- Generates human-readable reasoning explaining why these pipelines were chosen
- `ExperimentPlan` dataclass: ordered list of `PlannedExperiment`s + ensemble flag
- `PlannedExperiment` dataclass: name, pipeline reference, config, tune flag, trial count

---

#### `runner.py`
**Purpose:** Executes experiments, runs Optuna tuning, manages model artifacts.

**What it handles:**
- Cross-validation setup: uses dataset's `fold_id` column if available, otherwise StratifiedKFold (classification) or KFold (regression). Adapts fold count for rare classes.
- `_evaluate_cv()` — trains and evaluates a pipeline config via cross-validation, returns the mean primary metric (Spearman or macro F1)
- `_tune()` — runs Optuna hyperparameter optimization: creates a study, defines objective function that calls `_evaluate_cv()`, returns best params + score
- `_run_ensemble()` — takes top-2 models, cross-validates their averaged/voted predictions
- `execute()` — runs all planned experiments sequentially, saves model.joblib + config.json + metrics.json for each
- `select_best()` — returns the experiment with highest primary metric
- `generate_test_predictions()` — loads the best model and predicts on test data, handles ensembles

---

#### `reporter.py`
**Purpose:** Generates a Markdown report with figures.

**What it handles:**
- `ReportBuilder` class with sections: executive summary, dataset profile, methodology, results, analysis, reproducibility, recommendations
- Can reload profile and metrics from saved JSON files (for `cosci report` command)
- Generates matplotlib/seaborn figures:
  - `label_distribution.png` — class distribution bar chart or label statistics
  - `validation_metrics.png` — model comparison bar chart (best model highlighted in green)
- Modality-specific recommendations (different suggestions for sequence vs expression vs tabular)

---

#### `exporter.py`
**Purpose:** Generates a portable, standalone project from a completed run.

**What it handles:**
- `ProjectExporter` class: creates `exported_project/` directory with:
  - `train.py` — fully self-contained training script that can retrain the model from scratch (data-type-specific: sequence, anndata, or tabular templates)
  - `predict.py` — inference script that loads model.joblib and predicts on new data (CSV or h5ad)
  - `requirements.txt` — minimal dependencies for the exported project
  - `README.md` — quickstart instructions
  - `artifacts/model.joblib` — the trained model
  - `artifacts/pipeline_config.json` — hyperparameter config
- `run_inference()` — standalone function used by `cosci predict` to run a saved model on new data
- Contains string template constants for each data type's train.py and predict.py

---

#### `console.py`
**Purpose:** Rich terminal formatting helpers.

**What it handles:**
- `header()` — double-bordered panel for section headers
- `section()` — yellow section titles with horizontal rule
- `info()`, `success()`, `warning()`, `error()` — colored status messages
- `step()` — numbered step indicators
- `metrics_table()` — formatted table of metric name/value pairs with star marker for primary metric
- `experiments_table()` — formatted table of experiment results (name, pipeline, score, time)
- `data_profile_panel()` — rounded panel showing dataset characteristics
- `spinner_progress()` — progress bar with spinner for long operations

---

### adapters/

#### `adapters/base.py`
**Purpose:** Abstract base class for dataset adapters + auto-detection factory.

**What it handles:**
- `DatasetAdapter` ABC with abstract methods: `get_features()`, `get_labels()`, `format_predictions()`, `profile()`, `save_train_data()`, `save_test_features()`
- `get_adapter(train, test)` — factory function that checks `isinstance(train, AnnData)` then `isinstance(train, DataFrame)` and returns the appropriate adapter. Raises `TypeError` for unknown types.

#### `adapters/dataframe_adapter.py`
**Purpose:** Handles pandas DataFrame datasets (both sequence and tabular).

**What it handles:**
- Auto-detects the label column by checking for common names: "labels", "label", "target", "y"
- Auto-detects sequence column by checking for: "sequence", "sequences", "seq", "dna", "rna", "protein"
- Infers task type: float labels with >20 unique values → regression, otherwise → classification
- `get_features()` — returns the sequence column as a Series (for sequence data) or numeric columns as a numpy array (for tabular data)
- `get_labels()` — extracts the label column as numpy array
- `format_predictions()` — copies the DataFrame and replaces the label column with predictions
- `profile()` — computes DataProfile with label stats, class distribution, sequence length statistics
- `save_train_data()` / `save_test_features()` — pickle serialization for agent workspace

#### `adapters/anndata_adapter.py`
**Purpose:** Handles AnnData single-cell expression datasets.

**What it handles:**
- Auto-detects label column in `.obs` by checking: "cell_type_label", "cell_type", "label", etc.
- `get_features()` — converts `.X` to dense numpy array (handles scipy sparse matrices)
- `get_labels()` — extracts from `.obs[label_col]`
- `format_predictions()` — copies AnnData and sets `obs[label_col]` to predictions
- `profile()` — computes DataProfile with class distribution, gene counts
- `save_train_data()` / `save_test_features()` — h5ad serialization for agent workspace

---

### pipelines/

#### `pipelines/base.py`
**Purpose:** Abstract pipeline interface and global registry.

**What it handles:**
- `PipelineBase` ABC: defines the contract every pipeline must implement — `train()`, `predict()`, `get_search_space()`, `get_default_config()`, plus `save_model()` / `load_model()` using joblib
- `PipelineRegistry` class: a global list of registered pipelines. Pipelines self-register at import time. `get_candidates(data_type, task_type)` returns all matching pipelines.

#### `pipelines/kmer_regression.py`
**Purpose:** Three pipelines for DNA/RNA sequence regression tasks.

**What it handles:**
- `DensifyTransformer` — converts sparse matrices to dense (needed for HistGradientBoosting which doesn't accept sparse input)
- `_gc_content()` — computes GC content (fraction of G+C bases) as an engineered feature
- `_seq_length()` — computes sequence length as an engineered feature
- `KmerRidgePipeline` — CountVectorizer(char n-grams) + optional GC/length + StandardScaler + Ridge
- `KmerElasticNetPipeline` — same featurization + ElasticNet (L1+L2 regularization)
- `KmerGBRPipeline` — same featurization + DensifyTransformer + HistGradientBoostingRegressor
- All three register themselves with `PipelineRegistry` at module import time

#### `pipelines/expression_classification.py`
**Purpose:** Three pipelines for single-cell expression classification.

**What it handles:**
- `_scanpy_preprocess()` — mimics scanpy's normalize_total + log1p without requiring scanpy at inference time (uses sklearn normalize with L1 norm × 10,000)
- `ExprLogRegPipeline` — preprocess → StandardScaler → PCA → LogisticRegression (balanced, saga solver)
- `ExprSVMPipeline` — preprocess → StandardScaler → PCA → CalibratedClassifierCV(LinearSVC, balanced). Adapts CV folds for rare classes.
- `ExprMLPPipeline` — preprocess → StandardScaler → PCA → MLPClassifier (2 hidden layers, early stopping)
- Preprocessing config is stored on the fitted pipeline object (`_cosci_preprocess_config`) so predict() can apply the same transform

#### `pipelines/generic_tabular.py`
**Purpose:** Five fallback pipelines for unknown tabular datasets.

**What it handles:**
- `TabularRidgePipeline` — StandardScaler + Ridge (regression)
- `TabularGBRPipeline` — HistGradientBoostingRegressor (regression)
- `TabularLogRegPipeline` — StandardScaler + LogisticRegression balanced (classification)
- `TabularGBCPipeline` — HistGradientBoostingClassifier balanced (classification)
- `TabularRFPipeline` — RandomForestClassifier balanced (classification)
- These activate for any DataFrame without a sequence column, serving as a robust fallback for hidden/unknown datasets

---

### agents/

#### `agents/invoker.py`
**Purpose:** Detects, launches, and monitors AI coding agents.

**What it handles:**
- `detect_agents()` — checks PATH for `claude` and `codex` binaries
- `select_agent()` — picks the best available agent (prefers Claude Code)
- `AgentInvoker` class:
  - `run()` — full workflow: scaffold workspace → invoke agent → validate results
  - `_invoke_agent()` — builds the subprocess command for claude or codex, runs it with timeout, captures output to log file
  - `_validate_results()` — loads `output/predictions.pkl`, formats via adapter, returns dict with predictions
  - `_get_timeout()` — returns timeout in seconds based on budget tier (300/600/1200)

#### `agents/workspace.py`
**Purpose:** Creates the self-contained workspace an agent needs.

**What it handles:**
- `WorkspaceScaffolder` class:
  - `scaffold()` — creates directory structure, saves data, writes all scaffold files
  - `_write_instructions()` — generates INSTRUCTIONS.md with task description, data format, metric, constraints, baseline score (if hybrid)
  - `_write_starter()` — generates a working starter.py that loads data, trains a simple baseline, and saves predictions. Data-type-specific (AnnData, sequence, tabular).
  - `_write_evaluate_locally()` — generates evaluate_locally.py that loads the agent's model and runs cross-validation on training data so the agent can check its own performance

#### `agents/validator.py`
**Purpose:** Validates agent output against the official evaluation.

**What it handles:**
- `validate_agent_output()` — loads predictions.pkl, formats via adapter, calls task.evaluate(), returns metrics dict. Catches and reports all errors gracefully.

---

### llm/

#### `llm/client.py`
**Purpose:** Unified LLM API client with provider auto-detection.

**What it handles:**
- `get_llm_client()` — checks environment for API keys, returns appropriate client or None
- `LLMClient` base class with `chat(system, user, max_tokens)` method
- `AnthropicClient` — wraps the Anthropic Python SDK, uses Claude Sonnet
- `OpenAIClient` — wraps the OpenAI Python SDK, uses GPT-4o
- Graceful degradation: if no API key or SDK not installed, returns None

#### `llm/advisor.py`
**Purpose:** The "language-based" interface — translates structured data into natural language.

**What it handles:**
- `Advisor` class with methods:
  - `interpret_profile()` — describes dataset characteristics in plain English
  - `suggest_approach()` — recommends modeling strategies with rationale
  - `interpret_results()` — explains why certain models performed better/worse
  - `polish_report()` — rewrites the Markdown report with better prose
  - `chat()` — handles free-form questions in interactive mode
- Every method has a template fallback that works without an LLM
- `has_llm` property for checking if LLM is available

#### `llm/prompts.py`
**Purpose:** Prompt templates for LLM interactions.

**What it handles:**
- `SYSTEM_ADVISOR` — system prompt positioning the LLM as a bioinformatics ML advisor
- `SYSTEM_REPORT_WRITER` — system prompt for scientific report editing
- `INTERPRET_PROFILE` — user prompt template for dataset analysis
- `SUGGEST_APPROACH` — user prompt template for experiment planning
- `INTERPRET_RESULTS` — user prompt template for results interpretation

---

### templates/

#### `templates/.gitkeep`
Placeholder to ensure the templates directory is tracked by git. Reserved for future Jinja2 templates if the report/export system evolves beyond the current string-based templates.
