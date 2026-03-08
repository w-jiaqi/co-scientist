# GenBio Co-Scientist

A language-based ML co-scientist CLI that automates building machine learning models for GenBio leaderboard datasets. It analyzes datasets, trains and tunes models, generates reports, and exports portable codebases -- supporting both interactive collaboration and fully autonomous operation.

## Quickstart

### Installation

```bash
git clone <this-repo>
cd co-scientist
pip install -e .
```

For LLM-powered features (optional):

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=sk-...  # or OPENAI_API_KEY
```

### Run the Co-Scientist

```bash
# Fully autonomous (no interaction required)
cosci run "RNA/translation-efficiency-muscle" --fold 0 --budget standard

# Interactive mode (asks for input at decision points)
cosci run "expression/cell-type-classification-segerstolpe" --fold 0 --mode interactive

# Hybrid strategy: built-in pipelines + AI coding agent (requires claude or codex CLI)
cosci run "RNA/translation-efficiency-muscle" --fold 0 --strategy hybrid
```

### Other Commands

```bash
# List available datasets
cosci list-datasets

# Describe a dataset
cosci describe "RNA/translation-efficiency-muscle"

# Run inference with a trained model
cosci predict --run-dir runs/<run_id> --input data.csv --output predictions.csv

# Regenerate a report
cosci report --run-dir runs/<run_id>
```

## Operating Modes

### Autonomous (`--mode auto`, default)
Runs end-to-end without user input. Selects models, tunes hyperparameters, evaluates, and generates a full report and portable codebase automatically.

### Interactive (`--mode interactive`)
Pauses at key decision points:
1. After planning -- review and adjust the experiment plan
2. After training -- review results, ask questions, decide whether to continue
3. With an LLM API key, you can ask free-form questions ("why did SVM perform poorly?")

## Strategy Tiers

### Built-in (`--strategy builtin`, default)
Uses pre-built, tested scikit-learn pipelines. No API keys or external tools needed. Always produces reliable results.

### Agentic (`--strategy agentic`)
Delegates model development to an AI coding agent (Claude Code or Codex). The co-scientist scaffolds a workspace with data, instructions, and evaluation tools, then lets the agent write and iterate on custom training code.

### Hybrid (`--strategy hybrid`)
Best of both worlds: runs built-in pipelines first to establish a baseline, then invokes an AI coding agent to try to beat it. You always get at least the baseline result.

## Budget Tiers

| Budget | Models | Optuna Trials | Ensemble | Use Case |
|--------|--------|---------------|----------|----------|
| `fast` | 2 | 10/model | No | Quick exploration |
| `standard` | 3 | 30/model | Yes | Default, good results |
| `thorough` | 5 | 60/model | Yes | Maximum performance |

## Output Structure

Each run produces a self-contained directory:

```
runs/<run_id>/
    config.yaml              # Full configuration for reproducibility
    data_profile.json        # Dataset statistics
    test_metrics.json        # Final test evaluation
    report.md                # Full report with analysis
    figures/                 # Visualizations
    experiments/             # Individual experiment artifacts
    exported_project/        # Portable codebase
        README.md
        train.py             # Re-train from scratch
        predict.py           # Run inference on new data
        requirements.txt     # Pinned dependencies
        artifacts/
            model.joblib     # Trained model
```

## Architecture

The co-scientist is built as a 3-tier system:

- **Tier 1 -- Built-in pipelines**: Deterministic sklearn pipelines (k-mer regression, expression classification, generic tabular) with Optuna hyperparameter tuning.
- **Tier 2 -- LLM advisor** (optional): Natural language explanations, experiment suggestions, and report polishing via Claude or GPT-4.
- **Tier 3 -- Agentic delegation** (optional): Full AI coding agent (Claude Code / Codex CLI) integration for creative model development.

## Supported Data Types

| Data Type | Format | Example Task |
|-----------|--------|-------------|
| DNA/RNA sequences | `pd.DataFrame` with sequence column | Translation efficiency |
| Single-cell expression | `sc.AnnData` | Cell type classification |
| Generic tabular | `pd.DataFrame` with numeric features | Any tabular task |

---

## GenBio Leaderboard API

### Python SDK

```python
import genbio.leaderboard as gl

task = gl.BenchmarkTask(name='RNA/translation-efficiency-muscle', fold='0', user='your_name')
task.describe()
train, test = task.setup()
task.evaluate(preds, targets)
task.submit(preds, name='my_model', description='Description')
```

### Leaderboard CLI

```bash
genbio-leaderboard leaderboard --dataset RNA/translation-efficiency-muscle --fold 0
genbio-leaderboard history --dataset RNA/translation-efficiency-muscle --fold 0 --user your_name
```

## Adding New Datasets

Add new datasets under `src/genbio/datasets/`. Each entry should include:
- `README.md`: Description of the dataset
- `load.py`: `load(fold)` returns `{"train": ..., "test": ...}`
- `evaluate.py`: `evaluate(preds, targets)` returns metrics dict with `primary_metric`
- `__init__.py` (can be empty)
