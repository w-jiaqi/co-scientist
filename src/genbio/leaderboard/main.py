from genbio.datasets.utils import _get_datasets_dir, _load_dataset_module
from genbio.leaderboard import reporting
from pathlib import Path

__all__ = ['BenchmarkTask', 'describe']


class BenchmarkTask:
    def __init__(self, name: str, fold: str, user: str):
        self.name = name
        self.fold = fold
        self.user = user

    def setup(self):
        """Load the train and test datasets for the current dataset and fold."""
        module = _load_dataset_module(self.name, 'load')
        data = module.load(self.fold)
        self._test_data = data['test']
        return data['train'], data['test']

    def evaluate(self, preds, targets):
        if not hasattr(self, '_test_data'):
            raise ValueError("Must call setup() before evaluate()")

        module = _load_dataset_module(self.name, 'evaluate')
        results = module.evaluate(preds, targets)

        print("\nEvaluation Results:")
        print(f"Primary metric: {results['primary_metric']}")
        for key, value in results.items():
            if key != 'primary_metric':
                print(f"  {key}: {value:.6f}")
        return results

    def submit(self, preds, name=None, description=None) -> None:
        """Calls evaluate(preds, _test_data) and submits the results.

        Note:
            If preds is identical to _test_data, it will be logged as a dummy submission.

        Args:
            preds (pd.DataFrame): DataFrame containing your predictions in the format required by evaluate().
            name (str): Name for the submission.
            description (str): Description for the submission.

        Returns:
            None.

        """
        if not hasattr(self, '_test_data'):
            raise ValueError("Must call setup() before submit()")
        if name is None:
            name = "No name provided"
        if description is None:
            description = "No description provided"
        module = _load_dataset_module(self.name, 'evaluate')
        results = module.evaluate(preds, self._test_data)

        if preds is self._test_data:
            print("> Logging as dummy submission (submission data matches test data)")
        else:
            # Save submission to disk
            filepath = reporting.save_submission(
                dataset=self.name,
                fold=self.fold,
                user=self.user,
                name=name,
                description=description,
                metrics=results,
            )
            print(f"\nSubmission saved to: {filepath}")

        # Print results
        print("\nEvaluation Results:")
        print(f"Primary metric: {results['primary_metric']}")
        for key, value in results.items():
            if key != 'primary_metric':
                print(f"  {key}: {value:.6f}")

    def describe(self) -> None:
        """
        Print dataset README and documentation for load, evaluate, and submit
        functions.
        """
        describe(self)


def describe(task: BenchmarkTask) -> None:
    """Print dataset README and documentation for load, evaluate, and submit functions."""
    # Get the README file from the datasets directory
    datasets_dir = _get_datasets_dir()
    dataset_module_name = task.name.replace('-', '_')
    readme_path = datasets_dir / dataset_module_name / 'README.md'

    # Print README
    try:
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        print(readme_content)
    except FileNotFoundError:
        print(f"No README.md found for dataset: {task.name}")

    # Print load() documentation
    load_module = _load_dataset_module(task.name, 'load')
    if hasattr(load_module, 'load') and load_module.load.__doc__:
        print(f"\n{'='*80}")
        print(f"load() function documentation for {task.name}:")
        print("=" * 80)
        print(load_module.load.__doc__)
    else:
        print(f"\nNo documentation found for load() function in dataset: {task.name}")

    # Print evaluate() documentation
    evaluate_module = _load_dataset_module(task.name, 'evaluate')
    if hasattr(evaluate_module, 'evaluate') and evaluate_module.evaluate.__doc__:
        print(f"\n{'='*80}")
        print(f"evaluate(preds, targets) function documentation for {task.name}:")
        print("=" * 80)
        print(evaluate_module.evaluate.__doc__)
    else:
        print(f"\nNo documentation found for evaluate() function in dataset: {task.name}")

    # Print submit() documentation
    print(f"\n{'='*80}")
    print("submit(preds) function documentation:")
    print("=" * 80)
    if task.submit.__doc__:
        print(task.submit.__doc__)
    else:
        print("No documentation found for submit() function")
