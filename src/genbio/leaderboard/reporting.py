"""CLI and dashboard reporting tools for the leaderboard."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


# Submission directory path
SUBMISSION_DIR = Path(__file__).parent.parent.parent.parent / "submissions" 


# Storage functions

def save_submission(
    dataset: str,
    fold: str,
    user: str,
    metrics: Dict[str, float],
    name: str,
    description: str,
) -> str:
    """
    Save a submission to disk.

    Args:
        dataset: Dataset name
        fold: Fold identifier
        user: User identifier
        metrics: Dictionary of metric names and values
        name: Submission name
        description: Submission description

    Returns:
        Path to the saved submission file
    """
    # Create directory structure
    submission_dir = SUBMISSION_DIR / "genbio-leaderboard/submissions" / dataset / fold / user
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().isoformat()

    # Create submission data
    submission_data = {
        "timestamp": timestamp,
        "user": user,
        "dataset": dataset,
        "fold": fold,
        "metrics": metrics,
        "name": name,
        "description": description,
    }

    # Save to file
    filename = f"{timestamp.replace(':', '-')}.json"
    filepath = submission_dir / filename

    with open(filepath, 'w') as f:
        json.dump(submission_data, f, indent=2)

    return str(filepath)


def load_submissions(
    dataset: str,
    fold: str,
    user: Optional[str] = None,
) -> List[Dict]:
    """
    Load submissions from disk.

    Args:
        dataset: Dataset name
        fold: Fold identifier
        user: Optional user identifier. If None, load all users.

    Returns:
        List of submission dictionaries
    """
    submissions = []

    base_dir = Path(SUBMISSION_DIR) / "genbio-leaderboard/submissions" / dataset / fold

    if not base_dir.exists():
        return submissions

    if user:
        # Load submissions for specific user
        user_dir = base_dir / user
        if user_dir.exists():
            for filepath in user_dir.glob("*.json"):
                with open(filepath, 'r') as f:
                    submissions.append(json.load(f))
    else:
        # Load submissions for all users
        for user_dir in base_dir.iterdir():
            if user_dir.is_dir():
                for filepath in user_dir.glob("*.json"):
                    with open(filepath, 'r') as f:
                        submissions.append(json.load(f))

    # Sort by timestamp
    submissions.sort(key=lambda x: x['timestamp'])

    return submissions


def get_leaderboard_data(dataset: str, fold: str) -> List[Dict]:
    """
    Get leaderboard data (best submission per user).

    Args:
        dataset: Dataset name
        fold: Fold identifier

    Returns:
        List of best submissions per user, sorted by primary metric
    """
    all_submissions = load_submissions(dataset, fold)

    if not all_submissions:
        return []

    # Get primary metric name
    primary_metric = all_submissions[0]['metrics']['primary_metric']

    # Group by user and get best submission
    user_best = {}
    for submission in all_submissions:
        user = submission['user']
        metric_value = submission['metrics'][primary_metric]

        if user not in user_best or metric_value > user_best[user]['metrics'][primary_metric]:
            user_best[user] = submission

    # Convert to list and sort by primary metric (descending)
    leaderboard = list(user_best.values())
    leaderboard.sort(key=lambda x: x['metrics'][primary_metric], reverse=True)

    return leaderboard


def get_user_history(dataset: str, fold: str, user: str) -> List[Dict]:
    """
    Get submission history for a specific user.

    Args:
        dataset: Dataset name
        fold: Fold identifier
        user: User identifier

    Returns:
        List of submissions sorted by timestamp
    """
    return load_submissions(dataset, fold, user)


# Display functions

def display_leaderboard(dataset: str, fold: str):
    """Display the leaderboard for the specified dataset and fold."""
    # Get leaderboard data
    dataset = dataset.replace('-', '_')
    leaderboard_data = get_leaderboard_data(dataset, fold)

    if not leaderboard_data:
        print(f"\nNo submissions found for {dataset} fold {fold}")
        return

    # Get primary metric name
    primary_metric = leaderboard_data[0]['metrics']['primary_metric']

    # Print header
    print(f"\n{'='*100}")
    print(f"Leaderboard: {dataset} (Fold {fold})")
    print(f"Primary Metric: {primary_metric}")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'User':<20} {'Name':<25} {'Score':<12} {'Timestamp':<30}")
    print(f"{'-'*100}")

    # Print entries
    for rank, entry in enumerate(leaderboard_data, 1):
        user = entry['user']
        name = entry.get('name', 'Unnamed')  # Handle old submissions without name
        score = entry['metrics'][primary_metric]
        timestamp = entry['timestamp'][:19]  # Remove microseconds

        print(f"{rank:<6} {user:<20} {name:<25} {score:<12.6f} {timestamp:<30}")

    print(f"{'='*100}\n")


def display_history(dataset: str, fold: str, user: str):
    """Display submission history for a user."""
    # Get user history
    user_submissions = get_user_history(dataset, fold, user)

    if not user_submissions:
        print(f"\nNo submissions found for user '{user}' on {dataset} fold {fold}")
        return

    # Get primary metric name
    primary_metric = user_submissions[0]['metrics']['primary_metric']

    # Print header
    print(f"\n{'='*120}")
    print(f"Submission History: {user} - {dataset} (Fold {fold})")
    print(f"Primary Metric: {primary_metric}")
    print(f"{'='*120}")
    print(f"{'#':<4} {'Name':<25} {'Timestamp':<22} {primary_metric:<12} {'Change':<10} {'Other Metrics'}")
    print(f"{'-'*120}")

    # Print entries
    prev_score = None
    for idx, submission in enumerate(user_submissions, 1):
        name = submission.get('name', 'Unnamed')  # Handle old submissions without name
        timestamp = submission['timestamp'][:19]  # Remove microseconds
        score = submission['metrics'][primary_metric]

        # Calculate change
        if prev_score is not None:
            change = score - prev_score
            change_str = f"{change:+.4f}" if change != 0 else "  --"
        else:
            change_str = "  --"

        # Format other metrics
        other_metrics = []
        for key, value in submission['metrics'].items():
            if key not in ['primary_metric', primary_metric]:
                other_metrics.append(f"{key}={value:.4f}")
        other_metrics_str = ", ".join(other_metrics)

        print(f"{idx:<4} {name:<25} {timestamp:<22} {score:<12.6f} {change_str:<10} {other_metrics_str}")

        prev_score = score

    print(f"{'='*120}")

    # Show improvement summary
    if len(user_submissions) > 1:
        first_score = user_submissions[0]['metrics'][primary_metric]
        best_score = max(s['metrics'][primary_metric] for s in user_submissions)
        latest_score = user_submissions[-1]['metrics'][primary_metric]

        print(f"\nSummary:")
        print(f"  First submission:  {first_score:.6f}")
        print(f"  Best submission:   {best_score:.6f}")
        print(f"  Latest submission: {latest_score:.6f}")
        print(f"  Total improvement: {latest_score - first_score:+.6f}")
        print()


# Export functions

def export_benchmark_data(
    output_file: str = "benchmark_export.csv",
) -> str:
    """
    Export all benchmark data to CSV.

    Args:
        output_file: Path to output CSV file (default: "benchmark_export.csv")

    Returns:
        Path to the exported CSV file
    """
    base_dir = SUBMISSION_DIR / "genbio-leaderboard/submissions"

    if not base_dir.exists():
        raise ValueError(f"Submissions directory not found: {base_dir}")

    # Find all submission JSON files recursively
    all_submissions = []
    for submission_file in base_dir.glob("**/*.json"):
        with open(submission_file, 'r') as f:
            submission = json.load(f)
            # Flatten metrics into top-level fields
            flat_submission = {
                'dataset': submission['dataset'],
                'fold': submission['fold'],
                'user': submission['user'],
                'timestamp': submission['timestamp'],
                'name': submission.get('name', ''),
                'description': submission.get('description', ''),
            }
            # Add all metrics as separate columns
            for key, value in submission['metrics'].items():
                flat_submission[f'metric_{key}'] = value
            all_submissions.append(flat_submission)

    if not all_submissions:
        raise ValueError("No submissions found to export")

    # Create DataFrame and sort by timestamp
    df = pd.DataFrame(all_submissions)
    df = df.sort_values('timestamp')
    df.to_csv(output_file, index=False)
    return output_file


# CLI entry point

def cli():
    """Main CLI entry point for genbio-leaderboard."""
    parser = argparse.ArgumentParser(
        description='GenBio Leaderboard CLI',
        prog='genbio-leaderboard'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Leaderboard command
    leaderboard_parser = subparsers.add_parser('leaderboard', help='Display leaderboard')
    leaderboard_parser.add_argument('--dataset', required=True, help='Dataset name')
    leaderboard_parser.add_argument('--fold', required=True, help='Fold identifier')
    # History command
    history_parser = subparsers.add_parser('history', help='Display submission history')
    history_parser.add_argument('--dataset', required=True, help='Dataset name')
    history_parser.add_argument('--fold', required=True, help='Fold identifier')
    history_parser.add_argument('--user', required=True, help='User identifier')
    # Export command
    export_parser = subparsers.add_parser('export', help='Export all benchmark data to CSV')
    export_parser.add_argument(
        '-o', '--output',
        default='benchmark_export.csv',
        help='Output CSV filename (default: benchmark_export.csv)'
    )

    args = parser.parse_args()
    if args.command == 'leaderboard':
        display_leaderboard(args.dataset, args.fold)
    elif args.command == 'history':
        display_history(args.dataset, args.fold, args.user)
    elif args.command == 'export':
        output_file = export_benchmark_data(
            output_file=args.output,
        )
        print(f"\nBenchmark data successfully exported to: {output_file}")
    else:
        parser.print_help()
