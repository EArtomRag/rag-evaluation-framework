"""Utility for comparing two evaluation runs."""

from pathlib import Path
import pandas as pd


def load_metrics(run_dir: Path) -> pd.DataFrame:
    """Loads the metrics.csv file from a run directory."""
    metrics_file = run_dir / "metrics.csv"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found in '{run_dir}'")
    return pd.read_csv(metrics_file).set_index("metric")


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> pd.DataFrame:
    """
    Compares the aggregated metrics from two different evaluation runs.

    Args:
        baseline_dir: Path to the directory of the baseline run.
        candidate_dir: Path to the directory of the candidate run.

    Returns:
        A pandas DataFrame detailing the comparison.
    """
    try:
        baseline_metrics = load_metrics(baseline_dir)
        candidate_metrics = load_metrics(candidate_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame()

    # Rename columns for clarity before merging
    baseline_metrics.rename(columns={"score": "baseline"}, inplace=True)
    candidate_metrics.rename(columns={"score": "candidate"}, inplace=True)

    # Merge the two dataframes
    comparison_df = pd.merge(
        baseline_metrics, candidate_metrics, left_index=True, right_index=True, how="outer"
    )
    comparison_df.fillna(0, inplace=True)

    # Calculate delta and percentage change
    comparison_df["delta"] = comparison_df["candidate"] - comparison_df["baseline"]
    
    # Calculate percentage change, handling division by zero
    comparison_df["delta_pct"] = (
        (comparison_df["delta"] / comparison_df["baseline"].abs()) * 100
    ).where(comparison_df["baseline"] != 0, float('inf'))


    return comparison_df

