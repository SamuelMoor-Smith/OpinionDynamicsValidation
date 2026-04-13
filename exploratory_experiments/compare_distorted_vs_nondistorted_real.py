import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.plotting.plotting_utils import calculate_explained_variance_real


DEFAULT_DISTORTED = Path("results/real/distorted_gestefeld_lorenz_3.jsonl")
DEFAULT_BASELINE = Path("results/real/gestefeld_lorenz_3.jsonl")
DEFAULT_OUTPUT = Path("exploratory_experiments/results/distorted_vs_nondistorted_gestefeld_lorenz_pairwise.csv")


def sample_rows(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    replace = len(df) < sample_size
    return df.sample(n=sample_size, replace=replace, random_state=seed).reset_index(drop=True)


def compare_for_dataset(
    distorted_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ess_key: str,
    sample_size: int,
    seed: int,
) -> dict:
    distorted_dataset = distorted_df[distorted_df["ess_key"] == ess_key].copy()
    baseline_dataset = baseline_df[baseline_df["ess_key"] == ess_key].copy()

    distorted_sample = sample_rows(distorted_dataset, sample_size=sample_size, seed=seed)
    baseline_sample = sample_rows(baseline_dataset, sample_size=sample_size, seed=seed + 1)

    distorted_better_loss = distorted_sample["loss"].to_numpy() < baseline_sample["loss"].to_numpy()
    distorted_better_ev = (
        distorted_sample["explained_variance"].to_numpy()
        > baseline_sample["explained_variance"].to_numpy()
    )

    ties_loss = distorted_sample["loss"].to_numpy() == baseline_sample["loss"].to_numpy()
    ties_ev = (
        distorted_sample["explained_variance"].to_numpy()
        == baseline_sample["explained_variance"].to_numpy()
    )

    country = distorted_dataset["ess_country"].iloc[0] if "ess_country" in distorted_dataset else ""

    return {
        "ess_key": ess_key,
        "ess_country": country,
        # "distorted_n_rows": len(distorted_dataset),
        # "baseline_n_rows": len(baseline_dataset),
        "sample_size": sample_size,
        # "distorted_better_loss_pct": 100 * distorted_better_loss.mean(),
        # "ties_loss_pct": 100 * ties_loss.mean(),
        "distorted_better_%": 100 * distorted_better_ev.mean(),
        # "ties_explained_variance_pct": 100 * ties_ev.mean(),
        # "distorted_sample_mean_loss": distorted_sample["loss"].mean(),
        # "baseline_sample_mean_loss": baseline_sample["loss"].mean(),
        "distorted_sample_mean_explained_variance": distorted_sample["explained_variance"].mean(),
        "baseline_sample_mean_explained_variance": baseline_sample["explained_variance"].mean(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distorted-file", type=Path, default=DEFAULT_DISTORTED)
    parser.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    distorted_df = pd.read_json(args.distorted_file, lines=True)
    baseline_df = pd.read_json(args.baseline_file, lines=True)

    distorted_df = calculate_explained_variance_real(distorted_df)
    baseline_df = calculate_explained_variance_real(baseline_df)

    shared_datasets = sorted(set(distorted_df["ess_key"]).intersection(baseline_df["ess_key"]))

    results = []
    for index, ess_key in enumerate(shared_datasets):
        results.append(
            compare_for_dataset(
                distorted_df=distorted_df,
                baseline_df=baseline_df,
                ess_key=ess_key,
                sample_size=args.sample_size,
                seed=args.seed + index * 1000,
            )
        )

    results_df = pd.DataFrame(results).sort_values(
        by="distorted_better_%",
        ascending=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print(f"Saved pairwise comparison results to {args.output}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
