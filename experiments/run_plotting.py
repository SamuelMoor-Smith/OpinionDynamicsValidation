import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.plotting.plotting import produce_figure, produce_stripplot


def get_models_from_results_file(filepath: Path) -> tuple[str, str]:
    with filepath.open("r") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            row = json.loads(line)
            return row["true_model"], row["prediction_model"]
    raise ValueError(f"No JSON rows found in {filepath}")


def rerun_optimized_figures(results_dir: Path) -> None:
    jsonl_files = sorted(
        filepath
        for filepath in results_dir.rglob("*.jsonl")
        if not filepath.name.endswith("_raw.jsonl")
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under {results_dir}")

    print(f"Found {len(jsonl_files)} optimized result files in {results_dir}")

    for filepath in jsonl_files:
        generator, predictor = get_models_from_results_file(filepath)
        print(
            "Regenerating optimized figure for "
            f"generator={generator}, predictor={predictor} "
            f"from {filepath}"
        )
        produce_figure(
            generator=generator,
            predictor=predictor,
            filepath=str(filepath),
            experiment="optimized",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimized-dir",
        type=Path,
        default=Path("results/optimized"),
        help="Directory containing optimized .jsonl result files.",
    )
    parser.add_argument(
        "--stripplot",
        action="store_true",
        help="Produce the real-data stripplot instead of optimized figures.",
    )
    parser.add_argument(
        "--distorted",
        action="store_true",
        help="Use distorted real-data files when producing the stripplot.",
    )
    args = parser.parse_args()

    if args.stripplot:
        produce_stripplot(distorted=args.distorted)
    else:
        rerun_optimized_figures(args.optimized_dir)
