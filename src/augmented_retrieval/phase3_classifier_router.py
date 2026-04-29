import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 classifier router entrypoint (wrapper over phase2_router classifier mode)."
    )
    parser.add_argument("--data_file", type=str, default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--generation_root", type=str, default="generation_logs/augmented")
    parser.add_argument("--output_dir", type=str, default="results/phase3")
    parser.add_argument(
        "--cache_file",
        type=str,
        default="results/phase3/classification_cache_v2.json",
        help="Classifier cache file for Phase 3 runs.",
    )
    parser.add_argument("--classifier_model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_organization", type=str, default=None)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep_seconds", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "src.augmented_retrieval.phase2_router",
        "--mode",
        "classifier",
        "--data_file",
        args.data_file,
        "--generation_root",
        args.generation_root,
        "--output_dir",
        args.output_dir,
        "--cache_file",
        args.cache_file,
        "--classifier_model",
        args.classifier_model,
        "--max_retries",
        str(args.max_retries),
        "--retry_sleep_seconds",
        str(args.retry_sleep_seconds),
    ]
    if args.openai_api_key:
        cmd.extend(["--openai_api_key", args.openai_api_key])
    if args.openai_organization:
        cmd.extend(["--openai_organization", args.openai_organization])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
