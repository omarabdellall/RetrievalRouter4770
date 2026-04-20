import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


DEFAULT_CONFIGS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--retrieval_root", type=str, default="retrieval_logs/augmented")
    parser.add_argument("--generation_root", type=str, default="generation_logs/augmented")
    parser.add_argument("--output_dir", type=str, default="results/phase1")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_reference(data_file: str) -> Dict[str, dict]:
    entries = json.load(open(data_file, "r"))
    return {entry["question_id"]: entry for entry in entries}


def normalize_qtype(entry: dict) -> str:
    if entry["question_id"].endswith("_abs"):
        return "abstention"
    return entry["question_type"]


def latest_eval_file(generation_dir: Path) -> Optional[Path]:
    files = sorted(generation_dir.glob("*.eval-results-gpt-4o"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def qa_accuracy_by_type(eval_entries: List[dict], qid_to_ref: Dict[str, dict]) -> Dict[str, float]:
    by_type: Dict[str, List[float]] = {}
    all_vals = []
    for row in eval_entries:
        qid = row["question_id"]
        ref_entry = qid_to_ref.get(qid)
        if ref_entry is None:
            continue
        qtype = normalize_qtype(ref_entry)
        label = float(bool(row["autoeval_label"]["label"]))
        by_type.setdefault(qtype, []).append(label)
        all_vals.append(label)

    out = {"overall": float(np.mean(all_vals)) if all_vals else 0.0}
    for qtype in sorted(by_type.keys()):
        out[qtype] = float(np.mean(by_type[qtype])) if by_type[qtype] else 0.0
    return out


def retrieval_metrics_by_type(ret_entries: List[dict], qid_to_ref: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    per_type_turn_r5: Dict[str, List[float]] = {}
    per_type_turn_n5: Dict[str, List[float]] = {}
    per_type_sess_r5: Dict[str, List[float]] = {}
    per_type_sess_n5: Dict[str, List[float]] = {}

    overall_turn_r5, overall_turn_n5 = [], []
    overall_sess_r5, overall_sess_n5 = [], []

    for row in ret_entries:
        qid = row["question_id"]
        if qid.endswith("_abs"):
            continue

        ref_entry = qid_to_ref.get(qid)
        if ref_entry is None:
            continue
        qtype = normalize_qtype(ref_entry)

        turn = row["retrieval_results"]["metrics"]["turn"]
        sess = row["retrieval_results"]["metrics"]["session"]
        t_r5 = float(turn["recall_any@5"])
        t_n5 = float(turn["ndcg_any@5"])
        s_r5 = float(sess["recall_any@5"])
        s_n5 = float(sess["ndcg_any@5"])

        overall_turn_r5.append(t_r5)
        overall_turn_n5.append(t_n5)
        overall_sess_r5.append(s_r5)
        overall_sess_n5.append(s_n5)

        per_type_turn_r5.setdefault(qtype, []).append(t_r5)
        per_type_turn_n5.setdefault(qtype, []).append(t_n5)
        per_type_sess_r5.setdefault(qtype, []).append(s_r5)
        per_type_sess_n5.setdefault(qtype, []).append(s_n5)

    def mean_map(metric_map: Dict[str, List[float]]) -> Dict[str, float]:
        return {k: float(np.mean(v)) if v else 0.0 for k, v in metric_map.items()}

    return {
        "turn_recall@5": {"overall": float(np.mean(overall_turn_r5)) if overall_turn_r5 else 0.0, **mean_map(per_type_turn_r5)},
        "turn_ndcg@5": {"overall": float(np.mean(overall_turn_n5)) if overall_turn_n5 else 0.0, **mean_map(per_type_turn_n5)},
        "session_recall@5": {"overall": float(np.mean(overall_sess_r5)) if overall_sess_r5 else 0.0, **mean_map(per_type_sess_r5)},
        "session_ndcg@5": {"overall": float(np.mean(overall_sess_n5)) if overall_sess_n5 else 0.0, **mean_map(per_type_sess_n5)},
    }


def collect_question_types(qid_to_ref: Dict[str, dict]) -> List[str]:
    types = set()
    for entry in qid_to_ref.values():
        types.add(normalize_qtype(entry))
    return sorted(types)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    qid_to_ref = load_reference(args.data_file)
    qtypes = collect_question_types(qid_to_ref)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_results = {}
    retrieval_results = {}

    for cfg in args.configs:
        ret_log_path = Path(args.retrieval_root) / f"config_{cfg}" / "longmemeval_s_cleaned_retrievallog_turn_contriever.jsonl"
        gen_dir = Path(args.generation_root) / f"config_{cfg}"
        eval_path = latest_eval_file(gen_dir)

        if not ret_log_path.exists():
            raise FileNotFoundError(f"Missing retrieval log for {cfg}: {ret_log_path}")
        if eval_path is None or not eval_path.exists():
            raise FileNotFoundError(f"Missing evaluation output for {cfg} in {gen_dir}")

        ret_entries = load_jsonl(ret_log_path)
        eval_entries = load_jsonl(eval_path)
        qa_results[cfg] = qa_accuracy_by_type(eval_entries, qid_to_ref)
        retrieval_results[cfg] = retrieval_metrics_by_type(ret_entries, qid_to_ref)

    qa_headers = ["config", "overall"] + qtypes
    qa_rows = []
    for cfg in args.configs:
        row = [cfg, fmt(qa_results[cfg].get("overall", 0.0))]
        for qtype in qtypes:
            row.append(fmt(qa_results[cfg].get(qtype, 0.0)))
        qa_rows.append(row)

    turn_headers = ["config", "overall"] + qtypes
    turn_rows = []
    for cfg in args.configs:
        row = [cfg, f"{fmt(retrieval_results[cfg]['turn_recall@5']['overall'])}/{fmt(retrieval_results[cfg]['turn_ndcg@5']['overall'])}"]
        for qtype in qtypes:
            row.append(
                f"{fmt(retrieval_results[cfg]['turn_recall@5'].get(qtype, 0.0))}/"
                f"{fmt(retrieval_results[cfg]['turn_ndcg@5'].get(qtype, 0.0))}"
            )
        turn_rows.append(row)

    session_headers = ["config", "overall"] + qtypes
    session_rows = []
    for cfg in args.configs:
        row = [
            cfg,
            f"{fmt(retrieval_results[cfg]['session_recall@5']['overall'])}/{fmt(retrieval_results[cfg]['session_ndcg@5']['overall'])}",
        ]
        for qtype in qtypes:
            row.append(
                f"{fmt(retrieval_results[cfg]['session_recall@5'].get(qtype, 0.0))}/"
                f"{fmt(retrieval_results[cfg]['session_ndcg@5'].get(qtype, 0.0))}"
            )
        session_rows.append(row)

    summary_md = "\n\n".join(
        [
            "# Phase 1 Summary",
            "## QA Accuracy by Question Type",
            build_markdown_table(qa_headers, qa_rows),
            "## Retrieval (Turn) by Question Type",
            "_Cells are `Recall@5/NDCG@5`._",
            build_markdown_table(turn_headers, turn_rows),
            "## Retrieval (Session) by Question Type",
            "_Cells are `Recall@5/NDCG@5`._",
            build_markdown_table(session_headers, session_rows),
        ]
    )

    (output_dir / "summary.md").write_text(summary_md)
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump({"qa": qa_results, "retrieval": retrieval_results}, f, indent=2)

    print(f"Saved summary to {(output_dir / 'summary.md')}")
    print(f"Saved raw results to {(output_dir / 'raw_results.json')}")


if __name__ == "__main__":
    main()
