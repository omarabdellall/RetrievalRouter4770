import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CONFIGS = [f"C{i}" for i in range(8)]

ORACLE_TYPE_TO_CONFIG = {
    "temporal-reasoning": "C4",
    "knowledge-update": "C2",
    "multi-session": "C2",
    "single-session-user": "C0",
    "single-session-assistant": "C2",
    "single-session-preference": "C4",
    "abstention": "C3",
}

TYPE_DISPLAY_ORDER = [
    "overall",
    "abstention",
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
]
TYPE_HEADER_NAMES = [
    "overall",
    "abst",
    "ku",
    "multi",
    "ss-a",
    "ss-p",
    "ss-u",
    "temp",
]

CLASSIFIER_PROMPT_VERSION = "v2_config_router_fewshot"

# Routing labels -> retrieval config.
# DEFAULT should dominate when uncertain to avoid harmful over-routing away from C2.
ROUTE_LABEL_TO_CONFIG = {
    "DEFAULT": "C2",
    "TEMPORAL": "C4",
    "PREFERENCE": "C4",
    "ABSTENTION": "C3",
    "SIMPLE_FACT": "C0",
}
ROUTE_LABEL_ORDER = ["DEFAULT", "TEMPORAL", "PREFERENCE", "ABSTENTION", "SIMPLE_FACT", "INVALID"]
VALID_ROUTE_LABELS = set(ROUTE_LABEL_TO_CONFIG.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 oracle + LLM classifier router.")
    parser.add_argument("--data_file", type=str, default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--generation_root", type=str, default="generation_logs/augmented")
    parser.add_argument("--output_dir", type=str, default="results/phase2")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["oracle", "classifier", "all"],
        default="all",
    )
    parser.add_argument(
        "--classifier_model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model used for query-only classification.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="Optional override for OPENAI_API_KEY (needed for classifier mode).",
    )
    parser.add_argument(
        "--openai_organization",
        type=str,
        default=None,
        help="Optional override for OPENAI_ORGANIZATION.",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        help="Optional cache path for classifier predictions. Defaults to <output_dir>/classification_cache.json",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum retries per classifier API call.",
    )
    parser.add_argument(
        "--retry_sleep_seconds",
        type=float,
        default=1.5,
        help="Base sleep time (seconds) between retries.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_qtype(entry: dict) -> str:
    if entry["question_id"].endswith("_abs"):
        return "abstention"
    return entry["question_type"]


def load_reference(data_file: Path) -> Tuple[List[dict], Dict[str, dict], Dict[str, str]]:
    entries = json.loads(data_file.read_text())
    qid_to_entry = {entry["question_id"]: entry for entry in entries}
    qid_to_type = {qid: normalize_qtype(entry) for qid, entry in qid_to_entry.items()}
    return entries, qid_to_entry, qid_to_type


def choose_latest(paths: List[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No candidate files found.")
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def discover_generation_and_eval_files(generation_root: Path) -> Dict[str, Dict[str, Path]]:
    discovered: Dict[str, Dict[str, Path]] = {}
    for cfg in CONFIGS:
        cfg_dir = generation_root / f"config_{cfg}"
        if not cfg_dir.exists():
            raise FileNotFoundError(f"Missing directory for {cfg}: {cfg_dir}")
        eval_candidates = list(cfg_dir.glob("*.eval-results-gpt-4o"))
        if not eval_candidates:
            raise FileNotFoundError(f"Missing eval file for {cfg} in {cfg_dir}")
        eval_path = choose_latest(eval_candidates)

        generation_candidates = [
            p
            for p in cfg_dir.iterdir()
            if p.is_file() and not p.name.endswith(".eval-results-gpt-4o")
        ]
        if not generation_candidates:
            raise FileNotFoundError(f"Missing generation output for {cfg} in {cfg_dir}")
        generation_path = choose_latest(generation_candidates)
        discovered[cfg] = {"generation": generation_path, "eval": eval_path}
    return discovered


def load_config_outputs(
    discovered_paths: Dict[str, Dict[str, Path]],
    expected_qids: set,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, bool]]]:
    config_hypotheses: Dict[str, Dict[str, str]] = {}
    config_labels: Dict[str, Dict[str, bool]] = {}

    for cfg, files in discovered_paths.items():
        gen_rows = load_jsonl(files["generation"])
        eval_rows = load_jsonl(files["eval"])

        hypothesis_map = {row["question_id"]: row["hypothesis"] for row in gen_rows}
        label_map = {row["question_id"]: bool(row["autoeval_label"]["label"]) for row in eval_rows}

        if set(hypothesis_map.keys()) != expected_qids:
            missing = sorted(expected_qids - set(hypothesis_map.keys()))
            extra = sorted(set(hypothesis_map.keys()) - expected_qids)
            raise ValueError(
                f"{cfg} generation qid mismatch. "
                f"Missing={len(missing)} Extra={len(extra)}"
            )
        if set(label_map.keys()) != expected_qids:
            missing = sorted(expected_qids - set(label_map.keys()))
            extra = sorted(set(label_map.keys()) - expected_qids)
            raise ValueError(
                f"{cfg} eval qid mismatch. "
                f"Missing={len(missing)} Extra={len(extra)}"
            )

        config_hypotheses[cfg] = hypothesis_map
        config_labels[cfg] = label_map

    return config_hypotheses, config_labels


def summarize_discovery(discovered_paths: Dict[str, Dict[str, Path]]) -> None:
    print("Discovered Phase 1 files:")
    for cfg in CONFIGS:
        files = discovered_paths[cfg]
        print(f"  {cfg}")
        print(f"    generation: {files['generation']}")
        print(f"    eval:       {files['eval']}")


def compute_accuracy_from_assignments(
    qid_to_assigned_config: Dict[str, str],
    qid_to_type: Dict[str, str],
    config_labels: Dict[str, Dict[str, bool]],
) -> Dict[str, float]:
    by_type: Dict[str, List[int]] = defaultdict(list)
    overall = []
    for qid, cfg in qid_to_assigned_config.items():
        label = 1 if config_labels[cfg][qid] else 0
        qtype = qid_to_type[qid]
        overall.append(label)
        by_type[qtype].append(label)

    metrics = {"overall": sum(overall) / len(overall) if overall else 0.0}
    for qtype in sorted(by_type.keys()):
        vals = by_type[qtype]
        metrics[qtype] = sum(vals) / len(vals) if vals else 0.0
    return metrics


def build_hypothesis_rows(
    qid_to_assigned_config: Dict[str, str],
    config_hypotheses: Dict[str, Dict[str, str]],
) -> List[dict]:
    rows = []
    for qid in sorted(qid_to_assigned_config.keys()):
        cfg = qid_to_assigned_config[qid]
        rows.append({"question_id": qid, "hypothesis": config_hypotheses[cfg][qid]})
    return rows


def oracle_assignments(qid_to_type: Dict[str, str]) -> Dict[str, str]:
    return {qid: ORACLE_TYPE_TO_CONFIG[qtype] for qid, qtype in qid_to_type.items()}


def per_question_oracle_metrics(
    qid_to_type: Dict[str, str],
    config_labels: Dict[str, Dict[str, bool]],
) -> Dict[str, float]:
    by_type: Dict[str, List[int]] = defaultdict(list)
    all_vals = []
    for qid, qtype in qid_to_type.items():
        label = 1 if any(config_labels[cfg][qid] for cfg in CONFIGS) else 0
        by_type[qtype].append(label)
        all_vals.append(label)
    out = {"overall": sum(all_vals) / len(all_vals) if all_vals else 0.0}
    for qtype in sorted(by_type.keys()):
        vals = by_type[qtype]
        out[qtype] = sum(vals) / len(vals) if vals else 0.0
    return out


def canonicalize_route_label(raw_text: str) -> str:
    s = raw_text.strip().lower()
    s = s.replace("_", "-")
    s = s.replace("simple-fact", "simple fact")
    s = s.replace("single-fact", "simple fact")
    s = s.replace("single fact", "simple fact")

    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    s = s.strip(" .\"'")

    aliases = {
        "default": "DEFAULT",
        "c2": "DEFAULT",
        "temporal": "TEMPORAL",
        "c4": "TEMPORAL",
        "preference": "PREFERENCE",
        "abstention": "ABSTENTION",
        "c3": "ABSTENTION",
        "simple fact": "SIMPLE_FACT",
        "simple_fact": "SIMPLE_FACT",
        "c0": "SIMPLE_FACT",
    }
    if s in aliases:
        return aliases[s]
    return "INVALID"


def route_label_from_true_type(qtype: str) -> str:
    if qtype == "temporal-reasoning":
        return "TEMPORAL"
    if qtype == "single-session-preference":
        return "PREFERENCE"
    if qtype == "abstention":
        return "ABSTENTION"
    if qtype == "single-session-user":
        return "SIMPLE_FACT"
    return "DEFAULT"


def build_fewshot_examples(
    data_entries: List[dict],
    qid_to_type: Dict[str, str],
    config_labels: Dict[str, Dict[str, bool]],
    max_per_label: int = 2,
) -> Dict[str, List[dict]]:
    qid_to_entry = {entry["question_id"]: entry for entry in data_entries}
    oracle_cfg_by_qid = oracle_assignments(qid_to_type)
    buckets: Dict[str, List[dict]] = defaultdict(list)

    for qid in sorted(qid_to_type.keys()):
        oracle_cfg = oracle_cfg_by_qid[qid]
        if oracle_cfg == "C2":
            continue
        if config_labels[oracle_cfg][qid] and not config_labels["C2"][qid]:
            qtype = qid_to_type[qid]
            label = route_label_from_true_type(qtype)
            if label == "DEFAULT":
                continue
            buckets[label].append(
                {
                    "question_id": qid,
                    "question": qid_to_entry[qid]["question"],
                    "true_type": qtype,
                    "oracle_config": oracle_cfg,
                }
            )

    selected: Dict[str, List[dict]] = {}
    for label in ["TEMPORAL", "PREFERENCE", "ABSTENTION", "SIMPLE_FACT"]:
        selected[label] = buckets[label][:max_per_label]
    return selected


def format_fewshot_block(fewshot_examples: Dict[str, List[dict]]) -> str:
    lines: List[str] = []
    for label in ["TEMPORAL", "PREFERENCE", "ABSTENTION", "SIMPLE_FACT"]:
        examples = fewshot_examples.get(label, [])
        if not examples:
            continue
        lines.append(f"{label} examples:")
        for ex in examples:
            lines.append(f'- Q: "{ex["question"]}"')
            lines.append(f"  A: {label}")
    return "\n".join(lines)


def build_classifier_prompt(question_text: str, fewshot_examples: Dict[str, List[dict]]) -> str:
    fewshot_block = format_fewshot_block(fewshot_examples)
    return f"""You are a retrieval router for a conversational AI memory system.
Given a user's question, decide which retrieval strategy to use.

Strategies:
- DEFAULT: General retrieval with temporal weighting. Use for most questions, especially facts/counts and potentially updated or multi-conversation information.
- TEMPORAL: Enhanced temporal + query expansion. Use ONLY when the question explicitly asks when something happened, temporal ordering, or duration.
- PREFERENCE: Same retrieval config as TEMPORAL. Use for subjective user likes/dislikes, tastes, or personal preferences.
- ABSTENTION: Diversity-focused retrieval. Use ONLY when the question is likely asking about something never discussed.
- SIMPLE_FACT: Baseline retrieval. Use for direct single-fact user recall with no strong temporal/update/multi-session signal.

If unsure, choose DEFAULT.

Few-shot examples:
{fewshot_block}

Question: "{question_text}"

Answer with ONLY one of: DEFAULT, TEMPORAL, PREFERENCE, ABSTENTION, SIMPLE_FACT"""


def create_openai_client(api_key: str = None, organization: str = None):
    from openai import OpenAI

    final_key = api_key or os.getenv("OPENAI_API_KEY")
    if not final_key:
        raise ValueError("OPENAI_API_KEY is missing. Set env var or pass --openai_api_key.")

    kwargs = {"api_key": final_key}
    final_org = organization or os.getenv("OPENAI_ORGANIZATION")
    if final_org:
        kwargs["organization"] = final_org
    return OpenAI(**kwargs)


def classify_question(
    client,
    prompt: str,
    model: str,
    max_retries: int,
    retry_sleep_seconds: float,
) -> str:
    last_error = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=16,
            )
            raw = completion.choices[0].message.content or ""
            return canonicalize_route_label(raw)
        except Exception as exc:  # pragma: no cover - runtime safety for API retries.
            last_error = exc
            sleep_s = retry_sleep_seconds * (2 ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(f"Classification failed after {max_retries} retries: {last_error}")


def load_cache(cache_file: Path) -> Dict[str, dict]:
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return {}


def save_cache(cache_file: Path, cache: Dict[str, dict]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=True))


def run_classifier_routing(
    data_entries: List[dict],
    fewshot_examples: Dict[str, List[dict]],
    cache_file: Path,
    model: str,
    api_key: str,
    organization: str,
    max_retries: int,
    retry_sleep_seconds: float,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, dict]]:
    cache = load_cache(cache_file)
    client = None
    predicted_label_by_qid: Dict[str, str] = {}
    assigned_config_by_qid: Dict[str, str] = {}

    cache_hits = 0
    cache_misses = 0
    invalid_count = 0

    for idx, entry in enumerate(data_entries, start=1):
        qid = entry["question_id"]
        question = entry["question"]

        cached = cache.get(qid)
        if (
            cached
            and cached.get("model") == model
            and cached.get("prompt_version") == CLASSIFIER_PROMPT_VERSION
            and cached.get("predicted_label")
        ):
            predicted_label = cached["predicted_label"]
            cache_hits += 1
        else:
            if client is None:
                client = create_openai_client(api_key=api_key, organization=organization)
            prompt = build_classifier_prompt(question_text=question, fewshot_examples=fewshot_examples)
            predicted_label = classify_question(
                client=client,
                prompt=prompt,
                model=model,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
            )
            cache[qid] = {
                "predicted_label": predicted_label,
                "model": model,
                "prompt_version": CLASSIFIER_PROMPT_VERSION,
                "question": question,
            }
            cache_misses += 1
            if cache_misses % 20 == 0:
                save_cache(cache_file, cache)

        if predicted_label not in VALID_ROUTE_LABELS:
            predicted_label_by_qid[qid] = "INVALID"
            assigned_config_by_qid[qid] = "C2"
            invalid_count += 1
        else:
            predicted_label_by_qid[qid] = predicted_label
            assigned_config_by_qid[qid] = ROUTE_LABEL_TO_CONFIG[predicted_label]

        if idx % 50 == 0:
            print(f"Classifier progress: {idx}/{len(data_entries)}")

    save_cache(cache_file, cache)
    print(
        f"Classifier cache summary: hits={cache_hits}, misses={cache_misses}, invalid_predictions={invalid_count}"
    )
    return assigned_config_by_qid, predicted_label_by_qid, cache


def build_confusion_matrix(
    qid_to_true_type: Dict[str, str],
    qid_to_pred_label: Dict[str, str],
) -> Dict[str, Dict[str, int]]:
    true_labels = sorted(set(qid_to_true_type.values()))
    pred_labels = [label for label in ROUTE_LABEL_ORDER if label in set(qid_to_pred_label.values()) | {"INVALID"}]
    for pred in sorted(set(qid_to_pred_label.values())):
        if pred not in pred_labels:
            pred_labels.append(pred)

    matrix: Dict[str, Dict[str, int]] = {}
    for true_label in true_labels:
        matrix[true_label] = {pred_label: 0 for pred_label in pred_labels}

    for qid, true_type in qid_to_true_type.items():
        pred_type = qid_to_pred_label.get(qid, "INVALID")
        if true_type not in matrix:
            matrix[true_type] = {pred_label: 0 for pred_label in pred_labels}
        if pred_type not in matrix[true_type]:
            for row in matrix.values():
                row[pred_type] = 0
        matrix[true_type][pred_type] += 1
    return matrix


def routing_accuracy(
    qid_to_assigned_config: Dict[str, str],
    qid_to_oracle_config: Dict[str, str],
) -> float:
    matches = sum(
        1
        for qid, assigned_cfg in qid_to_assigned_config.items()
        if assigned_cfg == qid_to_oracle_config[qid]
    )
    return matches / len(qid_to_assigned_config) if qid_to_assigned_config else 0.0


def gain_decomposition_against_c2(
    qid_to_assigned_config: Dict[str, str],
    config_labels: Dict[str, Dict[str, bool]],
) -> Dict[str, int]:
    helped = 0
    hurt = 0
    tied_correct = 0
    tied_wrong = 0
    switched = 0

    for qid, assigned_cfg in qid_to_assigned_config.items():
        assigned_correct = bool(config_labels[assigned_cfg][qid])
        c2_correct = bool(config_labels["C2"][qid])
        if assigned_cfg != "C2":
            switched += 1
        if assigned_correct and not c2_correct:
            helped += 1
        elif c2_correct and not assigned_correct:
            hurt += 1
        elif assigned_correct and c2_correct:
            tied_correct += 1
        else:
            tied_wrong += 1

    return {
        "helped_vs_c2": helped,
        "hurt_vs_c2": hurt,
        "net_gain_vs_c2": helped - hurt,
        "switched_from_c2": switched,
        "tied_correct": tied_correct,
        "tied_wrong": tied_wrong,
    }


def format_metric_row(name: str, metrics: Dict[str, float]) -> str:
    cols = [name.ljust(24)]
    for key in TYPE_DISPLAY_ORDER:
        cols.append(f"{metrics.get(key, 0.0):.3f}".rjust(8))
    return " ".join(cols)


def print_table(rows: List[Tuple[str, Dict[str, float]]]) -> str:
    header = (
        "Strategy".ljust(24)
        + " "
        + " ".join([name.rjust(8) for name in TYPE_HEADER_NAMES])
    )
    lines = [header]
    for name, metrics in rows:
        lines.append(format_metric_row(name, metrics))
    table_text = "\n".join(lines)
    print(table_text)
    return table_text


def main() -> None:
    args = parse_args()

    data_file = Path(args.data_file)
    generation_root = Path(args.generation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = Path(args.cache_file) if args.cache_file else output_dir / "classification_cache.json"

    data_entries, qid_to_entry, qid_to_type = load_reference(data_file)
    expected_qids = set(qid_to_entry.keys())

    discovered_paths = discover_generation_and_eval_files(generation_root)
    summarize_discovery(discovered_paths)

    config_hypotheses, config_labels = load_config_outputs(discovered_paths, expected_qids)
    print(f"\nLoaded all config outputs. Verified {len(expected_qids)} questions per config.")

    oracle_cfg_by_qid = oracle_assignments(qid_to_type)
    c2_cfg_by_qid = {qid: "C2" for qid in qid_to_entry.keys()}

    c2_metrics = compute_accuracy_from_assignments(c2_cfg_by_qid, qid_to_type, config_labels)
    oracle_metrics = compute_accuracy_from_assignments(oracle_cfg_by_qid, qid_to_type, config_labels)
    pq_oracle_metrics = per_question_oracle_metrics(qid_to_type, config_labels)

    out_payload = {
        "metadata": {
            "data_file": str(data_file),
            "generation_root": str(generation_root),
            "output_dir": str(output_dir),
            "mode": args.mode,
            "classifier_model": args.classifier_model,
        },
        "baselines": {
            "best_fixed_c2": c2_metrics,
            "oracle_type_level": oracle_metrics,
            "oracle_per_question": pq_oracle_metrics,
        },
        "file_discovery": {
            cfg: {"generation": str(v["generation"]), "eval": str(v["eval"])}
            for cfg, v in discovered_paths.items()
        },
    }

    if args.mode in {"oracle", "all"}:
        oracle_rows = build_hypothesis_rows(oracle_cfg_by_qid, config_hypotheses)
        oracle_hyp_path = output_dir / "oracle_hypothesis.jsonl"
        write_jsonl(oracle_hyp_path, oracle_rows)
        oracle_gain = gain_decomposition_against_c2(oracle_cfg_by_qid, config_labels)
        out_payload["oracle"] = {
            "hypothesis_file": str(oracle_hyp_path),
            "metrics": oracle_metrics,
            "gain_vs_c2": oracle_gain,
        }
        print(f"\nSaved oracle hypothesis file to {oracle_hyp_path}")

    classifier_metrics = None
    classifier_payload = None

    if args.mode in {"classifier", "all"}:
        fewshot_examples = build_fewshot_examples(
            data_entries=data_entries,
            qid_to_type=qid_to_type,
            config_labels=config_labels,
            max_per_label=2,
        )
        print("\nFew-shot selection summary:")
        for label in ["TEMPORAL", "PREFERENCE", "ABSTENTION", "SIMPLE_FACT"]:
            print(f"  {label}: {len(fewshot_examples.get(label, []))} examples")

        assigned_cfg_by_qid, predicted_label_by_qid, _ = run_classifier_routing(
            data_entries=data_entries,
            fewshot_examples=fewshot_examples,
            cache_file=cache_file,
            model=args.classifier_model,
            api_key=args.openai_api_key,
            organization=args.openai_organization,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )

        classifier_rows = build_hypothesis_rows(assigned_cfg_by_qid, config_hypotheses)
        classifier_hyp_path = output_dir / "classifier_hypothesis.jsonl"
        write_jsonl(classifier_hyp_path, classifier_rows)

        classifier_metrics = compute_accuracy_from_assignments(assigned_cfg_by_qid, qid_to_type, config_labels)
        confusion = build_confusion_matrix(qid_to_type, predicted_label_by_qid)
        route_acc = routing_accuracy(assigned_cfg_by_qid, oracle_cfg_by_qid)
        classifier_gain = gain_decomposition_against_c2(assigned_cfg_by_qid, config_labels)

        classifier_payload = {
            "hypothesis_file": str(classifier_hyp_path),
            "metrics": classifier_metrics,
            "routing_accuracy_against_oracle": route_acc,
            "gain_vs_c2": classifier_gain,
            "predicted_label_counts": dict(Counter(predicted_label_by_qid.values())),
            "assigned_config_counts": dict(Counter(assigned_cfg_by_qid.values())),
            "confusion_matrix": confusion,
            "cache_file": str(cache_file),
            "prompt_version": CLASSIFIER_PROMPT_VERSION,
            "fewshot_examples": fewshot_examples,
        }
        out_payload["classifier"] = classifier_payload
        print(f"\nSaved classifier hypothesis file to {classifier_hyp_path}")

    rows_for_table: List[Tuple[str, Dict[str, float]]] = [
        ("Best fixed (C2)", c2_metrics),
        ("Type oracle", oracle_metrics),
        ("Per-question oracle", pq_oracle_metrics),
    ]
    if classifier_metrics is not None:
        rows_for_table.append(("LLM classifier", classifier_metrics))

    print("\nComparison table:")
    table_text = print_table(rows_for_table)

    summary_txt_path = output_dir / "summary_table.txt"
    summary_txt_path.write_text(table_text + "\n")

    metrics_json_path = output_dir / "summary_metrics.json"
    metrics_json_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=True))

    if classifier_payload is not None:
        confusion_path = output_dir / "classifier_confusion_matrix.json"
        confusion_path.write_text(
            json.dumps(classifier_payload["confusion_matrix"], indent=2, ensure_ascii=True)
        )

    print(f"\nSaved summary table to {summary_txt_path}")
    print(f"Saved summary metrics to {metrics_json_path}")
    if classifier_payload is not None:
        print(f"Saved confusion matrix to {output_dir / 'classifier_confusion_matrix.json'}")


if __name__ == "__main__":
    main()
