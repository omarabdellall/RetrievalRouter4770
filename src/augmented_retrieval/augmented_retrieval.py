import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backoff
import numpy as np
import openai
import torch
from openai import OpenAI
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.eval_utils import evaluate_retrieval, evaluate_retrieval_turn2session


CONFIG_DEFS = {
    "C0": {"qe": False, "td": False, "mmr": False},
    "C1": {"qe": True, "td": False, "mmr": False},
    "C2": {"qe": False, "td": True, "mmr": False},
    "C3": {"qe": False, "td": False, "mmr": True},
    "C4": {"qe": True, "td": True, "mmr": False},
    "C5": {"qe": True, "td": False, "mmr": True},
    "C6": {"qe": False, "td": True, "mmr": True},
    "C7": {"qe": True, "td": True, "mmr": True},
}

METRIC_K_VALUES = [1, 3, 5, 10, 30, 50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        help="Subset of configs to run, e.g. --configs C0 C2",
    )
    parser.add_argument("--n_questions", type=int, default=500)
    parser.add_argument("--lambda_decay", type=float, default=0.1)
    parser.add_argument("--lambda_mmr", type=float, default=0.5)
    parser.add_argument("--n_expansions", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_n_mmr", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--qe_cache_path",
        type=str,
        default="cache/query_expansion_cache.json",
    )
    return parser.parse_args()


def load_data(path: str) -> List[dict]:
    return json.load(open(path, "r"))


def get_output_filename(in_file: str) -> str:
    return f"{Path(in_file).stem}_retrievallog_turn_contriever.jsonl"


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def init_contriever(device: str) -> Tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever").to(device)
    model.eval()
    return tokenizer, model


def mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]


def embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int = 128,
) -> np.ndarray:
    all_vectors = []
    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            vectors = mean_pooling(outputs[0], inputs["attention_mask"]).detach().cpu().numpy()
            all_vectors.append(vectors)
    if not all_vectors:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(all_vectors, axis=0).astype(np.float32)


def process_item_flat_index(
    session_data: List[dict], sess_id: str, timestamp: str
) -> Tuple[List[str], List[str], List[str]]:
    corpus = []
    ids = []
    for i_turn, turn in enumerate(session_data):
        if turn["role"] != "user":
            continue
        corpus.append(turn["content"])
        cur_id = f"{sess_id}_{i_turn + 1}"
        if "answer" in sess_id:
            has_answer = bool(turn.get("has_answer", False))
            if not has_answer:
                cur_id = cur_id.replace("answer", "noans")
        ids.append(cur_id)
    return corpus, ids, [timestamp for _ in corpus]


def build_turn_corpus(entry: dict) -> Tuple[List[str], List[str], List[str]]:
    corpus, corpus_ids, corpus_timestamps = [], [], []
    for sess_id, sess_data, ts in zip(
        entry["haystack_session_ids"],
        entry["haystack_sessions"],
        entry["haystack_dates"],
    ):
        cur_corpus, cur_ids, cur_ts = process_item_flat_index(sess_data, sess_id, ts)
        corpus.extend(cur_corpus)
        corpus_ids.extend(cur_ids)
        corpus_timestamps.extend(cur_ts)
    return corpus, corpus_ids, corpus_timestamps


def load_qe_cache(cache_path: str) -> Dict[str, List[str]]:
    cache_file = Path(cache_path)
    if not cache_file.exists():
        return {}
    try:
        return json.load(open(cache_file, "r"))
    except Exception:
        return {}


def persist_qe_cache(cache: Dict[str, List[str]], cache_path: str) -> None:
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError), max_tries=6)
def chat_completions_with_backoff(client: OpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)


def safe_parse_json_array(raw_text: str) -> Optional[List[str]]:
    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            cleaned = [str(x).strip() for x in parsed if str(x).strip()]
            return cleaned
    except Exception:
        pass

    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(raw_text[start : end + 1])
            if isinstance(parsed, list):
                cleaned = [str(x).strip() for x in parsed if str(x).strip()]
                return cleaned
        except Exception:
            return None
    return None


def expand_query(
    question_id: str,
    question: str,
    n_expansions: int,
    client: OpenAI,
    cache: Dict[str, List[str]],
    cache_path: str,
) -> List[str]:
    if question_id in cache:
        return cache[question_id][:n_expansions]

    prompt = (
        "Given the following question about past conversations, generate "
        f"{n_expansions} alternative phrasings that preserve the original intent "
        "but use different wording.\n"
        "Return ONLY a JSON array of strings, no other text.\n\n"
        f"Question: {question}"
    )

    try:
        completion = chat_completions_with_backoff(
            client,
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        response_text = completion.choices[0].message.content.strip()
        parsed = safe_parse_json_array(response_text)
        if parsed is None:
            expansions = []
        else:
            expansions = parsed[:n_expansions]
    except Exception:
        expansions = []

    cache[question_id] = expansions
    persist_qe_cache(cache, cache_path)
    return expansions


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str[:10], "%Y/%m/%d")


def compute_temporal_decay(
    question_date_str: str,
    corpus_timestamps: List[str],
    lambda_decay: float,
) -> np.ndarray:
    q_date = parse_date(question_date_str)
    decay = []
    for ts in corpus_timestamps:
        c_date = parse_date(ts)
        delta_days = (q_date - c_date).days
        delta_months = max(0.0, delta_days / 30.0)
        decay.append(math.exp(-lambda_decay * delta_months))
    return np.array(decay, dtype=np.float32)


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.ones_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def mmr_rerank(
    scores: np.ndarray,
    corpus_vectors: np.ndarray,
    lambda_mmr: float,
    top_n: int,
    top_k: int,
) -> np.ndarray:
    full_order = np.argsort(-scores)
    n_docs = len(full_order)
    if n_docs == 0:
        return full_order

    n_candidates = min(top_n, n_docs)
    candidate_indices = full_order[:n_candidates]
    if n_candidates <= 1:
        return full_order

    candidate_scores = scores[candidate_indices]
    relevance = minmax_normalize(candidate_scores)
    candidate_vectors = l2_normalize(corpus_vectors[candidate_indices])
    sim_matrix = candidate_vectors @ candidate_vectors.T

    selected_local = [0]
    remaining_local = list(range(1, n_candidates))
    target_selected = min(top_k, n_candidates)

    while remaining_local and len(selected_local) < target_selected:
        best_idx = None
        best_score = -float("inf")
        for local_idx in remaining_local:
            max_sim_to_selected = float(sim_matrix[local_idx, selected_local].max())
            mmr_score = (
                lambda_mmr * float(relevance[local_idx])
                - (1.0 - lambda_mmr) * max_sim_to_selected
            )
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = local_idx
        selected_local.append(best_idx)
        remaining_local.remove(best_idx)

    selected_global = [candidate_indices[i] for i in selected_local]
    remaining_candidates_global = [candidate_indices[i] for i in range(n_candidates) if i not in selected_local]
    tail_global = full_order[n_candidates:].tolist()
    final_ranking = np.array(selected_global + remaining_candidates_global + tail_global, dtype=np.int64)
    return final_ranking


def compute_metrics(rankings: np.ndarray, correct_docs: List[str], corpus_ids: List[str]) -> Dict[str, Dict[str, float]]:
    metric_out = {"session": {}, "turn": {}}
    for k in METRIC_K_VALUES:
        turn_recall_any, turn_recall_all, turn_ndcg = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
        metric_out["turn"][f"recall_any@{k}"] = turn_recall_any
        metric_out["turn"][f"recall_all@{k}"] = turn_recall_all
        metric_out["turn"][f"ndcg_any@{k}"] = turn_ndcg

        sess_recall_any, sess_recall_all, sess_ndcg = evaluate_retrieval_turn2session(
            rankings, correct_docs, corpus_ids, k=k
        )
        metric_out["session"][f"recall_any@{k}"] = sess_recall_any
        metric_out["session"][f"recall_all@{k}"] = sess_recall_all
        metric_out["session"][f"ndcg_any@{k}"] = sess_ndcg
    return metric_out


def apply_config(
    config: Dict[str, bool],
    base_scores: np.ndarray,
    qe_scores: Optional[np.ndarray],
    td_decay: Optional[np.ndarray],
    corpus_vectors: np.ndarray,
    lambda_mmr: float,
    top_n_mmr: int,
    top_k: int,
) -> np.ndarray:
    scores = qe_scores.copy() if config["qe"] and qe_scores is not None else base_scores.copy()
    if config["td"] and td_decay is not None:
        scores = scores * td_decay
    if config["mmr"]:
        return mmr_rerank(scores, corpus_vectors, lambda_mmr=lambda_mmr, top_n=top_n_mmr, top_k=top_k)
    return np.argsort(-scores)


def compute_correct_docs(corpus_ids: List[str]) -> List[str]:
    return list(set([cid for cid in corpus_ids if "answer" in cid]))


def make_empty_ranking_entry(entry: dict) -> dict:
    zero_metrics = {"session": {}, "turn": {}}
    for k in METRIC_K_VALUES:
        for gran in ["session", "turn"]:
            zero_metrics[gran][f"recall_any@{k}"] = 0.0
            zero_metrics[gran][f"recall_all@{k}"] = 0.0
            zero_metrics[gran][f"ndcg_any@{k}"] = 0.0

    return {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "question": entry["question"],
        "answer": entry["answer"],
        "question_date": entry["question_date"],
        "haystack_dates": entry["haystack_dates"],
        "haystack_sessions": entry["haystack_sessions"],
        "haystack_session_ids": entry["haystack_session_ids"],
        "answer_session_ids": entry["answer_session_ids"],
        "retrieval_results": {
            "query": entry["question"],
            "ranked_items": [],
            "metrics": zero_metrics,
        },
    }


def build_output_entry(
    entry: dict,
    rankings: np.ndarray,
    corpus: List[str],
    corpus_ids: List[str],
    corpus_timestamps: List[str],
    metrics: Dict[str, Dict[str, float]],
) -> dict:
    ranked_items = [
        {
            "corpus_id": corpus_ids[rid],
            "text": corpus[rid],
            "timestamp": corpus_timestamps[rid],
        }
        for rid in rankings.tolist()
    ]
    return {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "question": entry["question"],
        "answer": entry["answer"],
        "question_date": entry["question_date"],
        "haystack_dates": entry["haystack_dates"],
        "haystack_sessions": entry["haystack_sessions"],
        "haystack_session_ids": entry["haystack_session_ids"],
        "answer_session_ids": entry["answer_session_ids"],
        "retrieval_results": {
            "query": entry["question"],
            "ranked_items": ranked_items,
            "metrics": metrics,
        },
    }


def print_aggregated_metrics(results_by_config: Dict[str, List[dict]]) -> None:
    print("\n=== Aggregated Retrieval Metrics (Skipping Abstention / No-Target) ===")
    for config_name, entries in results_by_config.items():
        averaged = {"session": {}, "turn": {}}
        ignored_abs = 0
        ignored_no_target = 0
        for gran in ["session", "turn"]:
            for metric_name in entries[0]["retrieval_results"]["metrics"][gran].keys():
                values = []
                for entry in entries:
                    if "_abs" in entry["question_id"]:
                        ignored_abs += 1
                        continue
                    has_target = any(
                        (turn.get("role") == "user") and bool(turn.get("has_answer", False))
                        for session in entry["haystack_sessions"]
                        for turn in session
                    )
                    if not has_target:
                        ignored_no_target += 1
                        continue
                    values.append(entry["retrieval_results"]["metrics"][gran][metric_name])
                averaged[gran][metric_name] = float(np.mean(values)) if values else 0.0

        print(f"\n[{config_name}]")
        print(
            json.dumps(
                {
                    "ignored_abstention_instances": ignored_abs,
                    "ignored_no_target_instances": ignored_no_target,
                    "averaged_metrics": averaged,
                },
                indent=2,
            )
        )


def validate_configs(config_names: List[str]) -> None:
    unknown = [c for c in config_names if c not in CONFIG_DEFS]
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}. Valid: {list(CONFIG_DEFS.keys())}")


def main(args: argparse.Namespace) -> None:
    validate_configs(args.configs)
    in_data = load_data(args.in_file)
    in_data = in_data[: min(args.n_questions, len(in_data))]

    out_filename = get_output_filename(args.in_file)
    out_files = {}
    for config_name in args.configs:
        out_path = Path("retrieval_logs") / "augmented" / f"config_{config_name}" / out_filename
        ensure_parent_dir(str(out_path))
        out_files[config_name] = open(out_path, "w")

    tokenizer, model = init_contriever(args.device)
    run_qe = any(CONFIG_DEFS[c]["qe"] for c in args.configs)

    qe_cache = {}
    qe_client = None
    if run_qe:
        qe_cache = load_qe_cache(args.qe_cache_path)
        api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("QE requested but no API key found. Pass --openai_api_key or set OPENAI_API_KEY.")
        qe_client = OpenAI(api_key=api_key)

    results_by_config = {c: [] for c in args.configs}

    for entry in tqdm(in_data):
        corpus, corpus_ids, corpus_timestamps = build_turn_corpus(entry)
        if len(corpus) == 0:
            for config_name in args.configs:
                out_entry = make_empty_ranking_entry(entry)
                print(json.dumps(out_entry), file=out_files[config_name], flush=True)
                results_by_config[config_name].append(out_entry)
            continue

        corpus_vectors = embed_texts(
            corpus,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            batch_size=args.batch_size,
        )
        query_vector = embed_texts(
            [entry["question"]],
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            batch_size=args.batch_size,
        )
        base_scores = np.atleast_1d((query_vector @ corpus_vectors.T).squeeze())

        qe_scores = None
        if run_qe:
            expansions = expand_query(
                question_id=entry["question_id"],
                question=entry["question"],
                n_expansions=args.n_expansions,
                client=qe_client,
                cache=qe_cache,
                cache_path=args.qe_cache_path,
            )
            all_queries = [entry["question"]] + expansions
            all_query_vectors = embed_texts(
                all_queries,
                tokenizer=tokenizer,
                model=model,
                device=args.device,
                batch_size=args.batch_size,
            )
            qe_scores = np.atleast_1d((all_query_vectors @ corpus_vectors.T).max(axis=0).squeeze())

        td_decay = None
        if any(CONFIG_DEFS[c]["td"] for c in args.configs):
            td_decay = compute_temporal_decay(entry["question_date"], corpus_timestamps, args.lambda_decay)

        correct_docs = compute_correct_docs(corpus_ids)

        for config_name in args.configs:
            cfg = CONFIG_DEFS[config_name]
            rankings = apply_config(
                config=cfg,
                base_scores=base_scores,
                qe_scores=qe_scores,
                td_decay=td_decay,
                corpus_vectors=corpus_vectors,
                lambda_mmr=args.lambda_mmr,
                top_n_mmr=args.top_n_mmr,
                top_k=args.top_k,
            )
            metrics = compute_metrics(rankings, correct_docs, corpus_ids)
            out_entry = build_output_entry(
                entry=entry,
                rankings=rankings,
                corpus=corpus,
                corpus_ids=corpus_ids,
                corpus_timestamps=corpus_timestamps,
                metrics=metrics,
            )
            print(json.dumps(out_entry), file=out_files[config_name], flush=True)
            results_by_config[config_name].append(out_entry)

    for f in out_files.values():
        f.close()

    print_aggregated_metrics(results_by_config)


if __name__ == "__main__":
    main(parse_args())
