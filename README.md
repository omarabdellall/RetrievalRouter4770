# RetrievalRouter4770

Adaptive retrieval for LongMemEval
Omar Abdellall, Aryan Agarwal

This repository extends [LongMemEval](https://github.com/xiaowu0162/longmemeval) with a full Phase 1 combinatorial ablation over three retrieval augmentations:

- Query Expansion (QE)
- Temporal Decay (TD)
- MMR Diversity (MMR)

Phase 1 evaluates all 8 configs (`C0`-`C7`) on `longmemeval_s_cleaned` and reports:

- QA accuracy (LLM-as-judge)
- Retrieval metrics (Recall@5, NDCG@5; turn + session)
- Per-question-type breakdown

## Repository additions for this project

- `src/augmented_retrieval/augmented_retrieval.py`  
Runs all ablation configs and writes retrieval logs in generation-compatible format.
- `src/augmented_retrieval/aggregate_results.py`  
Aggregates per-config QA/retrieval results into summary tables.
- `run_phase1.sh`  
End-to-end orchestrator: retrieval -> generation -> QA evaluation -> aggregation.
- `results/phase1/`  
Generated Phase 1 outputs and tables.

## 1) Environment setup

Use Python 3.14 venv (same as current project execution):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-full.txt
pip install tiktoken
```

If `requirements-full.txt` install is too heavy for your machine, install the minimal set used by Phase 1:

```bash
pip install torch transformers openai backoff tqdm numpy==1.26.4 scikit-learn tiktoken
```

Note: `numpy==1.26.4` is required because LongMemEval's eval utils call `np.asfarray`.

## 2) Data setup

Download the cleaned benchmark files:

```bash
mkdir -p data
cd data
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" -o "longmemeval_s_cleaned.json"
curl -L "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" -o "longmemeval_oracle.json"
cd ..
```

## 3) API key setup

Create a local `.env` file (already gitignored) with:

```bash
OPENAI_API_KEY=your_key_here
# optional if your org requires it:
# OPENAI_ORGANIZATION=your_org_here
```

No quotes are required.

## 4) Run the full Phase 1 pipeline

```bash
source .venv/bin/activate
set -a && source .env && set +a
./run_phase1.sh
```

What this does:

1. Runs augmented retrieval for all configs `C0`-`C7`.
2. Runs generation with `gpt-4o-mini-2024-07-18`.
3. Runs QA judging with `gpt-4o-2024-08-06`.
4. Aggregates all outputs to `results/phase1/`.

## 5) Main outputs

- `results/phase1/summary.md`  
Aggregated QA + retrieval tables.
- `results/phase1/raw_results.json`  
Raw structured metrics by config/type.
- `results/phase1/tables.md`  
Assignment-ready tables (overall, per-type, winner summary).

Intermediate logs:

- `retrieval_logs/augmented/config_C*/longmemeval_s_cleaned_retrievallog_turn_contriever.jsonl`
- `generation_logs/augmented/config_C*/...`

## 6) Quick smoke test (before full run)

```bash
source .venv/bin/activate
.venv/bin/python src/augmented_retrieval/augmented_retrieval.py \
  --in_file data/longmemeval_s_cleaned.json \
  --configs C0 C2 \
  --n_questions 10 \
  --device cpu
```

## 7) Reproducibility checklist response

- Code and scripts provided for full experiment pipeline
- Exact commands documented
- Data source and download commands documented
- Environment setup documented
- Randomness/API nondeterminism caveat documented
- All reported metrics generated from committed code paths
- Output artifact locations documented

Notes:

- OpenAI-based generation/judging introduces minor run-to-run variance.
- Costs and wall-clock runtime depend on API latency and local hardware.

## 8) Upstream attribution

This project is built on LongMemEval:

```bibtex
@article{wu2024longmemeval,
  title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author={Wu, Di and Wang, Hongwei and Yu, Wenhao and Zhang, Yuwei and Chang, Kai-Wei and Yu, Dong},
  year={2024},
  eprint={2410.10813},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

