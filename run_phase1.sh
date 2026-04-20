#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATA_FILE="${DATA_FILE:-data/longmemeval_s_cleaned.json}"
CONFIGS=(C0 C1 C2 C3 C4 C5 C6 C7)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found at ${PYTHON_BIN}. Set PYTHON_BIN to a valid interpreter."
  exit 1
fi

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Data file not found: ${DATA_FILE}"
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required for query expansion, generation, and evaluation."
  exit 1
fi

mkdir -p results/phase1

echo "[1/4] Running augmented retrieval for all configs..."
"${PYTHON_BIN}" src/augmented_retrieval/augmented_retrieval.py \
  --in_file "${DATA_FILE}" \
  --configs "${CONFIGS[@]}"

for cfg in "${CONFIGS[@]}"; do
  echo "[2/4] Running generation for ${cfg}..."
  RETRIEVAL_LOG="retrieval_logs/augmented/config_${cfg}/longmemeval_s_cleaned_retrievallog_turn_contriever.jsonl"
  GEN_OUT_DIR="generation_logs/augmented/config_${cfg}"
  mkdir -p "${GEN_OUT_DIR}"

  if [[ -n "${OPENAI_ORGANIZATION:-}" ]]; then
    "${PYTHON_BIN}" src/generation/run_generation.py \
      --in_file "${RETRIEVAL_LOG}" \
      --out_dir "${GEN_OUT_DIR}" \
      --out_file_suffix "_${cfg}" \
      --model_name "gpt-4o-mini-2024-07-18" \
      --model_alias "gpt-4o-mini" \
      --openai_key "${OPENAI_API_KEY}" \
      --openai_organization "${OPENAI_ORGANIZATION}" \
      --retriever_type "flat-turn" \
      --topk_context 5 \
      --history_format "json" \
      --useronly "false" \
      --cot "true" \
      --con "false" \
      --merge_key_expansion_into_value "none"
  else
    "${PYTHON_BIN}" src/generation/run_generation.py \
      --in_file "${RETRIEVAL_LOG}" \
      --out_dir "${GEN_OUT_DIR}" \
      --out_file_suffix "_${cfg}" \
      --model_name "gpt-4o-mini-2024-07-18" \
      --model_alias "gpt-4o-mini" \
      --openai_key "${OPENAI_API_KEY}" \
      --retriever_type "flat-turn" \
      --topk_context 5 \
      --history_format "json" \
      --useronly "false" \
      --cot "true" \
      --con "false" \
      --merge_key_expansion_into_value "none"
  fi

  GEN_FILE="$(ls -t "${GEN_OUT_DIR}"/*"${cfg}" 2>/dev/null | head -1)"
  if [[ -z "${GEN_FILE}" ]]; then
    echo "Failed to locate generation output for ${cfg}"
    exit 1
  fi

  echo "[3/4] Running QA evaluation for ${cfg}..."
  OPENAI_API_KEY="${OPENAI_API_KEY}" OPENAI_ORGANIZATION="${OPENAI_ORGANIZATION:-}" \
    "${PYTHON_BIN}" src/evaluation/evaluate_qa.py gpt-4o "${GEN_FILE}" "${DATA_FILE}"
done

echo "[4/4] Aggregating Phase 1 results..."
"${PYTHON_BIN}" src/augmented_retrieval/aggregate_results.py \
  --data_file "${DATA_FILE}" \
  --retrieval_root "retrieval_logs/augmented" \
  --generation_root "generation_logs/augmented" \
  --output_dir "results/phase1"

echo "Phase 1 complete. Summary: results/phase1/summary.md"
