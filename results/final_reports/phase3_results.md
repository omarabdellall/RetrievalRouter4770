# Phase 3 Results (LLM Classifier Router)

## Objective

Implement and evaluate the real classifier router (query-only) using `gpt-4o-mini`.

## Main Comparison


| Strategy        | Overall | Abst  | KU    | Multi | SS-A  | SS-P  | SS-U  | Temp  |
| --------------- | ------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Best fixed (C2) | 0.690   | 0.767 | 0.806 | 0.504 | 0.821 | 0.600 | 0.906 | 0.638 |
| Type oracle     | 0.706   | 0.867 | 0.806 | 0.504 | 0.821 | 0.700 | 0.922 | 0.646 |
| LLM classifier  | 0.700   | 0.833 | 0.778 | 0.512 | 0.821 | 0.700 | 0.891 | 0.654 |


## Classifier vs C2

- Helped vs C2: 25 questions
- Hurt vs C2: 20 questions
- Net gain: +5 questions (+1.0 points overall)
- Routing accuracy vs oracle choices: 0.604

## Notes

- New v2 prompt + few-shot config routing run used a fresh cache (`classification_cache_v2.json`).
- This moved classifier performance above fixed C2 and closer to type-level oracle.

## Artifacts

- `results/phase3/classifier_hypothesis.jsonl`
- `results/phase3/summary_metrics.json`
- `results/phase3/summary_table.txt`
- `results/phase3/classifier_confusion_matrix.json`

