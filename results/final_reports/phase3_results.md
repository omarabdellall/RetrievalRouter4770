# Phase 3 Results (LLM Classifier Router)

## Objective

Implement and evaluate the real classifier router (query-only) using `gpt-4o-mini`.

## Main Comparison


| Strategy        | Overall | Abst  | KU    | Multi | SS-A  | SS-P  | SS-U  | Temp  |
| --------------- | ------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Best fixed (C2) | 0.690   | 0.767 | 0.806 | 0.504 | 0.821 | 0.600 | 0.906 | 0.638 |
| Type oracle     | 0.706   | 0.867 | 0.806 | 0.504 | 0.821 | 0.700 | 0.922 | 0.646 |
| LLM classifier  | 0.680   | 0.800 | 0.778 | 0.463 | 0.821 | 0.667 | 0.891 | 0.638 |


## Classifier vs C2

- Helped vs C2: 18 questions
- Hurt vs C2: 23 questions
- Net gain: -5 questions (-1.0 points overall)
- Routing accuracy vs oracle choices: 0.474

## Notes

- Run completed from cache (`hits=500`, `misses=0`) for efficiency.
- Main failure mode is over-routing to `single-session-user` (and therefore `C0`).

## Artifacts

- `results/phase3/classifier_hypothesis.jsonl`
- `results/phase3/summary_metrics.json`
- `results/phase3/summary_table.txt`
- `results/phase3/classifier_confusion_matrix.json`