# Phase 2 Results (Oracle Router)

## Objective

Implement and evaluate the Oracle Router as the type-level routing upper bound.

The router reuses precomputed Phase 1 eval labels (`*.eval-results-gpt-4o`) and does not rerun QA judging.

## Main Comparison


| Strategy            | Overall   | Abst  | KU    | Multi | SS-A  | SS-P  | SS-U  | Temp  |
| ------------------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Best fixed (C2)     | 0.690     | 0.767 | 0.806 | 0.504 | 0.821 | 0.600 | 0.906 | 0.638 |
| Type oracle         | **0.706** | 0.867 | 0.806 | 0.504 | 0.821 | 0.700 | 0.922 | 0.646 |
| Per-question oracle | **0.806** | 0.900 | 0.931 | 0.620 | 0.839 | 0.833 | 0.969 | 0.787 |


## Oracle vs C2

- Helped vs C2: 17 questions
- Hurt vs C2: 9 questions
- Net gain: +8 questions (+1.6 points overall)

## Takeaways

- Type-level routing has real but modest headroom over best fixed config.
- Per-question oracle remains much higher than type oracle, indicating headroom from finer-grained routing signals.

## Artifacts

- `results/phase2/oracle_hypothesis.jsonl`
- `results/phase2/summary_metrics.json`
- `results/phase2/summary_table.txt`

