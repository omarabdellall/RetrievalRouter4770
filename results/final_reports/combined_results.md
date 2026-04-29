# Combined Results (Phase 1 -> Phase 2 -> Phase 3)

## Phase 1 (Ablation Baseline)

- Best fixed config: `C2 (TD)` at **0.690**
- Different question types favor different configs
- Conclusion: fixed retrieval is suboptimal; routing is motivated

Reference: `results/final_reports/phase1_results.md`

## Phase 2 (Oracle Router)

- Type oracle upper bound: **0.706** (+1.6 over C2)
- Per-question oracle ceiling: **0.806**

Interpretation:

- There is measurable type-level routing headroom over fixed `C2`.
- The larger per-question oracle gap indicates additional headroom beyond coarse type labels.

Reference: `results/final_reports/phase2_results.md`

## Phase 3 (Classifier Router)

- Query-only LLM classifier: **0.680** (below C2 by 1.0 point)
- Gain decomposition vs C2: +18 helped, -23 hurt, net -5
- Routing accuracy vs oracle choices: 0.474

Reference: `results/final_reports/phase3_results.md`

## End-to-End Summary

1. Phase 1 establishes `C2` as strongest fixed baseline.
2. Phase 2 shows a modest but real oracle gain over fixed retrieval.
3. Phase 3 shows the current classifier does not yet recover enough signal to beat `C2`.