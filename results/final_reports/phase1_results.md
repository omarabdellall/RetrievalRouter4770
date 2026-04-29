# Phase 1 Results

## Objective
Evaluate all 8 retrieval configurations (`C0`-`C7`) across Query Expansion (QE), Temporal Decay (TD), and MMR Diversity on `longmemeval_s_cleaned` (500 questions).

## Overall QA Accuracy

| Config | QA Acc (overall) |
| --- | ---: |
| C0 (baseline) | 0.6840 |
| C1 (QE) | 0.6500 |
| C2 (TD) | **0.6900** |
| C3 (MMR) | 0.6440 |
| C4 (QE+TD) | 0.6820 |
| C5 (QE+MMR) | 0.6340 |
| C6 (TD+MMR) | 0.6420 |
| C7 (QE+TD+MMR) | 0.6300 |

## Per-Type Winner Summary

| Question Type | Best Config | Score |
| --- | --- | ---: |
| Overall | C2 (TD) | 0.6900 |
| Abstention | C3 (MMR) | 0.8667 |
| Knowledge-Update | C0 / C2 (tie) | 0.8056 |
| Multi-Session | C2 (TD) | 0.5041 |
| Single-Session-Assistant | C2 / C3 / C5 / C6 / C7 (tie) | 0.8214 |
| Single-Session-Preference | C4 (QE+TD) | 0.7000 |
| Single-Session-User | C0 / C3 (tie) | 0.9219 |
| Temporal-Reasoning | C4 (QE+TD) | 0.6457 |

## Takeaways
- No single config dominates every question type.
- `C2` (TD only) is the strongest fixed default overall.
- Stacking all augmentations (`C7`) performs worst overall.
- Results justify adaptive routing rather than one fixed retriever.
