# Phase 1 Tables

## Table 1: Overall Ablation Comparison (C0-C7)

| Config | QA Acc (overall) | Turn R@5 | Turn NDCG@5 | Session R@5 | Session NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C0 (baseline) | 0.6840 | **0.8021** | **0.5794** | 0.8489 | **0.6147** |
| C1 (QE) | 0.6500 | 0.7936 | 0.5615 | 0.8447 | 0.5984 |
| C2 (TD) | **0.6900** | **0.8021** | 0.5701 | **0.8532** | 0.6071 |
| C3 (MMR) | 0.6440 | 0.7787 | 0.5425 | 0.8319 | 0.5838 |
| C4 (QE+TD) | 0.6820 | 0.7936 | 0.5560 | **0.8532** | 0.5923 |
| C5 (QE+MMR) | 0.6340 | 0.7745 | 0.5305 | 0.8319 | 0.5740 |
| C6 (TD+MMR) | 0.6420 | 0.7787 | 0.5293 | 0.8362 | 0.5727 |
| C7 (QE+TD+MMR) | 0.6300 | 0.7745 | 0.5128 | 0.8362 | 0.5600 |

## Table 2: QA Accuracy by Question Type

| Config | Overall | Abstention | Knowledge-Update | Multi-Session | Single-Session-Assistant | Single-Session-Preference | Single-Session-User | Temporal-Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 0.6840 | 0.8000 | **0.8056** | 0.4711 | 0.8036 | 0.6000 | **0.9219** | 0.6378 |
| C1 | 0.6500 | 0.7333 | 0.7778 | 0.4463 | 0.8036 | 0.5333 | 0.8750 | 0.5984 |
| C2 | **0.6900** | 0.7667 | **0.8056** | **0.5041** | **0.8214** | 0.6000 | 0.9062 | 0.6378 |
| C3 | 0.6440 | **0.8667** | 0.7222 | 0.4132 | **0.8214** | 0.5667 | **0.9219** | 0.5669 |
| C4 | 0.6820 | 0.8000 | 0.7917 | 0.4628 | 0.8036 | **0.7000** | 0.8750 | **0.6457** |
| C5 | 0.6340 | 0.7667 | 0.7222 | 0.4132 | **0.8214** | 0.5667 | 0.8750 | 0.5748 |
| C6 | 0.6420 | 0.8333 | 0.7083 | 0.3967 | **0.8214** | 0.5000 | 0.8906 | 0.6220 |
| C7 | 0.6300 | 0.8000 | 0.7083 | 0.4050 | **0.8214** | 0.5333 | 0.9062 | 0.5591 |

## Table 3: Winner Summary by Question Type

| Question Type | Best Config | Score |
| --- | --- | ---: |
| Overall | C2 (TD) | 0.6900 |
| Abstention | C3 (MMR) | 0.8667 |
| Knowledge-Update | C0 (baseline) / C2 (TD) *(tie)* | 0.8056 |
| Multi-Session | C2 (TD) | 0.5041 |
| Single-Session-Assistant | C2 / C3 / C5 / C6 / C7 *(tie)* | 0.8214 |
| Single-Session-Preference | C4 (QE+TD) | 0.7000 |
| Single-Session-User | C0 (baseline) / C3 (MMR) *(tie)* | 0.9219 |
| Temporal-Reasoning | C4 (QE+TD) | 0.6457 |
