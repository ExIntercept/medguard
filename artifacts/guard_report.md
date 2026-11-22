| signal | AUROC | TPR@FPR5% |
|---|---|---|
| inj | 0.993 | 0.974 |
| inj_smooth_mean | 0.990 | 0.967 |
| inj_smooth_std | 0.261 | 0.073 |
| template_sim | 0.990 | 0.934 |
| obfuscation | 0.513 | 0.026 |
| offscope | 0.795 | 0.060 |
| instr_regex | 0.714 | 0.430 |
| **fused (all)** | **0.996** | **0.980** |

Leave-one-signal-out (refit on fit set, scored on eval set):

| dropped | AUROC | delta |
|---|---|---|
| inj | 0.995 | -0.000 |
| inj_smooth_mean | 0.994 | -0.001 |
| inj_smooth_std | 0.995 | -0.000 |
| template_sim | 0.995 | -0.001 |
| obfuscation | 0.996 | -0.000 |
| offscope | 0.998 | +0.002 |
| instr_regex | 0.995 | -0.000 |

Operating point (default thresholds):

| set | n | attacks flagged (>=harden) | attacks blocked | benign flagged | benign blocked |
|---|---|---|---|---|---|
| jackhhao_test | 262 | 136/139 | 132/139 | 3/123 | 2/123 |
| medmcqa_val_benign | 300 | - | - | 0/300 | 0/300 |
| med_attacks_custom | 20 | 8/12 | 7/12 | 1/8 | 0/8 |
