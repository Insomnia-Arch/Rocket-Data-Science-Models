# Published model evaluation

These are aggregate diagnostics from the published v1.3 snapshot, not a new analysis run.
The snapshot contains 2,195,201 matches and began at 2026-09-26T02:20:42.513097+00:00.

The full diagnostics, including calibration, ROC points, feature importance, coefficients, playlist checks and unseen-player checks, are in [model-evaluation.json](model-evaluation.json). They contain no player or match records.

Each variant chooses a model by validation AUC. The table retains all three models; selected models have an asterisk. Scores use completed-match statistics and are not evidence of pre-match prediction or causal car effects.

## Match partitions

| View | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| all-all | 1,276,928 | 458,218 | 460,055 |
| all-duels | 505,117 | 168,618 | 168,451 |
| all-team | 794,872 | 282,173 | 275,970 |
| ranked-all | 697,869 | 246,968 | 242,358 |
| ranked-duels | 272,848 | 90,693 | 90,742 |
| ranked-team | 439,995 | 148,454 | 144,463 |
| casual-all | 602,139 | 203,035 | 202,832 |
| casual-duels | 233,040 | 77,459 | 77,404 |
| casual-team | 372,105 | 124,220 | 123,778 |

## Classifier results

Full uses the allowed gameplay statistics; Style also excludes shots and saves. AUC is not accuracy. Accuracy is shown as a fraction; lower Brier scores are better.

| View | Variant | Model | Validation AUC | Test AUC | Accuracy | Brier | Baseline accuracy | Baseline Brier |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all-all | Full | Logistic regression | 0.9498 | 0.9512 | 0.8746 | 0.0898 | 0.5061 | 0.2500 |
| all-all | Full | Random forest | 0.9364 | 0.9370 | 0.8574 | 0.1057 | 0.5061 | 0.2500 |
| all-all | Full | Gradient boosting * | 0.9513 | 0.9519 | 0.8751 | 0.0881 | 0.5061 | 0.2500 |
| all-all | Style | Logistic regression | 0.8857 | 0.8879 | 0.7992 | 0.1366 | 0.5061 | 0.2500 |
| all-all | Style | Random forest | 0.8857 | 0.8865 | 0.7990 | 0.1399 | 0.5061 | 0.2500 |
| all-all | Style | Gradient boosting * | 0.8906 | 0.8918 | 0.8038 | 0.1340 | 0.5061 | 0.2500 |
| all-duels | Full | Logistic regression | 0.9695 | 0.9695 | 0.9030 | 0.0692 | 0.5233 | 0.2500 |
| all-duels | Full | Random forest | 0.9627 | 0.9630 | 0.8940 | 0.0798 | 0.5233 | 0.2500 |
| all-duels | Full | Gradient boosting * | 0.9697 | 0.9698 | 0.9034 | 0.0691 | 0.5233 | 0.2500 |
| all-duels | Style | Logistic regression | 0.9147 | 0.9156 | 0.8304 | 0.1178 | 0.5233 | 0.2500 |
| all-duels | Style | Random forest | 0.9068 | 0.9075 | 0.8239 | 0.1272 | 0.5233 | 0.2500 |
| all-duels | Style | Gradient boosting * | 0.9162 | 0.9169 | 0.8303 | 0.1167 | 0.5233 | 0.2500 |
| all-team | Full | Logistic regression * | 0.9533 | 0.9532 | 0.8778 | 0.0865 | 0.5044 | 0.2500 |
| all-team | Full | Random forest | 0.9363 | 0.9355 | 0.8570 | 0.1062 | 0.5044 | 0.2500 |
| all-team | Full | Gradient boosting | 0.9522 | 0.9520 | 0.8755 | 0.0878 | 0.5044 | 0.2500 |
| all-team | Style | Logistic regression * | 0.9007 | 0.8990 | 0.8107 | 0.1293 | 0.5044 | 0.2500 |
| all-team | Style | Random forest | 0.8877 | 0.8858 | 0.7981 | 0.1396 | 0.5044 | 0.2500 |
| all-team | Style | Gradient boosting | 0.8996 | 0.8979 | 0.8098 | 0.1300 | 0.5044 | 0.2500 |
| ranked-all | Full | Logistic regression | 0.9516 | 0.9514 | 0.8747 | 0.0891 | 0.5062 | 0.2500 |
| ranked-all | Full | Random forest | 0.9394 | 0.9387 | 0.8585 | 0.1039 | 0.5062 | 0.2500 |
| ranked-all | Full | Gradient boosting * | 0.9521 | 0.9519 | 0.8757 | 0.0880 | 0.5062 | 0.2500 |
| ranked-all | Style | Logistic regression | 0.8915 | 0.8895 | 0.8003 | 0.1355 | 0.5062 | 0.2500 |
| ranked-all | Style | Random forest | 0.8905 | 0.8883 | 0.7990 | 0.1381 | 0.5062 | 0.2500 |
| ranked-all | Style | Gradient boosting * | 0.8945 | 0.8927 | 0.8042 | 0.1334 | 0.5062 | 0.2500 |
| ranked-duels | Full | Logistic regression * | 0.9702 | 0.9700 | 0.9032 | 0.0685 | 0.4674 | 0.2501 |
| ranked-duels | Full | Random forest | 0.9634 | 0.9633 | 0.8936 | 0.0793 | 0.4674 | 0.2501 |
| ranked-duels | Full | Gradient boosting | 0.9698 | 0.9697 | 0.9023 | 0.0693 | 0.4674 | 0.2501 |
| ranked-duels | Style | Logistic regression * | 0.9183 | 0.9167 | 0.8306 | 0.1167 | 0.4674 | 0.2501 |
| ranked-duels | Style | Random forest | 0.9104 | 0.9085 | 0.8240 | 0.1252 | 0.4674 | 0.2501 |
| ranked-duels | Style | Gradient boosting | 0.9180 | 0.9164 | 0.8298 | 0.1171 | 0.4674 | 0.2501 |
| ranked-team | Full | Logistic regression * | 0.9539 | 0.9525 | 0.8774 | 0.0872 | 0.5048 | 0.2500 |
| ranked-team | Full | Random forest | 0.9380 | 0.9363 | 0.8563 | 0.1051 | 0.5048 | 0.2500 |
| ranked-team | Full | Gradient boosting | 0.9527 | 0.9513 | 0.8754 | 0.0884 | 0.5048 | 0.2500 |
| ranked-team | Style | Logistic regression * | 0.9005 | 0.8986 | 0.8101 | 0.1296 | 0.5048 | 0.2500 |
| ranked-team | Style | Random forest | 0.8892 | 0.8868 | 0.7984 | 0.1384 | 0.5048 | 0.2500 |
| ranked-team | Style | Gradient boosting | 0.8995 | 0.8976 | 0.8094 | 0.1302 | 0.5048 | 0.2500 |
| casual-all | Full | Logistic regression | 0.9507 | 0.9526 | 0.8777 | 0.0885 | 0.4965 | 0.2500 |
| casual-all | Full | Random forest | 0.9399 | 0.9422 | 0.8640 | 0.1004 | 0.4965 | 0.2500 |
| casual-all | Full | Gradient boosting * | 0.9523 | 0.9542 | 0.8799 | 0.0856 | 0.4965 | 0.2500 |
| casual-all | Style | Logistic regression | 0.8846 | 0.8866 | 0.7997 | 0.1374 | 0.4965 | 0.2500 |
| casual-all | Style | Random forest | 0.8846 | 0.8866 | 0.7994 | 0.1390 | 0.4965 | 0.2500 |
| casual-all | Style | Gradient boosting * | 0.8885 | 0.8907 | 0.8034 | 0.1346 | 0.4965 | 0.2500 |
| casual-duels | Full | Logistic regression | 0.9714 | 0.9714 | 0.9075 | 0.0664 | 0.5041 | 0.2500 |
| casual-duels | Full | Random forest | 0.9664 | 0.9666 | 0.8989 | 0.0753 | 0.5041 | 0.2500 |
| casual-duels | Full | Gradient boosting * | 0.9722 | 0.9723 | 0.9075 | 0.0660 | 0.5041 | 0.2500 |
| casual-duels | Style | Logistic regression | 0.9199 | 0.9176 | 0.8345 | 0.1156 | 0.5041 | 0.2500 |
| casual-duels | Style | Random forest | 0.9145 | 0.9130 | 0.8293 | 0.1219 | 0.5041 | 0.2500 |
| casual-duels | Style | Gradient boosting * | 0.9221 | 0.9209 | 0.8355 | 0.1136 | 0.5041 | 0.2500 |
| casual-team | Full | Logistic regression * | 0.9546 | 0.9561 | 0.8821 | 0.0834 | 0.4966 | 0.2500 |
| casual-team | Full | Random forest | 0.9410 | 0.9424 | 0.8654 | 0.0998 | 0.4966 | 0.2500 |
| casual-team | Full | Gradient boosting | 0.9540 | 0.9551 | 0.8805 | 0.0847 | 0.4966 | 0.2500 |
| casual-team | Style | Logistic regression * | 0.8961 | 0.8979 | 0.8110 | 0.1298 | 0.4966 | 0.2500 |
| casual-team | Style | Random forest | 0.8847 | 0.8855 | 0.7993 | 0.1390 | 0.4966 | 0.2500 |
| casual-team | Style | Gradient boosting | 0.8957 | 0.8972 | 0.8108 | 0.1303 | 0.4966 | 0.2500 |
