\# Model Evaluation — Week 2 Day 4



\## Default threshold (0.5)

\- Precision: 0.854

\- Recall: 0.837

\- PR-AUC: 0.868

\- Fraud caught: 82/98

\- False alarms: 14 (out of 56,864 normal transactions)



\## Threshold comparison

| Threshold | Precision | Recall | Fraud Caught | False Alarms |

|---|---|---|---|---|

| 0.5 | 0.854 | 0.837 | 82/98 | 14 |

| 0.3 | 0.774 | 0.837 | 82/98 | 24 |

| 0.2 | 0.741 | 0.847 | 83/98 | 29 |



\## Conclusion

Lowering the threshold below 0.5 does not meaningfully improve fraud 

detection - recall stays nearly flat while false positives increase 

significantly. Kept default threshold (0.5) as the better tradeoff.

