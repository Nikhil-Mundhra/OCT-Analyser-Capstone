# OS orientation and fusion ablation

This controlled experiment used the same checkpoint and the same six volumes
(BEH0086, BEH0314, and BEH0335; OD and OS) for every variant. The three subjects
were selected for diagnosis and include difficult cases; this is not a new
population-level performance estimate.

| Variant | Eye | n | Median Dice | Median MABE (µm) | Median P95 (µm) | Median cup IoU |
|---|---:|---:|---:|---:|---:|---:|
| Corrected biplanar | OD | 3 | 0.8914 | 5.05 | 17.75 | 0.9082 |
| Legacy biplanar | OD | 3 | 0.8914 | 5.05 | 17.75 | 0.9082 |
| Corrected biplanar | OS | 3 | 0.8723 | 5.43 | 18.66 | 0.9047 |
| Legacy biplanar | OS | 3 | 0.6409 | 123.62 | 175.95 | 0.7363 |
| Corrected horizontal only | OS | 3 | 0.8703 | 6.18 | 20.79 | 0.9129 |
| Native-orientation biplanar | OS | 3 | 0.8705 | 10.11 | 36.16 | 0.8968 |

The exact equality of corrected and legacy OD results is the negative control:
the changed coordinate mapping is OS-specific. Correcting the OS vertical write
index removes the catastrophic error and improves OS MABE relative to both the
legacy path and horizontal-only inference. It does not make every scan pass:
BEH0335 remains a serious failure (18.80 µm MABE), and all six corrected scans
were routed to manual review by at least one operational engineering check.

See `orientation_ablation.csv` for scan-level metrics and
`orientation_ablation.json` for metrics, QC flags, diagnostics, and variant
metadata. QC thresholds are engineering defaults and are not clinically
validated.
