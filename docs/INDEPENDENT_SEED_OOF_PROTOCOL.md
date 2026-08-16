# PAIR-BST independent-seed complete-OOF classification protocol

Protocol ID: `cv3_independent_seed_oof_v1`

## Scope

This protocol is the canonical classification evaluation for the PAIR-BST
seven-model benchmark. It applies to every combination of seven feature
extractors, three ROI representations, and three classification tasks. The
retrieval protocol is unchanged.

The frozen patient-disjoint folds and outcome labels are not modified. The
operational patient identifier remains the curator-certified composite of
diagnosis and `patient_idx`. Validation against a future globally unique
patient identifier remains an external data-team dependency.

## Estimator

The fixed linear-probe seeds are `101`, `202`, `303`, `404`, and `505`. For
each seed, one independently trained head is evaluated on each of the three
held-out folds. The three held-out subsets are scattered back to their frozen
ROI positions and concatenated into one complete 2,252-ROI OOF prediction set
for that seed.

Balanced accuracy, macro-F1, accuracy, and weighted-F1 are calculated once
from each complete seed-specific OOF prediction set. The primary result is the
arithmetic mean and sample standard deviation across the five resulting
metric values. Sample standard deviation uses `ddof=1`.

The canonical path does not average probabilities or logits across seeds. It
does not use hard voting. It does not treat the 15 seed-and-fold metrics as
independent primary observations. Fold metrics are retained only as audit
evidence.

## Reconstruction source

The canonical v4 results are reconstructed from the 189 cryptographically
verified `seed_and_mean_probabilities.npz` files in the completed live run.
Only `seed_probabilities`, `test_indices`, and the verified seed order are used
for canonical predictions. Stored `mean_probabilities` are legacy audit data
and cannot feed a canonical output.

For every system, reconstruction verifies class order, fold assignment,
metadata order, probability dimensions, finite values, mutually exclusive
held-out indices, complete coverage, and one prediction per ROI per seed.
No linear head is retrained, no frozen feature is re-extracted, and retrieval
is not rerun.

## Canonical outputs

- `classification_seed_oof_metrics.csv`: 315 rows
- `classification_seed_fold_metrics.csv`: 945 audit rows
- `classification_seed_oof_predictions.csv.gz`: 709,380 rows
- `classification_seed_oof_probabilities/`: 63 compressed system files
- `classification_per_class_seed_oof.csv`: 5,250 rows
- `classification_per_class_seed_summary.csv`: 1,050 rows
- seed-specific and seed-summary confusion-matrix files
- `classification_patient_cluster_ci_by_seed.csv`: seed-specific supplementary intervals
- Table 5 generated only from the 315-row seed OOF metric file

Patient-cluster confidence intervals are calculated separately for each seed
and are not collapsed into one artificial interval. No canonical paired
significance claim is generated from five seeds. The previous ensemble-based
paired analysis is retained only as clearly labelled legacy evidence.

## Version separation

Previous v2 and v3 packages are immutable legacy and audit records. Their
classification estimator averaged probabilities within each fold and then
reported variability across folds. Files from those packages cannot be used
as canonical inputs for this protocol. Protocol IDs, estimator fields, source
hashes, output hashes, row-count checks, and retrieval hashes must all match
before a v4 result can pass validation.

