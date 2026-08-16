# PAIR-BST benchmark

This repository contains the cleaned, reproducible PAIR-BST benchmark and
revision pipeline. Obsolete root-level MuCoSA scripts were removed from the
active `main` tree; their history remains available through Git.

The public PAIR-BST data collection is hosted on
[Figshare](https://plus.figshare.com/collections/PAIR-BST_A_region-level_histopathology_dataset_for_rare_bone_and_soft_tissue_tumors/8223469).

## Public release boundary

This public repository contains source code, portable configuration templates,
tests, documentation, and non-identifying checksum contracts only. It excludes
raw/derived image data, model weights, H5 features, run outputs, credentials,
machine-local paths, and controlled fold manifests containing pseudonymous
patient or ROI identifiers. Copy `configs/paths.example.yaml` to the ignored
`configs/paths.local.yaml`, then configure only approved local artifacts.

## Protocols kept separate

- `legacy_50_10_40`: the recovered historical split (133/27/108 patients;
  1,161/199/892 ROIs), retained only to trace the submitted Table 5.
- `PAIRBST-REV-CV3-v1`: the completed legacy seed-probability-ensemble
  evaluation. It is retained only as immutable audit provenance.
- `cv3_independent_seed_oof_v1`: the canonical patient-grouped three-fold
  classification protocol. Each seed has one complete OOF evaluation and the
  primary result is the mean and sample SD across five seed-level metrics.

The curator-certified operational patient identifier is the pair
`(diagnosis, patient_idx)`. All slides and ROIs belonging to that identifier
must remain in the same fold.

## Current canonical execution status

The retained internal seven-model run used 189 held-out probability archives
to reconstruct the canonical independent-seed OOF results without retraining
linear heads, re-extracting frozen features, or rerunning retrieval. Its
canonical v4 validation status is `PASS`; controlled results are intentionally
not distributed in this repository.

The completed grid contains 7 models x 3 ROI strategies x 3 tasks, giving 63
systems. Canonical outputs contain 315 complete seed-specific OOF metrics,
945 seed-and-fold audit metrics, 709,380 seed-specific ROI predictions, 5,250
seed-level per-class rows, and 1,050 per-class summary rows. Table 5 has 21
model-and-strategy rows and consumes only the 315-row seed OOF metric file.

## Setup

The known compatible environment is:

```powershell
python -m pip install -r requirements.mucosa-cu128.lock.txt
python -m pip install -e .
```

The portable dependency snapshot is recorded in
`requirements.mucosa-cu128.lock.txt`. Machine-local environment paths are not
part of the portable code package.

Commands can also be run without installation:

```powershell
$env:PYTHONPATH = "src"
python -m pairbst.cli --help
```

## Preparation commands (safe before approval)

```powershell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
Copy-Item locks/EXECUTION_HOLD.example.json locks/EXECUTION_HOLD.json
pairbst manifest build --config configs/paths.local.yaml
pairbst splits build --config configs/paths.local.yaml --protocol configs/protocol_cv3_independent_seed_oof_v1.yaml
pairbst images verify-release --workers 4 --config configs/paths.local.yaml
pairbst audit run --hash-models --config configs/paths.local.yaml --protocol configs/protocol_cv3_independent_seed_oof_v1.yaml
```

`images verify-release` checks all 2,252 PNGs plus metadata against the
authoritative Figshare byte sizes and MD5 values. The audit re-hashes those
2,253 current files at time of use. Missing or hash-mismatched data,
checkpoints, folds, or class-coverage artifacts are fatal for feature
extraction.

Every result stage recursively binds its exact upstream outputs, H5 model and
transform identity, configs, frozen split, environment lock, and pipeline
source files by SHA-256. Cross-run mixing is rejected before statistics or
report generation.

Model checkpoints are not redistributed. Obtain ResNet and Swin-T through
their framework distributions, and obtain RetCCL, UNI, UNI2-H, Prov-GigaPath,
and Virchow2 only from their official upstream sources under the applicable
access and license terms. The checksums in `locks/models.expected.json` bind
the exact files used by the validated run.

## Canonical reconstruction command

The validated v4 result was generated in the retained source workspace by the
Path-A command below. The portable result package intentionally omits the 189
fold-level source probability NPZ files and the full legacy v2 package. A new
reconstruction therefore requires explicit paths to those retained external
artifacts.

```powershell
python scripts/reconstruct_independent_seed_oof.py `
  --classification-root <RETAINED_CLASSIFICATION_RESULT_DIRECTORY> `
  --folds <RETAINED_APPROVED_FOLD_MANIFEST> `
  --config configs/protocol_cv3_independent_seed_oof_v1.yaml `
  --prior-package <RETAINED_VALIDATED_V2_DIRECTORY> `
  --output <NEW_V4_OUTPUT_DIRECTORY>
```

This command reads the immutable seed probability archives and never trains a
model or executes retrieval. The v4 package preserves the reconstructed
probabilities and complete provenance, but it is not a duplicate of all 189
fold-level source archives.

Inside an extracted internal portable v4 package, run the packaged validator from the
`CODE` directory. It detects the package parent automatically.

```powershell
python scripts/validate_7model_results.py
```

## Full experiment commands

The commands below are retained for a future clean from-features execution.
They use the independent-seed protocol by default. They were not needed for
the v4 Path-A reconstruction.

```powershell
pairbst features pilot --model all --override-hold
pairbst features extract --model all --output-dir outputs/runs/features/official_model_specific_7model_v1 --override-hold
pairbst classify run --model all --features-dir outputs/runs/features/official_model_specific_7model_v1 --output-dir outputs/runs/classification/official_model_specific_7model_v1 --override-hold
pairbst retrieval run --model all --features-dir outputs/runs/features/official_model_specific_7model_v1 --output-dir outputs/runs/retrieval/official_model_specific_7model_v1 --override-hold
pairbst statistics run --classification-dir outputs/runs/classification/official_model_specific_7model_v1 --retrieval-dir outputs/runs/retrieval/official_model_specific_7model_v1 --output-dir outputs/runs/statistics/official_model_specific_7model_v1 --override-hold
pairbst report build --classification-dir outputs/runs/classification/official_model_specific_7model_v1 --retrieval-dir outputs/runs/retrieval/official_model_specific_7model_v1 --statistics-dir outputs/runs/statistics/official_model_specific_7model_v1 --output-dir outputs/final_7model_v1 --override-hold
```

The reporting stage writes a final-only manuscript bundle in CSV, Markdown,
and LaTeX, plus machine-readable audit artifacts. The main CSV follows
submitted Table 5 exactly:

`Model | Strategy | Diagnosis B.Acc | Diagnosis Macro-F1 | Differentiation B.Acc | Differentiation Macro-F1 | Growth Pattern B.Acc | Growth Pattern Macro-F1`

Each canonical classification result is rendered to three decimals as
`mean ± SD` across five complete seed-specific OOF metrics. Fold metrics are
secondary audit evidence and cannot feed the canonical Table 5 builder.

Classification patient-cluster confidence intervals are calculated separately
for each seed and remain supplementary. They are not collapsed into one
artificial interval and are not used in Table 5. No new canonical paired
significance claim is made from five seeds. Retrieval inference remains the
unchanged, previously validated analysis.
