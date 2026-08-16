# Public release boundary

The public `main` branch is a code-only scientific release. It includes the
pipeline source, portable configuration templates, tests, documentation, and
non-identifying model/dataset checksum contracts.

The following artifacts must remain in approved local or controlled storage:

- raw or derived histopathology images;
- model checkpoints and gated weights;
- H5 features and experiment outputs;
- fold, prediction, and QA files containing pseudonymous patient, slide, ROI,
  or coordinate identifiers;
- credentials, local path configuration, and environment-specific metadata.

Use `configs/paths.example.yaml` as the template for an ignored
`configs/paths.local.yaml`. The default execution hold must only be overridden
after the local dataset, split, and model-identity audits pass.

## Canonical evaluation

The canonical classification protocol is patient-grouped three-fold
cross-validation with five independent seeds. Each seed produces one complete
out-of-fold prediction set; Table 5 reports the mean and sample standard
deviation of the five seed-level metrics. Fold metrics are audit evidence and
must not be substituted for the canonical seed-level estimates.

## Repository history

Replacing the active tree does not erase historical Git objects. Any credential
that ever appeared in repository history must be revoked and rotated. A
separate coordinated history rewrite is required if historical object removal
is necessary.
