"""Command-line interface for the PAIR-BST benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

# PyTorch requires this variable to be present before the first CUDA context is
# initialized when deterministic matrix multiplication is requested.  Every
# benchmark command uses the deterministic protocol; a caller may still select
# the other PyTorch-supported value (":16:8") before launching this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .pipeline import (
    PipelineContext,
    build_manifest,
    build_report,
    build_splits,
    extract_features,
    run_feature_pilot,
    run_audit,
    run_classification,
    run_retrieval,
    run_statistics,
    verify_images,
    verify_release_integrity,
    verify_models,
)


DEFAULT_PATHS = "configs/paths.local.yaml"
DEFAULT_PROTOCOL = "configs/protocol_cv3_independent_seed_oof_v1.yaml"
DEFAULT_MODELS = "configs/models.yaml"
DEFAULT_COMPARISONS = "configs/comparisons.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _add_context_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_PATHS, help="Local paths YAML")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, help="Frozen protocol YAML")
    parser.add_argument("--models-config", default=DEFAULT_MODELS, help="Model registry YAML")
    parser.add_argument(
        "--comparisons-config", default=DEFAULT_COMPARISONS, help="Predeclared comparisons YAML"
    )


def _add_dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without doing work")


def _add_experiment_options(parser: argparse.ArgumentParser) -> None:
    _add_dry_run(parser)
    parser.add_argument(
        "--override-hold",
        action="store_true",
        help="Run despite EXECUTION_HOLD.json; use only after owner approval",
    )


def _add_model_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model ID/alias; repeat, comma-separate, or use 'all' (default)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pairbst", description="PAIR-BST revision benchmark")
    groups = parser.add_subparsers(dest="group", required=True)

    manifest = groups.add_parser("manifest", help="Dataset manifest operations")
    manifest_actions = manifest.add_subparsers(dest="action", required=True)
    manifest_build = manifest_actions.add_parser("build", help="Build frozen ROI manifest")
    _add_context_options(manifest_build)
    _add_dry_run(manifest_build)
    manifest_build.add_argument("--verify-dimensions", action="store_true")

    splits = groups.add_parser("splits", help="Patient-grouped split operations")
    split_actions = splits.add_subparsers(dest="action", required=True)
    split_build = split_actions.add_parser("build", help="Build deterministic CV3 split")
    _add_context_options(split_build)
    _add_dry_run(split_build)
    split_build.add_argument("--no-optimize-balance", action="store_true")
    split_build.add_argument("--max-balance-passes", type=int, default=30)

    images = groups.add_parser("images", help="Image integrity operations")
    image_actions = images.add_subparsers(dest="action", required=True)
    image_verify = image_actions.add_parser("verify", help="Decode ROIs and compare centers")
    _add_context_options(image_verify)
    _add_dry_run(image_verify)
    image_verify.add_argument("--legacy-center-root")
    image_verify.add_argument("--workers", type=int, default=1)
    image_verify.add_argument("--output-dir")
    release_verify = image_actions.add_parser(
        "verify-release", help="Hash every released file against the Figshare manifest"
    )
    _add_context_options(release_verify)
    _add_dry_run(release_verify)
    release_verify.add_argument("--workers", type=int, default=4)

    models = groups.add_parser("models", help="Checkpoint operations")
    model_actions = models.add_subparsers(dest="action", required=True)
    model_verify = model_actions.add_parser("verify", help="Verify checkpoint size and SHA-256")
    _add_context_options(model_verify)
    _add_dry_run(model_verify)
    model_verify.add_argument("--size-only", action="store_true", help="Skip SHA-256 calculation")
    model_verify.add_argument("--output")

    audit = groups.add_parser("audit", help="Combined preparation audit")
    audit_actions = audit.add_subparsers(dest="action", required=True)
    audit_run = audit_actions.add_parser("run", help="Run dataset/model/split/recovery audit")
    _add_context_options(audit_run)
    _add_dry_run(audit_run)
    audit_run.add_argument("--decode-all", action="store_true")
    audit_run.add_argument("--hash-models", action="store_true")
    audit_run.add_argument("--output")

    features = groups.add_parser("features", help="Feature extraction")
    feature_actions = features.add_subparsers(dest="action", required=True)
    feature_extract = feature_actions.add_parser("extract", help="Extract center/mean/max features")
    _add_context_options(feature_extract)
    _add_experiment_options(feature_extract)
    _add_model_selector(feature_extract)
    feature_extract.add_argument("--profile")
    feature_extract.add_argument("--device", default="cuda")
    feature_extract.add_argument("--batch-size", type=int)
    feature_extract.add_argument("--autocast-dtype", choices=("float16", "bfloat16"))
    feature_extract.add_argument("--output-dir")
    feature_pilot = feature_actions.add_parser(
        "pilot", help="Repeat a tiny extraction and verify deterministic/resume behavior"
    )
    _add_context_options(feature_pilot)
    _add_experiment_options(feature_pilot)
    _add_model_selector(feature_pilot)
    feature_pilot.add_argument("--profile")
    feature_pilot.add_argument("--device", default="cuda")
    feature_pilot.add_argument("--batch-size", type=int)
    feature_pilot.add_argument("--autocast-dtype", choices=("float16", "bfloat16"))
    feature_pilot.add_argument("--roi-count", type=int, default=2)
    feature_pilot.add_argument("--atol", type=float, default=0.0)
    feature_pilot.add_argument("--rtol", type=float, default=0.0)
    feature_pilot.add_argument("--output-dir")
    feature_pilot.add_argument("--summary-json")

    classify = groups.add_parser("classify", help="Linear probing")
    classify_actions = classify.add_subparsers(dest="action", required=True)
    classify_run = classify_actions.add_parser("run", help="Run all three-fold linear probes")
    _add_context_options(classify_run)
    _add_experiment_options(classify_run)
    _add_model_selector(classify_run)
    classify_run.add_argument("--profile")
    classify_run.add_argument("--device", default="cuda")
    classify_run.add_argument("--features-dir")
    classify_run.add_argument("--output-dir")

    retrieval = groups.add_parser("retrieval", help="Patient-disjoint retrieval")
    retrieval_actions = retrieval.add_subparsers(dest="action", required=True)
    retrieval_run = retrieval_actions.add_parser("run", help="Run exact-cosine CV retrieval")
    _add_context_options(retrieval_run)
    _add_experiment_options(retrieval_run)
    _add_model_selector(retrieval_run)
    retrieval_run.add_argument("--profile")
    retrieval_run.add_argument("--features-dir")
    retrieval_run.add_argument("--output-dir")
    retrieval_run.add_argument("--query-chunk-size", type=int, default=256)

    statistics = groups.add_parser("statistics", help="Uncertainty and paired tests")
    statistics_actions = statistics.add_subparsers(dest="action", required=True)
    statistics_run = statistics_actions.add_parser("run", help="Run patient-cluster bootstrap")
    _add_context_options(statistics_run)
    _add_experiment_options(statistics_run)
    statistics_run.add_argument("--profile")
    statistics_run.add_argument("--classification-dir")
    statistics_run.add_argument("--retrieval-dir")
    statistics_run.add_argument("--output-dir")
    statistics_run.add_argument("--bootstrap-iterations", type=int)
    statistics_run.add_argument(
        "--retrieval-ci-metric",
        action="append",
        dest="retrieval_ci_metrics",
        help="Query metric to bootstrap; repeat to select more",
    )

    report = groups.add_parser("report", help="Manuscript-ready result tables")
    report_actions = report.add_subparsers(dest="action", required=True)
    report_build = report_actions.add_parser("build", help="Build final CSV/Markdown/LaTeX bundle")
    _add_context_options(report_build)
    _add_experiment_options(report_build)
    report_build.add_argument("--profile")
    report_build.add_argument("--classification-dir")
    report_build.add_argument("--retrieval-dir")
    report_build.add_argument("--statistics-dir")
    report_build.add_argument("--output-dir")
    return parser


def _context(args: argparse.Namespace) -> PipelineContext:
    return PipelineContext.load(
        args.config,
        args.protocol,
        args.models_config,
        args.comparisons_config,
    )


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    context = _context(args)
    key = (args.group, args.action)
    if key == ("manifest", "build"):
        return build_manifest(
            context, verify_dimensions=args.verify_dimensions, dry_run=args.dry_run
        )
    if key == ("splits", "build"):
        return build_splits(
            context,
            optimize_balance=not args.no_optimize_balance,
            max_balance_passes=args.max_balance_passes,
            dry_run=args.dry_run,
        )
    if key == ("images", "verify"):
        return verify_images(
            context,
            legacy_center_root=args.legacy_center_root,
            workers=args.workers,
            output_directory=args.output_dir,
            dry_run=args.dry_run,
        )
    if key == ("images", "verify-release"):
        return verify_release_integrity(
            context,
            workers=args.workers,
            dry_run=args.dry_run,
        )
    if key == ("models", "verify"):
        return verify_models(
            context,
            hash_contents=not args.size_only,
            output_json=args.output,
            dry_run=args.dry_run,
        )
    if key == ("audit", "run"):
        return run_audit(
            context,
            decode_all=args.decode_all,
            hash_models=args.hash_models,
            output_json=args.output,
            dry_run=args.dry_run,
        )
    if key == ("features", "extract"):
        return extract_features(
            context,
            models=args.models,
            profile=args.profile,
            device=args.device,
            batch_size=args.batch_size,
            autocast_dtype=args.autocast_dtype,
            output_directory=args.output_dir,
            override_hold=args.override_hold,
            dry_run=args.dry_run,
        )
    if key == ("features", "pilot"):
        return run_feature_pilot(
            context,
            models=args.models,
            profile=args.profile,
            device=args.device,
            batch_size=args.batch_size,
            autocast_dtype=args.autocast_dtype,
            roi_count=args.roi_count,
            atol=args.atol,
            rtol=args.rtol,
            output_directory=args.output_dir,
            summary_json=args.summary_json,
            override_hold=args.override_hold,
            dry_run=args.dry_run,
        )
    if key == ("classify", "run"):
        return run_classification(
            context,
            models=args.models,
            profile=args.profile,
            device=args.device,
            feature_directory_path=args.features_dir,
            output_directory=args.output_dir,
            override_hold=args.override_hold,
            dry_run=args.dry_run,
        )
    if key == ("retrieval", "run"):
        return run_retrieval(
            context,
            models=args.models,
            profile=args.profile,
            feature_directory_path=args.features_dir,
            output_directory=args.output_dir,
            query_chunk_size=args.query_chunk_size,
            override_hold=args.override_hold,
            dry_run=args.dry_run,
        )
    if key == ("statistics", "run"):
        metrics = args.retrieval_ci_metrics or (
            "average_precision_at_k", "majority_vote_correct"
        )
        return run_statistics(
            context,
            profile=args.profile,
            classification_directory=args.classification_dir,
            retrieval_directory=args.retrieval_dir,
            output_directory=args.output_dir,
            n_bootstrap=args.bootstrap_iterations,
            retrieval_ci_metrics=metrics,
            override_hold=args.override_hold,
            dry_run=args.dry_run,
        )
    if key == ("report", "build"):
        return build_report(
            context,
            profile=args.profile,
            classification_directory=args.classification_dir,
            retrieval_directory=args.retrieval_dir,
            statistics_directory=args.statistics_dir,
            output_directory=args.output_dir,
            override_hold=args.override_hold,
            dry_run=args.dry_run,
        )
    raise AssertionError(f"Unhandled command: {key}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
