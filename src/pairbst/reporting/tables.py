"""Format final tables for direct manuscript transfer.

The main output deliberately mirrors the existing Table 5 structure and is
kept separate from raw predictions and audit artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


MODEL_ORDER = (
    "ResNet",
    "Swin-T",
    "RetCCL",
    "UNI",
    "UNI-2",
    "Prov-GigaPath",
    "Virchow2",
)
STRATEGY_ORDER = ("Center Crop", "Mean Pooling", "Max Pooling")
TASK_ORDER = ("diagnosis", "differentiation", "growth_pattern")
CANONICAL_CLASSIFICATION_PROTOCOL_ID = "cv3_independent_seed_oof_v1"
CANONICAL_SEEDS = (101, 202, 303, 404, 505)
TABLE5_NOTE = (
    "Values are the mean and sample standard deviation across five independently "
    "trained linear probes. Each seed was evaluated using complete patient-disjoint "
    "three-fold out-of-fold predictions."
)
MANUSCRIPT_TABLE_COLUMNS = (
    "Model",
    "Strategy",
    "Diagnosis B.Acc",
    "Diagnosis Macro-F1",
    "Differentiation B.Acc",
    "Differentiation Macro-F1",
    "Growth Pattern B.Acc",
    "Growth Pattern Macro-F1",
)

_MODEL_ALIASES = {
    "resnet": "ResNet",
    "resnet50": "ResNet",
    "resnet-50": "ResNet",
    "resnet50_v2": "ResNet",
    "swin": "Swin-T",
    "swin-t": "Swin-T",
    "swin_t": "Swin-T",
    "retccl": "RetCCL",
    "uni": "UNI",
    "uni-2": "UNI-2",
    "uni2": "UNI-2",
    "uni2-h": "UNI-2",
    "uni2_h": "UNI-2",
    "prov-gigapath": "Prov-GigaPath",
    "prov_gigapath": "Prov-GigaPath",
    "gigapath": "Prov-GigaPath",
    "virchow2": "Virchow2",
    "virchow-2": "Virchow2",
    "virchow_v2": "Virchow2",
}
_STRATEGY_ALIASES = {
    "center": "Center Crop",
    "center crop": "Center Crop",
    "center_crop": "Center Crop",
    "mean": "Mean Pooling",
    "mean pooling": "Mean Pooling",
    "mean_pooling": "Mean Pooling",
    "max": "Max Pooling",
    "max pooling": "Max Pooling",
    "max_pooling": "Max Pooling",
}
_TASK_ALIASES = {
    "diagnosis": "diagnosis",
    "dx": "diagnosis",
    "differentiation": "differentiation",
    "differentiation degree": "differentiation",
    "growth": "growth_pattern",
    "growth pattern": "growth_pattern",
    "growth_pattern": "growth_pattern",
}
_CELL_MAP = {
    ("diagnosis", "balanced_accuracy"): "Diagnosis B.Acc",
    ("diagnosis", "macro_f1"): "Diagnosis Macro-F1",
    ("differentiation", "balanced_accuracy"): "Differentiation B.Acc",
    ("differentiation", "macro_f1"): "Differentiation Macro-F1",
    ("growth_pattern", "balanced_accuracy"): "Growth Pattern B.Acc",
    ("growth_pattern", "macro_f1"): "Growth Pattern Macro-F1",
}


def _canonical(value: Any, aliases: Mapping[str, str]) -> str:
    text = str(value).strip()
    return aliases.get(text.casefold(), text)


def format_mean_sd(mean: float, sd: float, *, decimals: int = 3) -> str:
    """Format a manuscript cell with a true plus/minus character."""

    if not np.isfinite(mean) or not np.isfinite(sd):
        raise ValueError("Mean and SD must be finite.")
    return f"{mean:.{decimals}f} \N{PLUS-MINUS SIGN} {sd:.{decimals}f}"


def _prepare_classification_frame(seed_oof_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "protocol_id",
        "model_id",
        "model",
        "strategy",
        "task",
        "seed",
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "weighted_f1",
        "n_samples",
        "n_patients",
        "n_classes",
    }
    missing = sorted(required - set(seed_oof_metrics.columns))
    if missing:
        raise ValueError(
            f"Classification seed-specific OOF metrics are missing columns: {missing}"
        )
    if "held_fold" in seed_oof_metrics.columns:
        raise ValueError(
            "Legacy fold metrics cannot be consumed by the canonical Table 5 builder."
        )
    frame = seed_oof_metrics.copy()
    protocol_ids = set(frame["protocol_id"].astype(str))
    if protocol_ids != {CANONICAL_CLASSIFICATION_PROTOCOL_ID}:
        raise ValueError(
            "Table 5 requires only canonical independent-seed OOF metrics with "
            f"protocol_id={CANONICAL_CLASSIFICATION_PROTOCOL_ID!r}; got "
            f"{sorted(protocol_ids)}."
        )
    frame["model"] = frame["model"].map(lambda value: _canonical(value, _MODEL_ALIASES))
    frame["strategy"] = frame["strategy"].map(
        lambda value: _canonical(value, _STRATEGY_ALIASES)
    )
    frame["task"] = frame["task"].map(lambda value: _canonical(value, _TASK_ALIASES))
    for column in ("balanced_accuracy", "macro_f1", "accuracy", "weighted_f1"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all() or not frame[column].between(0, 1).all():
            raise ValueError(f"{column} must be finite and lie in [0, 1].")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    for column in ("n_samples", "n_patients", "n_classes"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if (frame[column] <= 0).any():
            raise ValueError(f"{column} must be positive.")
    return frame


def build_legacy_seed_probability_ensemble_table(
    fold_metrics: pd.DataFrame,
    *,
    model_order: Sequence[str] = MODEL_ORDER,
    strategy_order: Sequence[str] = STRATEGY_ORDER,
    expected_folds: int = 3,
    expected_fold_ids: Sequence[int] = (0, 1, 2),
    require_complete: bool = True,
    enabled: bool = False,
) -> pd.DataFrame:
    """Reject access to the retired seed-probability ensemble table path."""

    del fold_metrics, model_order, strategy_order
    del expected_folds, expected_fold_ids, require_complete, enabled
    raise RuntimeError(
        "Legacy seed-probability ensemble reporting is not available from the "
        "canonical reporting path."
    )

def build_manuscript_table(
    seed_oof_metrics: pd.DataFrame,
    *,
    model_order: Sequence[str] = MODEL_ORDER,
    strategy_order: Sequence[str] = STRATEGY_ORDER,
    expected_seeds: Sequence[int] = CANONICAL_SEEDS,
    seed_sd_ddof: int = 1,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Build Table 5 from complete seed-specific OOF metrics only.

    Input is one row per model, strategy, task, and independent seed. The five
    complete-OOF metrics are summarized with arithmetic mean and sample SD.
    Fold metrics and seed-ensemble results are rejected by schema and protocol.
    """

    expected_seed_values = tuple(int(value) for value in expected_seeds)
    required_seeds = set(expected_seed_values)
    if expected_seed_values != CANONICAL_SEEDS:
        raise ValueError(
            f"Canonical Table 5 seed order must be exactly {CANONICAL_SEEDS}."
        )
    if seed_sd_ddof != 1:
        raise ValueError("Canonical Table 5 requires sample SD with ddof=1.")

    frame = _prepare_classification_frame(seed_oof_metrics)
    duplicate_keys = ["model", "strategy", "task", "seed"]
    if frame.duplicated(duplicate_keys).any():
        examples = frame.loc[
            frame.duplicated(duplicate_keys, keep=False), duplicate_keys
        ]
        raise ValueError(
            f"Duplicate seed-specific OOF metric rows detected:\n{examples.head()}"
        )

    lookup: dict[tuple[str, str, str, str], str] = {}
    mean_lookup: dict[tuple[str, str, str, str], float] = {}
    for (model, strategy, task), group in frame.groupby(
        ["model", "strategy", "task"], sort=False
    ):
        observed_seeds = set(group["seed"].astype(int))
        if observed_seeds != required_seeds:
            raise ValueError(
                f"{model}/{strategy}/{task} has seeds {sorted(observed_seeds)}; "
                f"expected {sorted(required_seeds)}."
            )
        for column in ("n_samples", "n_patients", "n_classes"):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{model}/{strategy}/{task} has inconsistent {column} across seeds."
                )
        for metric in ("balanced_accuracy", "macro_f1"):
            values = group[metric].to_numpy(dtype=float)
            mean = float(values.mean())
            lookup[(model, strategy, task, metric)] = format_mean_sd(
                mean, float(values.std(ddof=seed_sd_ddof))
            )
            mean_lookup[(model, strategy, task, metric)] = mean

    rows: list[dict[str, Any]] = []
    missing_cells: list[str] = []
    for model in model_order:
        canonical_model = _canonical(model, _MODEL_ALIASES)
        for strategy in strategy_order:
            canonical_strategy = _canonical(strategy, _STRATEGY_ALIASES)
            row: dict[str, Any] = {
                "Model": canonical_model,
                "Strategy": canonical_strategy,
            }
            for (task, metric), column in _CELL_MAP.items():
                key = (canonical_model, canonical_strategy, task, metric)
                if key not in lookup:
                    missing_cells.append("/".join(key))
                    row[column] = ""
                else:
                    row[column] = lookup[key]
            rows.append(row)

    if require_complete and missing_cells:
        preview = ", ".join(missing_cells[:8])
        suffix = " ..." if len(missing_cells) > 8 else ""
        raise ValueError(f"Final manuscript table is incomplete: {preview}{suffix}")
    if require_complete:
        expected_systems = {
            (
                _canonical(model, _MODEL_ALIASES),
                _canonical(strategy, _STRATEGY_ALIASES),
                task,
            )
            for model in model_order
            for strategy in strategy_order
            for task in TASK_ORDER
        }
        observed_systems = set(
            frame[["model", "strategy", "task"]].itertuples(index=False, name=None)
        )
        missing_systems = expected_systems - observed_systems
        extras = observed_systems - expected_systems
        if missing_systems or extras:
            raise ValueError(
                "Final manuscript seed grid is incomplete or unexpected; "
                f"missing={sorted(missing_systems)[:5]}, extra={sorted(extras)[:5]}."
            )

    table = pd.DataFrame(rows, columns=MANUSCRIPT_TABLE_COLUMNS)
    best_cells: set[tuple[int, str]] = set()
    for (task, metric), column in _CELL_MAP.items():
        candidates = {
            key[:2]: value
            for key, value in mean_lookup.items()
            if key[2] == task and key[3] == metric
        }
        if not candidates:
            continue
        best_mean = max(candidates.values())
        best_systems = {
            system for system, value in candidates.items() if value == best_mean
        }
        for row_index, row in table.iterrows():
            if (row["Model"], row["Strategy"]) in best_systems:
                best_cells.add((int(row_index), column))
    table.attrs["best_cells"] = best_cells
    table.attrs["note"] = TABLE5_NOTE
    table.attrs["protocol_id"] = CANONICAL_CLASSIFICATION_PROTOCOL_ID
    table.attrs["seed_sd_ddof"] = seed_sd_ddof
    return table


def build_classification_ci_table(
    ci_metrics: pd.DataFrame,
    *,
    confidence_label: str = "95% CI",
    model_order: Sequence[str] = MODEL_ORDER,
    strategy_order: Sequence[str] = STRATEGY_ORDER,
    task_order: Sequence[str] = TASK_ORDER,
    metric_order: Sequence[str] = ("balanced_accuracy", "macro_f1"),
    require_complete: bool = True,
) -> pd.DataFrame:
    """Format pooled OOF estimates and patient-cluster confidence intervals."""

    required = {"model", "strategy", "task", "metric", "estimate", "ci_low", "ci_high"}
    missing = sorted(required - set(ci_metrics.columns))
    if missing:
        raise ValueError(f"CI metrics are missing columns: {missing}")
    frame = ci_metrics.copy()
    frame["model"] = frame["model"].map(lambda value: _canonical(value, _MODEL_ALIASES))
    frame["strategy"] = frame["strategy"].map(
        lambda value: _canonical(value, _STRATEGY_ALIASES)
    )
    frame["task"] = frame["task"].map(lambda value: _canonical(value, _TASK_ALIASES))
    selected_metrics = tuple(str(metric) for metric in metric_order)
    frame = frame[frame["metric"].astype(str).isin(selected_metrics)].copy()
    keys = ["model", "strategy", "task", "metric"]
    if frame.duplicated(keys).any():
        raise ValueError("CI metrics contain duplicate model/strategy/task/metric rows")
    expected_keys = [
        (
            _canonical(model, _MODEL_ALIASES),
            _canonical(strategy, _STRATEGY_ALIASES),
            _canonical(task, _TASK_ALIASES),
            metric,
        )
        for model in model_order
        for strategy in strategy_order
        for task in task_order
        for metric in selected_metrics
    ]
    observed_keys = set(frame[keys].itertuples(index=False, name=None))
    if require_complete and observed_keys != set(expected_keys):
        missing_keys = set(expected_keys) - observed_keys
        extra_keys = observed_keys - set(expected_keys)
        raise ValueError(
            "Classification CI grid is incomplete or unexpected; "
            f"missing={sorted(missing_keys)[:5]}, extra={sorted(extra_keys)[:5]}"
        )
    lookup = {tuple(record[key] for key in keys): record for record in frame.to_dict("records")}
    iteration_keys = expected_keys if require_complete else list(lookup)
    rows: list[dict[str, Any]] = []
    for key in iteration_keys:
        record = lookup[key]
        estimate, low, high = (
            float(record["estimate"]),
            float(record["ci_low"]),
            float(record["ci_high"]),
        )
        if not np.isfinite([estimate, low, high]).all():
            raise ValueError("CI metrics contain a non-finite value.")
        if not (0.0 <= estimate <= 1.0 and 0.0 <= low <= high <= 1.0):
            raise ValueError(
                f"Invalid bounded classification interval for {key}: "
                f"low={low}, estimate={estimate}, high={high}"
            )
        rows.append(
            {
                "Model": _canonical(record["model"], _MODEL_ALIASES),
                "Strategy": _canonical(record["strategy"], _STRATEGY_ALIASES),
                "Task": _canonical(record["task"], _TASK_ALIASES),
                "Metric": str(record["metric"]),
                "Estimate": f"{estimate:.3f}",
                confidence_label: f"{low:.3f}–{high:.3f}",
                "Bootstrap iterations": int(record.get("n_bootstrap", 10_000)),
            }
        )
    return pd.DataFrame(rows)


def build_retrieval_companion_table(
    retrieval_fold_metrics: pd.DataFrame,
    *,
    model_order: Sequence[str] = MODEL_ORDER,
    strategy_order: Sequence[str] = STRATEGY_ORDER,
    task_order: Sequence[str] = TASK_ORDER,
    k_order: Sequence[int] = (5, 10, 15, 20),
    expected_fold_ids: Sequence[int] = (0, 1, 2),
    require_complete: bool = True,
) -> pd.DataFrame:
    """Create a compact mean ± SD retrieval table across the three folds."""

    required = {
        "model",
        "strategy",
        "task",
        "held_fold",
        "k",
        "precision_at_k",
        "recall_at_k",
        "hit_at_k",
        "average_precision_at_k",
        "majority_vote_correct",
    }
    missing = sorted(required - set(retrieval_fold_metrics.columns))
    if missing:
        raise ValueError(f"Retrieval metrics are missing columns: {missing}")
    frame = retrieval_fold_metrics.copy()
    frame["model"] = frame["model"].map(lambda value: _canonical(value, _MODEL_ALIASES))
    frame["strategy"] = frame["strategy"].map(
        lambda value: _canonical(value, _STRATEGY_ALIASES)
    )
    frame["task"] = frame["task"].map(lambda value: _canonical(value, _TASK_ALIASES))
    frame["k"] = pd.to_numeric(frame["k"], errors="raise").astype(int)
    frame["held_fold"] = pd.to_numeric(frame["held_fold"], errors="raise").astype(int)
    metric_labels = {
        "precision_at_k": "Precision",
        "recall_at_k": "Recall",
        "hit_at_k": "Hit",
        "average_precision_at_k": "mAP",
        "majority_vote_correct": "Majority vote accuracy",
    }
    for metric in metric_labels:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        if not np.isfinite(frame[metric]).all() or not frame[metric].between(0, 1).all():
            raise ValueError(f"Retrieval metric {metric} must be finite and lie in [0, 1]")
    row_keys = ["model", "strategy", "task", "held_fold", "k"]
    if frame.duplicated(row_keys).any():
        raise ValueError("Retrieval metrics contain duplicate fold/K rows")
    expected_systems = [
        (
            _canonical(model, _MODEL_ALIASES),
            _canonical(strategy, _STRATEGY_ALIASES),
            _canonical(task, _TASK_ALIASES),
        )
        for model in model_order
        for strategy in strategy_order
        for task in task_order
    ]
    expected_rows = {
        (*system, int(fold), int(k))
        for system in expected_systems
        for fold in expected_fold_ids
        for k in k_order
    }
    observed_rows = set(frame[row_keys].itertuples(index=False, name=None))
    if require_complete and observed_rows != expected_rows:
        raise ValueError(
            "Retrieval result grid is incomplete or unexpected; "
            f"missing={sorted(expected_rows - observed_rows)[:5]}, "
            f"extra={sorted(observed_rows - expected_rows)[:5]}"
        )
    base_columns = ["Model", "Strategy", "Task"]
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["model", "strategy", "task"], sort=False)
    systems = expected_systems if require_complete else list(grouped.groups)
    for model, strategy, task in systems:
        group = grouped.get_group((model, strategy, task))
        row: dict[str, Any] = {"Model": model, "Strategy": strategy, "Task": task}
        for k in k_order:
            k_group = group[group["k"] == int(k)]
            observed_folds = set(k_group["held_fold"].astype(int))
            if observed_folds != {int(value) for value in expected_fold_ids}:
                raise ValueError(
                    f"{model}/{strategy}/{task}/K={k} has folds {sorted(observed_folds)}"
                )
            for metric, label in metric_labels.items():
                values = k_group[metric].to_numpy(dtype=float)
                row[f"{label}@{int(k)}"] = format_mean_sd(
                    float(values.mean()), float(values.std(ddof=1))
                )
        rows.append(row)
    dynamic_columns = sorted(
        {column for row in rows for column in row if column not in base_columns},
        key=lambda value: (
            int(value.rsplit("@", 1)[1]),
            list(metric_labels.values()).index(value.rsplit("@", 1)[0]),
        ),
    )
    return pd.DataFrame(rows, columns=base_columns + dynamic_columns)


def _markdown(frame: pd.DataFrame) -> str:
    best_cells = set(frame.attrs.get("best_cells", set()))

    def cell(value: Any, *, row_index: int | None = None, column: str | None = None) -> str:
        rendered = str(value).replace("|", "\\|").replace("\n", " ")
        if row_index is not None and column is not None:
            if (row_index, column) in best_cells:
                rendered = f"**{rendered}**"
        return rendered

    headers = [cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        lines.append(
            "| "
            + " | ".join(
                cell(value, row_index=row_index, column=str(column))
                for column, value in zip(frame.columns, row, strict=True)
            )
            + " |"
        )
    note = frame.attrs.get("note")
    if note:
        lines.extend(["", f"_Note._ {cell(note)}"])
    return "\n".join(lines) + "\n"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\N{PLUS-MINUS SIGN}": r"$\pm$",
        "–": "--",
    }
    return "".join(replacements.get(character, character) for character in text)


def _latex(frame: pd.DataFrame) -> str:
    alignment = "l" * len(frame.columns)
    best_cells = set(frame.attrs.get("best_cells", set()))
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(_latex_escape(column) for column in frame.columns) + r" \\")
    lines.append("\\midrule")
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        rendered: list[str] = []
        for column, value in zip(frame.columns, row, strict=True):
            escaped = _latex_escape(value)
            if (row_index, str(column)) in best_cells:
                escaped = f"\\textbf{{{escaped}}}"
            rendered.append(escaped)
        lines.append(" & ".join(rendered) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    note = frame.attrs.get("note")
    if note:
        lines.extend(
            [
                r"\par\smallskip",
                r"\noindent\textit{Note:} " + _latex_escape(note),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _atomic_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding=encoding)
    os.replace(temporary, path)


def write_table_bundle(
    table: pd.DataFrame,
    output_dir: str | Path,
    *,
    stem: str,
) -> dict[str, Path]:
    """Write identical CSV, Markdown, and booktabs-compatible LaTeX tables."""

    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a simple non-empty filename stem.")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": directory / f"{stem}.csv",
        "markdown": directory / f"{stem}.md",
        "latex": directory / f"{stem}.tex",
    }
    csv_content = table.to_csv(index=False, lineterminator="\n")
    _atomic_text(paths["csv"], "\ufeff" + csv_content)
    _atomic_text(paths["markdown"], _markdown(table))
    _atomic_text(paths["latex"], _latex(table))
    return paths


def write_final_results_bundle(
    classification_seed_oof_metrics: pd.DataFrame,
    retrieval_fold_metrics: pd.DataFrame,
    output_dir: str | Path,
    *,
    model_order: Sequence[str] = MODEL_ORDER,
    strategy_order: Sequence[str] = STRATEGY_ORDER,
    retrieval_task_order: Sequence[str] = TASK_ORDER,
    retrieval_k_order: Sequence[int] = (5, 10, 15, 20),
) -> dict[str, dict[str, Path]]:
    """Write canonical Table 5 and unchanged retrieval tables in all formats.

    Seed-specific patient-cluster intervals remain separate supplementary
    evidence. They are deliberately excluded from this primary writer so a
    legacy ensemble interval cannot be presented beside the canonical table.
    """

    classification = build_manuscript_table(
        classification_seed_oof_metrics,
        model_order=model_order,
        strategy_order=strategy_order,
    )
    retrieval = build_retrieval_companion_table(
        retrieval_fold_metrics,
        model_order=model_order,
        strategy_order=strategy_order,
        task_order=retrieval_task_order,
        k_order=retrieval_k_order,
    )
    directory = Path(output_dir).resolve()
    return {
        "table5": write_table_bundle(
            classification, directory, stem="table5_manuscript"
        ),
        "retrieval": write_table_bundle(
            retrieval, directory, stem="retrieval_primary"
        ),
    }
