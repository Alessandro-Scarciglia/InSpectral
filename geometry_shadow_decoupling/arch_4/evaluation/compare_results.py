#!/usr/bin/env python3

"""
compare_results.py

Compare multiple NeRF/BARF-like experiments stored in TensorBoard result
folders.

The experiment definitions are stored separately in:

    comparison_cases.py

Example:

    python compare_results.py
    python compare_results.py --list
    python compare_results.py --case main_comparison

Outputs:
    - PDF figures
    - PNG figures
    - CSV statistics
    - Excel statistics workbook

Expected TensorBoard tags:

Training:
    Train/Total_Loss
    Train/Photometric_Loss
    Train/TV_Loss
    Train/Sparsity_Loss
    Train/Driving_Loss
    Train/Appearance_Loss
    Train/Distortion_Loss

Validation:
    Val/Full_PSNR
    Val/Object_PSNR
    Val/Background_PSNR

    Val/Full_MAE
    Val/Object_MAE
    Val/Background_MAE

    Val/Full_SSIM
    Val/Object_SSIM
    Val/Background_SSIM
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from comparison_cases import COMPARISON_CASES


# =============================================================================
# GLOBAL SETTINGS
# =============================================================================

FIGURE_DPI = 300

# Paper-friendly figure sizes
FIGURE_WIDTH = 6.5
FIGURE_HEIGHT = 4.2

# Supported output formats
SAVE_PDF = True
SAVE_PNG = True

# Number of points used for plotting, if downsampling is necessary
MAX_PLOT_POINTS = 2000

# Include epochs 0 through 14 in epoch-based plots.
MAX_PLOTTED_EPOCH = 14

# Small numerical value used to avoid division by zero
EPS = 1e-12


# =============================================================================
# MATPLOTLIB STYLE
# =============================================================================

def setup_matplotlib() -> None:
    """
    Configure a clean scientific-paper / Matlab-like plotting style.
    """

    plt.rcParams.update(
        {
            "figure.figsize": (FIGURE_WIDTH, FIGURE_HEIGHT),
            "figure.dpi": 120,

            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,

            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],

            "font.size": 11,

            "axes.labelsize": 11,
            "axes.titlesize": 12,

            "xtick.labelsize": 10,
            "ytick.labelsize": 10,

            "legend.fontsize": 9,

            "axes.linewidth": 0.8,

            "lines.linewidth": 1.8,
            "lines.markersize": 5,

            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,

            "axes.spines.top": False,
            "axes.spines.right": False,

            "legend.frameon": False,

            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def find_event_file(folder: Path) -> Path:
    """
    Find a TensorBoard event file recursively.

    This allows the experiment folder to contain the event file either
    directly or inside a subdirectory.
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Experiment folder does not exist:\n{folder}"
        )

    event_files = sorted(
        folder.rglob("events.out.tfevents.*")
    )

    if not event_files:
        raise FileNotFoundError(
            f"No TensorBoard event file found in:\n{folder}"
        )

    if len(event_files) > 1:
        print(
            f"[WARNING] Multiple TensorBoard event files found in:\n"
            f"          {folder}"
        )
        print(
            f"          Using:\n"
            f"          {event_files[-1]}"
        )

    return event_files[-1]


def safe_name(text: str) -> str:
    """
    Convert arbitrary text to a filesystem-safe name.
    """

    invalid = '<>:"/\\|?*'

    result = str(text)

    for char in invalid:
        result = result.replace(char, "_")

    result = result.replace(" ", "_")

    return result


def downsample_dataframe(
    df: pd.DataFrame,
    max_points: int = MAX_PLOT_POINTS,
) -> pd.DataFrame:
    """
    Downsample a dataframe for plotting if necessary.
    """

    if len(df) <= max_points:
        return df

    indices = np.linspace(
        0,
        len(df) - 1,
        max_points,
        dtype=int,
    )

    return df.iloc[indices]


def clean_numeric_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove NaN and infinite values.
    """

    if df.empty:
        return df

    result = df.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        result[column] = result[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return result.dropna()


# =============================================================================
# TENSORBOARD LOADING
# =============================================================================

def load_tensorboard_scalars(
    folder: Path,
) -> Dict[str, pd.DataFrame]:
    """
    Load all scalar TensorBoard metrics from an experiment folder.

    Returns
    -------
    dict
        {
            tag: DataFrame(
                {
                    "step": ...,
                    "value": ...
                }
            )
        }
    """

    event_file = find_event_file(folder)

    print(
        f"[INFO] Loading:\n"
        f"       {folder}\n"
        f"       {event_file}"
    )

    accumulator = EventAccumulator(
        str(event_file),
        size_guidance={
            "scalars": 0,
        },
    )

    accumulator.Reload()

    tags = accumulator.Tags().get("scalars", [])

    if not tags:
        raise RuntimeError(
            f"No scalar TensorBoard metrics found in:\n"
            f"{event_file}"
        )

    scalars: Dict[str, pd.DataFrame] = {}

    for tag in tags:
        events = accumulator.Scalars(tag)

        if not events:
            continue

        data = pd.DataFrame(
            {
                "step": [event.step for event in events],
                "value": [event.value for event in events],
            }
        )

        data = data.drop_duplicates(
            subset=["step"],
            keep="last",
        )

        data = data.sort_values("step")

        scalars[tag] = data.reset_index(drop=True)

    print(
        f"[INFO] Found {len(scalars)} scalar metrics."
    )

    return scalars


# =============================================================================
# EXPERIMENT LOADING
# =============================================================================

def load_experiments(
    experiments_config: List[dict],
) -> Dict[str, Dict]:
    """
    Load all experiments belonging to a comparison case.

    Each experiment entry should look like:

        {
            "name": "TV = 1e-5",
            "folder": Path(...)
        }
    """

    experiments = {}

    for experiment in experiments_config:

        name = experiment["name"]
        folder = Path(experiment["folder"])

        scalars = load_tensorboard_scalars(folder)

        experiments[name] = {
            "name": name,
            "folder": folder,
            "scalars": scalars,
        }

    return experiments


# =============================================================================
# METRIC TAGS
# =============================================================================

METRIC_TAGS = {
    "PSNR": {
        "full": "Val/Full_PSNR",
        "object": "Val/Object_PSNR",
        "background": "Val/Background_PSNR",
    },

    "MAE": {
        "full": "Val/Full_MAE",
        "object": "Val/Object_MAE",
        "background": "Val/Background_MAE",
    },

    "SSIM": {
        "full": "Val/Full_SSIM",
        "object": "Val/Object_SSIM",
        "background": "Val/Background_SSIM",
    },
}


TRAINING_TAGS = {
    "Total Loss": "Train/Total_Loss",
    "Photometric Loss": "Train/Photometric_Loss",
    "TV Loss": "Train/TV_Loss",
    "Sparsity Loss": "Train/Sparsity_Loss",
    "Driving Loss": "Train/Driving_Loss",
    "Appearance Loss": "Train/Appearance_Loss",
    "Distortion Loss": "Train/Distortion_Loss",
}


def get_metric(
    experiment: Dict,
    metric: str,
    region: str,
) -> Optional[pd.DataFrame]:
    """
    Retrieve a validation metric for an experiment.

    Parameters
    ----------
    metric:
        PSNR, MAE, or SSIM

    region:
        full, object, or background
    """

    tag = METRIC_TAGS[metric][region]

    scalars = experiment["scalars"]

    if tag not in scalars:
        warnings.warn(
            f"Metric '{tag}' not found for "
            f"experiment '{experiment['name']}'."
        )
        return None

    df = scalars[tag].copy()

    df = df.rename(
        columns={
            "step": "epoch",
        }
    )

    df = df.loc[
        df["epoch"].between(0, MAX_PLOTTED_EPOCH)
    ]

    return df[["epoch", "value"]]


def get_training_metric(
    experiment: Dict,
    metric_name: str,
) -> Optional[pd.DataFrame]:
    """
    Retrieve a training loss.
    """

    tag = TRAINING_TAGS[metric_name]

    scalars = experiment["scalars"]

    if tag not in scalars:
        warnings.warn(
            f"Training metric '{tag}' not found for "
            f"experiment '{experiment['name']}'."
        )
        return None

    df = scalars[tag].copy()

    df = df.rename(
        columns={
            "step": "iteration",
        }
    )

    return df[["iteration", "value"]]


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
) -> None:
    """
    Save figure as PDF and PNG.
    """

    ensure_directory(output_dir)

    if SAVE_PDF:
        pdf_path = output_dir / f"{filename}.pdf"

        fig.savefig(
            pdf_path,
            format="pdf",
            dpi=FIGURE_DPI,
        )

        print(f"[SAVED] {pdf_path}")

    if SAVE_PNG:
        png_path = output_dir / f"{filename}.png"

        fig.savefig(
            png_path,
            format="png",
            dpi=FIGURE_DPI,
        )

        print(f"[SAVED] {png_path}")

    plt.close(fig)


def plot_metric_vs_epoch(
    experiments: Dict[str, Dict],
    metric: str,
    output_dir: Path,
) -> None:
    """
    Plot Full/Object/Background metric versus epoch.
    """

    fig, ax = plt.subplots()

    for name, experiment in experiments.items():

        df = get_metric(
            experiment,
            metric,
            "full",
        )

        if df is None or df.empty:
            continue

        df = downsample_dataframe(df)

        ax.plot(
            df["epoch"],
            df["value"],
            label=name,
        )

    if metric == "PSNR":
        ylabel = "PSNR [dB]"
    elif metric == "MAE":
        ylabel = "Depth MAE"
    else:
        ylabel = "SSIM"

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    ax.set_title(
        f"{metric} vs. Epoch"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        f"{metric.lower()}_vs_epoch",
    )


def plot_object_metrics(
    experiments: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Plot object-region PSNR, MAE and SSIM.

    Three separate figures are used rather than a dense subplot.
    """

    metrics = [
        ("PSNR", "Object PSNR [dB]"),
        ("MAE", "Object Depth MAE"),
        ("SSIM", "Object SSIM"),
    ]

    for metric, ylabel in metrics:

        fig, ax = plt.subplots()

        for name, experiment in experiments.items():

            df = get_metric(
                experiment,
                metric,
                "object",
            )

            if df is None or df.empty:
                continue

            df = downsample_dataframe(df)

            ax.plot(
                df["epoch"],
                df["value"],
                label=name,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)

        ax.set_title(
            f"Object {metric} vs. Epoch"
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            output_dir,
            f"object_{metric.lower()}",
        )


def plot_object_background_gap(
    experiments: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Plot object-background performance gaps.

    PSNR:
        Object PSNR - Background PSNR

    SSIM:
        Object SSIM - Background SSIM

    MAE:
        Object MAE / Background MAE

    The MAE ratio is useful because MAE is an error metric.
    """

    # -------------------------------------------------------------------------
    # PSNR GAP
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots()

    for name, experiment in experiments.items():

        object_df = get_metric(
            experiment,
            "PSNR",
            "object",
        )

        background_df = get_metric(
            experiment,
            "PSNR",
            "background",
        )

        if object_df is None or background_df is None:
            continue

        df = pd.merge(
            object_df,
            background_df,
            on="epoch",
            suffixes=("_object", "_background"),
        )

        if df.empty:
            continue

        df["gap"] = (
            df["value_object"]
            - df["value_background"]
        )

        df = downsample_dataframe(df)

        ax.plot(
            df["epoch"],
            df["gap"],
            label=name,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Object − Background PSNR [dB]"
    )

    ax.set_title(
        "Object–Background PSNR Gap"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "object_background_psnr_gap",
    )

    # -------------------------------------------------------------------------
    # SSIM GAP
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots()

    for name, experiment in experiments.items():

        object_df = get_metric(
            experiment,
            "SSIM",
            "object",
        )

        background_df = get_metric(
            experiment,
            "SSIM",
            "background",
        )

        if object_df is None or background_df is None:
            continue

        df = pd.merge(
            object_df,
            background_df,
            on="epoch",
            suffixes=("_object", "_background"),
        )

        if df.empty:
            continue

        df["gap"] = (
            df["value_object"]
            - df["value_background"]
        )

        df = downsample_dataframe(df)

        ax.plot(
            df["epoch"],
            df["gap"],
            label=name,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Object − Background SSIM"
    )

    ax.set_title(
        "Object–Background SSIM Gap"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "object_background_ssim_gap",
    )

    # -------------------------------------------------------------------------
    # MAE RATIO
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots()

    for name, experiment in experiments.items():

        object_df = get_metric(
            experiment,
            "MAE",
            "object",
        )

        background_df = get_metric(
            experiment,
            "MAE",
            "background",
        )

        if object_df is None or background_df is None:
            continue

        df = pd.merge(
            object_df,
            background_df,
            on="epoch",
            suffixes=("_object", "_background"),
        )

        if df.empty:
            continue

        df["ratio"] = (
            df["value_object"]
            /
            (
                df["value_background"]
                + EPS
            )
        )

        df = downsample_dataframe(df)

        ax.plot(
            df["epoch"],
            df["ratio"],
            label=name,
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(
        "Object / Background Depth MAE"
    )

    ax.set_title(
        "Object–Background Depth MAE Ratio"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "object_background_mae_ratio",
    )


# =============================================================================
# RELATIVE IMPROVEMENT
# =============================================================================

def calculate_relative_improvement(
    values: pd.Series,
    metric: str,
) -> pd.Series:
    """
    Calculate relative improvement from the first observed value.

    For PSNR and SSIM:
        improvement = (current - initial) / |initial|

    For MAE:
        improvement = (initial - current) / |initial|

    Therefore, positive values represent improvement for all metrics.
    """

    values = values.astype(float)

    if len(values) == 0:
        return values

    initial = values.iloc[0]

    denominator = abs(initial)

    if denominator < EPS:
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
        )

    if metric == "MAE":
        improvement = (
            initial - values
        ) / denominator

    else:
        improvement = (
            values - initial
        ) / denominator

    return improvement * 100.0


def plot_relative_object_improvement(
    experiments: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Plot relative improvement of object-region metrics.
    """

    metrics = [
        ("PSNR", "Relative Object PSNR Improvement [%]"),
        ("MAE", "Relative Object Depth MAE Improvement [%]"),
        ("SSIM", "Relative Object SSIM Improvement [%]"),
    ]

    for metric, ylabel in metrics:

        fig, ax = plt.subplots()

        for name, experiment in experiments.items():

            df = get_metric(
                experiment,
                metric,
                "object",
            )

            if df is None or df.empty:
                continue

            df = df.copy()

            df["improvement"] = (
                calculate_relative_improvement(
                    df["value"],
                    metric,
                )
            )

            df = downsample_dataframe(df)

            ax.plot(
                df["epoch"],
                df["improvement"],
                label=name,
            )

        ax.axhline(
            0.0,
            linestyle="--",
            linewidth=1.0,
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)

        ax.set_title(
            f"Relative Object {metric} Improvement"
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            output_dir,
            f"relative_object_{metric.lower()}_improvement",
        )


# =============================================================================
# TRAINING LOSSES
# =============================================================================

def plot_training_losses(
    experiments: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Plot training losses.

    Each loss gets its own figure to keep the plots suitable for
    scientific papers.
    """

    for loss_name in TRAINING_TAGS:

        fig, ax = plt.subplots()

        found_any = False

        for name, experiment in experiments.items():

            df = get_training_metric(
                experiment,
                loss_name,
            )

            if df is None or df.empty:
                continue

            found_any = True

            df = downsample_dataframe(df)

            ax.plot(
                df["iteration"],
                df["value"],
                label=name,
            )

        if not found_any:
            plt.close(fig)
            continue

        ax.set_xlabel("Training Iteration")
        ax.set_ylabel(loss_name)

        ax.set_title(
            f"Training {loss_name}"
        )

        ax.legend()

        fig.tight_layout()

        save_figure(
            fig,
            output_dir,
            f"training_{safe_name(loss_name).lower()}",
        )


# =============================================================================
# STATISTICS
# =============================================================================

def final_value(
    df: Optional[pd.DataFrame],
) -> float:
    """Return the final observed value."""

    if df is None or df.empty:
        return np.nan

    return float(
        df.iloc[-1]["value"]
    )


def best_value(
    df: Optional[pd.DataFrame],
    metric: str,
) -> float:
    """
    Return best observed value.

    PSNR / SSIM:
        maximum

    MAE:
        minimum
    """

    if df is None or df.empty:
        return np.nan

    values = df["value"].to_numpy(
        dtype=float
    )

    if metric == "MAE":
        return float(np.nanmin(values))

    return float(np.nanmax(values))


def best_epoch(
    df: Optional[pd.DataFrame],
    metric: str,
) -> float:
    """Return epoch corresponding to best value."""

    if df is None or df.empty:
        return np.nan

    if metric == "MAE":
        index = df["value"].idxmin()
    else:
        index = df["value"].idxmax()

    return float(
        df.loc[index, "epoch"]
    )


def calculate_auc(
    df: Optional[pd.DataFrame],
    metric: Optional[str] = None,
) -> float:
    """
    Calculate area under the metric-vs-epoch curve.

    Compatibility:
        NumPy >= 2.x:
            np.trapezoid()

        Older NumPy:
            np.trapz()

    For MAE the sign is inverted so that a larger AUC corresponds
    to better performance, consistently with PSNR and SSIM.
    """

    if df is None or len(df) < 2:
        return np.nan

    data = df[
        ["epoch", "value"]
    ].dropna()

    if len(data) < 2:
        return np.nan

    x = data["epoch"].to_numpy(
        dtype=float
    )

    y = data["value"].to_numpy(
        dtype=float
    )

    # MAE is an error metric. Invert it so that larger AUC corresponds
    # to better performance.
    if metric == "MAE":
        y = -y

    # -------------------------------------------------------------------------
    # NumPy >= 2.0
    # -------------------------------------------------------------------------
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(
            y,
            x,
        )

    # -------------------------------------------------------------------------
    # NumPy < 2.0
    # -------------------------------------------------------------------------
    else:
        area = np.trapz(
            y,
            x,
        )

    return float(area)


def convergence_epoch(
    df: Optional[pd.DataFrame],
    metric: str,
    tolerance: float = 0.01,
) -> float:
    """
    Estimate the first epoch at which the metric reaches within
    `tolerance` of its best observed value.

    For PSNR / SSIM:
        value >= best * (1 - tolerance)

    For MAE:
        value <= best * (1 + tolerance)
    """

    if df is None or df.empty:
        return np.nan

    data = df[
        ["epoch", "value"]
    ].dropna()

    if data.empty:
        return np.nan

    values = data["value"].to_numpy(
        dtype=float
    )

    if metric == "MAE":

        best = np.min(values)

        threshold = (
            best * (1.0 + tolerance)
        )

        mask = values <= threshold

    else:

        best = np.max(values)

        threshold = (
            best * (1.0 - tolerance)
        )

        mask = values >= threshold

    indices = np.where(mask)[0]

    if len(indices) == 0:
        return np.nan

    return float(
        data.iloc[indices[0]]["epoch"]
    )


# =============================================================================
# STATISTICAL TABLES
# =============================================================================

def calculate_final_metrics_table(
    experiments: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Final values for all validation metrics.
    """

    rows = []

    for name, experiment in experiments.items():

        row = {
            "Experiment": name,
        }

        for metric in [
            "PSNR",
            "MAE",
            "SSIM",
        ]:

            for region in [
                "Full",
                "Object",
                "Background",
            ]:

                df = get_metric(
                    experiment,
                    metric,
                    region.lower(),
                )

                row[
                    f"{metric}_{region}"
                ] = final_value(df)

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_best_metrics_table(
    experiments: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Best observed validation metrics and epochs.
    """

    rows = []

    for name, experiment in experiments.items():

        row = {
            "Experiment": name,
        }

        for metric in [
            "PSNR",
            "MAE",
            "SSIM",
        ]:

            for region in [
                "Full",
                "Object",
                "Background",
            ]:

                region_key = region.lower()

                df = get_metric(
                    experiment,
                    metric,
                    region_key,
                )

                row[
                    f"Best_{metric}_{region}"
                ] = best_value(
                    df,
                    metric,
                )

                row[
                    f"Best_{metric}_{region}_Epoch"
                ] = best_epoch(
                    df,
                    metric,
                )

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_auc_table(
    experiments: Dict[str, Dict],
) -> pd.DataFrame:
    """
    AUC of validation metrics over epochs.
    """

    rows = []

    for name, experiment in experiments.items():

        row = {
            "Experiment": name,
        }

        for metric in [
            "PSNR",
            "MAE",
            "SSIM",
        ]:

            for region in [
                "Full",
                "Object",
                "Background",
            ]:

                df = get_metric(
                    experiment,
                    metric,
                    region.lower(),
                )

                row[
                    f"AUC_{metric}_{region}"
                ] = calculate_auc(
                    df,
                    metric,
                )

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_convergence_table(
    experiments: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Epoch at which each metric reaches within 1% of its best
    observed value.
    """

    rows = []

    for name, experiment in experiments.items():

        row = {
            "Experiment": name,
        }

        for metric in [
            "PSNR",
            "MAE",
            "SSIM",
        ]:

            for region in [
                "Full",
                "Object",
                "Background",
            ]:

                df = get_metric(
                    experiment,
                    metric,
                    region.lower(),
                )

                row[
                    f"Convergence_{metric}_{region}"
                ] = convergence_epoch(
                    df,
                    metric,
                    tolerance=0.01,
                )

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_object_background_table(
    experiments: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Calculate final object-background differences and ratios.
    """

    rows = []

    for name, experiment in experiments.items():

        row = {
            "Experiment": name,
        }

        # ---------------------------------------------------------------------
        # PSNR
        # ---------------------------------------------------------------------

        object_psnr = get_metric(
            experiment,
            "PSNR",
            "object",
        )

        background_psnr = get_metric(
            experiment,
            "PSNR",
            "background",
        )

        final_object_psnr = final_value(
            object_psnr
        )

        final_background_psnr = final_value(
            background_psnr
        )

        row["Final_Object_PSNR"] = (
            final_object_psnr
        )

        row["Final_Background_PSNR"] = (
            final_background_psnr
        )

        row["PSNR_Object_Background_Gap"] = (
            final_object_psnr
            - final_background_psnr
        )

        # ---------------------------------------------------------------------
        # SSIM
        # ---------------------------------------------------------------------

        object_ssim = get_metric(
            experiment,
            "SSIM",
            "object",
        )

        background_ssim = get_metric(
            experiment,
            "SSIM",
            "background",
        )

        final_object_ssim = final_value(
            object_ssim
        )

        final_background_ssim = final_value(
            background_ssim
        )

        row["Final_Object_SSIM"] = (
            final_object_ssim
        )

        row["Final_Background_SSIM"] = (
            final_background_ssim
        )

        row["SSIM_Object_Background_Gap"] = (
            final_object_ssim
            - final_background_ssim
        )

        # ---------------------------------------------------------------------
        # MAE
        # ---------------------------------------------------------------------

        object_mae = get_metric(
            experiment,
            "MAE",
            "object",
        )

        background_mae = get_metric(
            experiment,
            "MAE",
            "background",
        )

        final_object_mae = final_value(
            object_mae
        )

        final_background_mae = final_value(
            background_mae
        )

        row["Final_Object_Depth_MAE"] = (
            final_object_mae
        )

        row["Final_Background_Depth_MAE"] = (
            final_background_mae
        )

        if (
            not np.isnan(final_object_mae)
            and not np.isnan(final_background_mae)
            and abs(final_background_mae) > EPS
        ):
            row["Object_Background_MAE_Ratio"] = (
                final_object_mae
                / final_background_mae
            )
        else:
            row["Object_Background_MAE_Ratio"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def save_statistics(
    experiments: Dict[str, Dict],
    output_dir: Path,
) -> None:
    """
    Generate and save all statistical tables.
    """

    ensure_directory(output_dir)

    print("[INFO] Calculating statistics...")

    final_df = calculate_final_metrics_table(
        experiments
    )

    best_df = calculate_best_metrics_table(
        experiments
    )

    auc_df = calculate_auc_table(
        experiments
    )

    convergence_df = calculate_convergence_table(
        experiments
    )

    object_background_df = (
        calculate_object_background_table(
            experiments
        )
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    csv_tables = {
        "final_metrics": final_df,
        "best_metrics": best_df,
        "auc_metrics": auc_df,
        "convergence_metrics": convergence_df,
        "object_background_metrics": (
            object_background_df
        ),
    }

    for filename, df in csv_tables.items():

        path = (
            output_dir
            / f"{filename}.csv"
        )

        df.to_csv(
            path,
            index=False,
        )

        print(
            f"[SAVED] {path}"
        )

    # -------------------------------------------------------------------------
    # Excel
    # -------------------------------------------------------------------------

    excel_path = (
        output_dir
        / "comparison_statistics.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        final_df.to_excel(
            writer,
            sheet_name="Final Metrics",
            index=False,
        )

        best_df.to_excel(
            writer,
            sheet_name="Best Metrics",
            index=False,
        )

        auc_df.to_excel(
            writer,
            sheet_name="AUC",
            index=False,
        )

        convergence_df.to_excel(
            writer,
            sheet_name="Convergence",
            index=False,
        )

        object_background_df.to_excel(
            writer,
            sheet_name="Object Background",
            index=False,
        )

    print(
        f"[SAVED] {excel_path}"
    )


# =============================================================================
# COMPARISON EXECUTION
# =============================================================================

def run_comparison(
    case_name: str,
    case_config: dict,
) -> None:
    """
    Run a single comparison case.
    """

    print()
    print("=" * 80)
    print(
        f"COMPARISON: {case_name}"
    )
    print("=" * 80)

    description = case_config.get(
        "description",
        "",
    )

    if description:
        print(description)

    print()

    experiments_config = case_config[
        "experiments"
    ]

    output_dir = Path(
        case_config["output"]
    )

    print(
        f"[INFO] Output directory:\n"
        f"       {output_dir}"
    )

    print()

    # -------------------------------------------------------------------------
    # Load experiments
    # -------------------------------------------------------------------------

    experiments = load_experiments(
        experiments_config
    )

    if len(experiments) == 0:
        raise RuntimeError(
            f"No experiments defined for "
            f"comparison '{case_name}'."
        )

    # -------------------------------------------------------------------------
    # Create output directory
    # -------------------------------------------------------------------------

    ensure_directory(output_dir)

    # -------------------------------------------------------------------------
    # Main validation plots
    # -------------------------------------------------------------------------

    plot_metric_vs_epoch(
        experiments,
        "PSNR",
        output_dir,
    )

    plot_metric_vs_epoch(
        experiments,
        "MAE",
        output_dir,
    )

    plot_metric_vs_epoch(
        experiments,
        "SSIM",
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Object metrics
    # -------------------------------------------------------------------------

    plot_object_metrics(
        experiments,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Object/background analysis
    # -------------------------------------------------------------------------

    plot_object_background_gap(
        experiments,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Relative improvement
    # -------------------------------------------------------------------------

    plot_relative_object_improvement(
        experiments,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Training losses
    # -------------------------------------------------------------------------

    plot_training_losses(
        experiments,
        output_dir,
    )

    # -------------------------------------------------------------------------
    # Statistical tables
    # -------------------------------------------------------------------------

    save_statistics(
        experiments,
        output_dir,
    )

    print()
    print(
        f"[DONE] Comparison '{case_name}' completed."
    )
    print(
        f"[DONE] Results saved to:\n"
        f"       {output_dir}"
    )


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def list_comparisons() -> None:
    """
    Print all available comparison cases.
    """

    print()
    print(
        "Available comparison cases:"
    )
    print(
        "-" * 80
    )

    for name, config in COMPARISON_CASES.items():

        description = config.get(
            "description",
            "",
        )

        experiments = config.get(
            "experiments",
            [],
        )

        print(
            f"\n{name}"
        )

        if description:
            print(
                f"    {description}"
            )

        print(
            "    Experiments:"
        )

        for experiment in experiments:

            print(
                f"        - {experiment['name']}"
            )

            print(
                f"          {experiment['folder']}"
            )

        print(
            f"    Output:"
        )

        print(
            f"        {config['output']}"
        )

    print()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple TensorBoard experiments."
        )
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help=(
            "Name of the comparison case to run. "
            "If omitted, all comparison cases are run."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help=(
            "List all available comparison cases."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point.
    """

    setup_matplotlib()

    args = parse_arguments()

    # -------------------------------------------------------------------------
    # List available cases
    # -------------------------------------------------------------------------

    if args.list:
        list_comparisons()
        return

    # -------------------------------------------------------------------------
    # Run one selected case
    # -------------------------------------------------------------------------

    if args.case is not None:

        if args.case not in COMPARISON_CASES:

            print(
                f"[ERROR] Unknown comparison case:"
                f" '{args.case}'"
            )

            print()

            list_comparisons()

            raise SystemExit(1)

        print(
            "[INFO] Running 1 comparison(s)."
        )

        run_comparison(
            args.case,
            COMPARISON_CASES[args.case],
        )

        return

    # -------------------------------------------------------------------------
    # Run all cases
    # -------------------------------------------------------------------------

    print(
        f"[INFO] Running "
        f"{len(COMPARISON_CASES)} comparison(s)."
    )

    for case_name, case_config in (
        COMPARISON_CASES.items()
    ):

        run_comparison(
            case_name,
            case_config,
        )

    print()
    print(
        "=" * 80
    )
    print(
        "[DONE] All comparisons completed."
    )
    print(
        "=" * 80
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()