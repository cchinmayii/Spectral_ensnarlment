"""
Spectral analysis and plotting tools for ensnarlment operators.

The module supports two related workflows:

1. Analyze a single ensnarlment operator Lambda using its singular-value
   decomposition.
2. Aggregate and plot spectra collected across a parameter sweep, such as
   lattice period.

New code should use ``ensnarlment_operator`` and ``singular_values`` naming.
Older saved result dictionaries containing keys such as ``P_new`` or
``singvals_P_nonzero`` remain readable for backward compatibility.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.linalg as spla
from scipy.optimize import curve_fit


__all__ = [
    "load_results",
    "clean_singular_values",
    "compute_svd",
    "nonzero_singular_values",
    "numerical_rank",
    "group_degenerate_singular_values",
    "extract_singular_subspace",
    "adjacent_singular_value_spacings",
    "extract_singular_values",
    "summarize_results",
    "build_spectral_tables",
    "build_spacing_table",
    "plot_ordered_spectra",
    "plot_singular_value_distributions",
    "plot_principal_singular_value",
    "plot_adjacent_singular_value_spacings",
    "plot_spacing_distribution",
    "plot_spacing_heatmap",
    "fit_cubic_rank_scaling",
    "plot_cubic_rank_scaling",
    "fit_power_law_rank_scaling",
    "plot_power_law_rank_scaling",
]


# ---------------------------------------------------------------------------
# Single-operator spectral analysis
# ---------------------------------------------------------------------------


def clean_singular_values(
    singular_values,
    tol: float = 1e-10,
    *,
    copy: bool = True,
) -> np.ndarray:
    """
    Set numerically tiny singular values to zero.

    Parameters
    ----------
    singular_values : array_like
        Singular values.
    tol : float, default=1e-10
        Values with absolute magnitude at most ``tol`` are set to zero.
    copy : bool, default=True
        If True, do not modify the input array.

    Returns
    -------
    ndarray
        Cleaned one-dimensional singular-value array.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative.")

    values = np.array(
        singular_values,
        dtype=float,
        copy=copy,
    ).reshape(-1)

    values[np.abs(values) <= tol] = 0.0
    return values


def compute_svd(
    ensnarlment_operator,
    *,
    tol: float = 1e-10,
    full_matrices: bool = False,
    check_finite: bool = True,
    validate: bool = False,
) -> dict[str, Any]:
    """
    Compute the singular-value decomposition of an ensnarlment operator.

    Parameters
    ----------
    ensnarlment_operator : array_like, shape (E1, E2)
        Ensnarlment operator Lambda.
    tol : float, default=1e-10
        Numerical zero threshold used for cleaned values and rank.
    full_matrices : bool, default=False
        Passed to ``scipy.linalg.svd``.
    check_finite : bool, default=True
        Passed to ``scipy.linalg.svd``.
    validate : bool, default=False
        If True, verify the SVD reconstruction numerically.

    Returns
    -------
    result : dict
        Contains ``U``, ``singular_values``, ``Vh``,
        ``nonzero_singular_values``, ``rank``, and optional validation data.
    """
    Lambda = np.asarray(ensnarlment_operator)

    if Lambda.ndim != 2:
        raise ValueError("ensnarlment_operator must be two-dimensional.")

    U, singular_values, Vh = spla.svd(
        Lambda,
        full_matrices=full_matrices,
        check_finite=check_finite,
    )

    singular_values = clean_singular_values(
        singular_values,
        tol=tol,
    )
    nonzero = singular_values[singular_values > tol]

    result = {
        "U": U,
        "singular_values": singular_values,
        "Vh": Vh,
        "nonzero_singular_values": nonzero,
        "rank": int(nonzero.size),
        "tol": float(tol),
    }

    if validate:
        k = singular_values.size
        reconstructed = (
            U[:, :k]
            @ np.diag(singular_values)
            @ Vh[:k, :]
        )

        result["reconstruction_is_close"] = bool(
            np.allclose(Lambda, reconstructed)
        )
        result["reconstruction_error"] = float(
            spla.norm(Lambda - reconstructed)
        )

    return result


def nonzero_singular_values(
    values_or_operator,
    tol: float = 1e-10,
    *,
    input_is_operator: bool | None = None,
) -> np.ndarray:
    """
    Return nonzero singular values in descending order.

    ``input_is_operator`` may be supplied explicitly. If omitted, a
    two-dimensional input is treated as an operator and a one-dimensional
    input as an existing singular-value array.
    """
    array = np.asarray(values_or_operator)

    if input_is_operator is None:
        input_is_operator = array.ndim == 2

    if input_is_operator:
        values = spla.svdvals(array)
    else:
        values = array.reshape(-1)

    values = clean_singular_values(values, tol=tol)
    return np.sort(values[values > tol])[::-1]


def numerical_rank(
    values_or_operator,
    tol: float = 1e-10,
    *,
    input_is_operator: bool | None = None,
) -> int:
    """Return the number of singular values above ``tol``."""
    return int(
        nonzero_singular_values(
            values_or_operator,
            tol=tol,
            input_is_operator=input_is_operator,
        ).size
    )


def group_degenerate_singular_values(
    singular_values,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
    include_zero: bool = False,
) -> list[dict[str, Any]]:
    """
    Group approximately equal singular values and report multiplicities.

    Singular values are sorted in descending order before grouping.
    """
    values = np.sort(
        np.asarray(singular_values, dtype=float).reshape(-1)
    )[::-1]

    if not include_zero:
        values = values[values > atol]

    groups: list[dict[str, Any]] = []

    for index, value in enumerate(values):
        if not groups or not np.isclose(
            value,
            groups[-1]["value"],
            atol=atol,
            rtol=rtol,
        ):
            groups.append({
                "value": float(value),
                "multiplicity": 1,
                "sorted_indices": [index],
            })
        else:
            groups[-1]["multiplicity"] += 1
            groups[-1]["sorted_indices"].append(index)

    return groups


def extract_singular_subspace(
    U,
    singular_values,
    Vh,
    group_index: int = 0,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> dict[str, Any]:
    """
    Extract one degenerate singular subspace.

    ``group_index=0`` selects the leading nonzero singular-value group.
    """
    U = np.asarray(U)
    singular_values = np.asarray(
        singular_values,
        dtype=float,
    ).reshape(-1)
    Vh = np.asarray(Vh)

    groups = group_degenerate_singular_values(
        singular_values,
        atol=atol,
        rtol=rtol,
        include_zero=False,
    )

    if group_index < 0 or group_index >= len(groups):
        raise IndexError(
            f"group_index={group_index} is out of range for "
            f"{len(groups)} nonzero singular-value groups."
        )

    target = groups[group_index]["value"]

    indices = np.flatnonzero(
        np.isclose(
            singular_values,
            target,
            atol=atol,
            rtol=rtol,
        )
    )

    return {
        "singular_value": float(target),
        "multiplicity": int(indices.size),
        "indices": indices,
        "left_subspace": U[:, indices],
        "right_subspace": Vh[indices, :].T,
    }


def adjacent_singular_value_spacings(
    singular_values,
    tol: float = 1e-10,
) -> np.ndarray:
    """Return descending adjacent spacings sigma_i - sigma_(i+1)."""
    values = nonzero_singular_values(
        singular_values,
        tol=tol,
        input_is_operator=False,
    )

    if values.size < 2:
        return np.empty(0, dtype=float)

    return values[:-1] - values[1:]


# ---------------------------------------------------------------------------
# Saved-result compatibility and table construction
# ---------------------------------------------------------------------------


def load_results(
    path: str | Path,
    *,
    sort_key: str | None = "periods",
) -> list[dict[str, Any]]:
    """Load a pickled list of result dictionaries."""
    path = Path(path)

    with path.open("rb") as file:
        results = pickle.load(file)

    if not isinstance(results, list):
        raise TypeError("The pickle must contain a list of result dictionaries.")

    if sort_key is not None:
        results = sorted(results, key=lambda result: result[sort_key])

    return results


def _first_available_key(
    result: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        if key in result:
            return key
    return None


def extract_singular_values(
    result: Mapping[str, Any],
    *,
    tol: float = 1e-10,
    nonzero_only: bool = True,
) -> np.ndarray:
    """
    Extract or compute singular values from a result dictionary.

    New keys are checked first. Legacy ``P``-based keys are supported so old
    pickle files remain usable.
    """
    singular_value_key = _first_available_key(
        result,
        (
            "nonzero_singular_values",
            "singular_values",
            "singvals_Lambda_nonzero",
            "singvals_Lambda",
            "singvals_P_nonzero",
            "singvals_P",
        ),
    )

    if singular_value_key is not None:
        values = np.asarray(
            result[singular_value_key],
            dtype=float,
        ).reshape(-1)
    else:
        operator_key = _first_available_key(
            result,
            (
                "ensnarlment_operator",
                "Lambda",
                "Lambda_new",
                "P_new",
                "P",
            ),
        )

        if operator_key is None:
            raise KeyError(
                "The result contains neither singular values nor a recognized "
                "ensnarlment-operator key."
            )

        values = spla.svdvals(
            np.asarray(result[operator_key])
        )

    values = clean_singular_values(values, tol=tol)
    values = np.sort(values)[::-1]

    if nonzero_only:
        values = values[values > tol]

    return values


def summarize_results(
    results: Iterable[Mapping[str, Any]],
    *,
    parameter_key: str = "periods",
    tol: float = 1e-10,
) -> None:
    """Print a compact summary of a spectral parameter sweep."""
    results = sorted(results, key=lambda result: result[parameter_key])

    print("Number of parameter values collected:", len(results))

    for result in results:
        singular_values = extract_singular_values(
            result,
            tol=tol,
        )

        print(
            f"{parameter_key}: {result[parameter_key]}",
            "| nodes:", result.get("nodes", "unknown"),
            "| edges:", result.get("edges", "unknown"),
            "| cycles:", result.get("n_cycles", "unknown"),
            "| rank:", len(singular_values),
            "| basis:", result.get("cycle_basis_type", "unknown"),
        )


def build_spectral_tables(
    results: Iterable[Mapping[str, Any]],
    *,
    parameter_key: str = "periods",
    tol: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build long-form singular-value and per-parameter summary tables.

    Returns
    -------
    singular_values_table, summary_table : DataFrame
    """
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    results = sorted(results, key=lambda result: result[parameter_key])

    for result in results:
        parameter = result[parameter_key]
        singular_values = extract_singular_values(
            result,
            tol=tol,
        )

        for index, singular_value in enumerate(
            singular_values,
            start=1,
        ):
            rows.append({
                parameter_key: parameter,
                "index": index,
                "singval": float(singular_value),
            })

        if singular_values.size:
            sigma1 = float(singular_values[0])
            smallest_nonzero = float(singular_values[-1])
            mean_nonzero = float(np.mean(singular_values))
        else:
            sigma1 = np.nan
            smallest_nonzero = np.nan
            mean_nonzero = np.nan

        summary_rows.append({
            parameter_key: parameter,
            "rank": int(singular_values.size),
            "n_nonzero": int(singular_values.size),
            "sigma1": sigma1,
            "smallest_nonzero": smallest_nonzero,
            "mean_nonzero": mean_nonzero,
            "cycle_basis_type": result.get(
                "cycle_basis_type",
                "unknown",
            ),
            "total_time_s": result.get(
                "total_time_s",
                np.nan,
            ),
        })

    singular_values_table = pd.DataFrame(
        rows,
        columns=[parameter_key, "index", "singval"],
    )
    summary_table = (
        pd.DataFrame(summary_rows)
        .sort_values(parameter_key)
        .reset_index(drop=True)
    )

    return singular_values_table, summary_table


def build_spacing_table(
    results: Iterable[Mapping[str, Any]],
    *,
    parameter_key: str = "periods",
    tol: float = 1e-10,
) -> pd.DataFrame:
    """Build a long-form table of adjacent singular-value spacings."""
    rows: list[dict[str, Any]] = []

    results = sorted(results, key=lambda result: result[parameter_key])

    for result in results:
        parameter = result[parameter_key]
        singular_values = extract_singular_values(
            result,
            tol=tol,
        )
        spacings = adjacent_singular_value_spacings(
            singular_values,
            tol=tol,
        )

        for index, spacing in enumerate(spacings, start=1):
            rows.append({
                parameter_key: parameter,
                "index": index,
                "spacing": float(spacing),
            })

    return pd.DataFrame(
        rows,
        columns=[parameter_key, "index", "spacing"],
    )


# ---------------------------------------------------------------------------
# Spectral plots across parameter sweeps
# ---------------------------------------------------------------------------


def plot_ordered_spectra(
    results: Iterable[Mapping[str, Any]],
    *,
    parameter_key: str = "periods",
    tol: float = 1e-10,
    title: str = "Ordered singular-value spectra",
    palette: str = "crest",
    figsize: tuple[float, float] = (9, 6),
    show: bool = True,
):
    """Plot descending nonzero singular values for each parameter value."""
    results = sorted(results, key=lambda result: result[parameter_key])
    colors = sns.color_palette(palette, n_colors=len(results))

    fig, ax = plt.subplots(figsize=figsize)

    for color, result in zip(colors, results):
        parameter = result[parameter_key]
        singular_values = extract_singular_values(
            result,
            tol=tol,
        )

        ax.plot(
            np.arange(1, singular_values.size + 1),
            singular_values,
            color=color,
            linewidth=1.2,
            label=f"{parameter_key}={parameter}",
        )

    ax.set_xlabel(r"Spectral index $i$")
    ax.set_ylabel(r"Singular value $\sigma_i$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_singular_value_distributions(
    singular_values_table: pd.DataFrame,
    *,
    parameter_key: str = "periods",
    stat: str = "count",
    bins: int = 50,
    kde: bool = True,
    kde_bw_adjust: float = 0.5,
    title: str = "Nonzero singular-value distributions",
    palette: str = "crest",
    figsize: tuple[float, float] = (9, 5),
    show: bool = True,
    common_bins: bool = True,   # Shared bins across all spectra.
    element: str = "step",     # Histogram outlines.
    fill: bool = False,
):
    """Plot singular-value histograms for each parameter value."""
    if stat not in {"count", "density", "probability", "percent", "frequency"}:
        raise ValueError("Unsupported histogram statistic.")

    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(
        data=singular_values_table,
        x="singval",
        hue=parameter_key,
        bins=bins,
        stat=stat,
        common_norm=False,
        palette=palette,
        kde=kde,
        kde_kws={"bw_adjust": kde_bw_adjust},
        ax=ax,
        common_bins=common_bins,
        element=element,
        fill=fill,
        **({"edgecolor": "white"} if element == "bars" else {}),
    )

    ax.set_xlabel("Nonzero singular value")
    ax.set_ylabel("Count" if stat == "count" else stat.capitalize())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_principal_singular_value(
    summary_table: pd.DataFrame,
    *,
    parameter_key: str = "periods",
    title: str = "Principal singular value",
    figsize: tuple[float, float] = (7, 5),
    show: bool = True,
):
    """Plot the principal singular value against the sweep parameter."""
    summary_table = summary_table.sort_values(parameter_key)

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=summary_table,
        x=parameter_key,
        y="sigma1",
        marker="o",
        ax=ax,
    )

    ax.set_xlabel(parameter_key.replace("_", " ").capitalize())
    ax.set_ylabel(r"Principal singular value $\sigma_1$")
    ax.set_title(title)
    ax.set_xticks(summary_table[parameter_key])
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_adjacent_singular_value_spacings(
    results: Iterable[Mapping[str, Any]],
    *,
    parameter_range: tuple[float, float] | None = None,
    parameter_key: str = "periods",
    tol: float = 1e-10,
    title: str = "Adjacent singular-value spacings",
    palette: str = "crest",
    figsize: tuple[float, float] = (8, 5),
    show: bool = True,
):
    """Plot adjacent spacings for selected parameter values."""
    results = list(results)

    if parameter_range is not None:
        parameter_min, parameter_max = parameter_range
        results = [
            result
            for result in results
            if parameter_min
            <= result[parameter_key]
            <= parameter_max
        ]

    results = sorted(results, key=lambda result: result[parameter_key])

    if not results:
        raise ValueError("No results fall within the requested parameter range.")

    colors = sns.color_palette(palette, n_colors=len(results))
    fig, ax = plt.subplots(figsize=figsize)

    for color, result in zip(colors, results):
        singular_values = extract_singular_values(
            result,
            tol=tol,
        )
        spacings = adjacent_singular_value_spacings(
            singular_values,
            tol=tol,
        )

        ax.plot(
            np.arange(1, spacings.size + 1),
            spacings,
            marker=".",
            color=color,
            label=f"{parameter_key}={result[parameter_key]}",
        )

    ax.set_xlabel(r"Spectral index $i$")
    ax.set_ylabel(
        r"Adjacent spacing $\sigma_i-\sigma_{i+1}$"
    )
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_spacing_distribution(
    spacing_table: pd.DataFrame,
    *,
    parameter_key: str = "periods",
    showfliers: bool = False,
    title: str = "Distribution of adjacent singular-value spacings",
    figsize: tuple[float, float] = (9, 5),
    show: bool = True,
):
    """Plot boxplots of adjacent singular-value spacings."""
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=spacing_table,
        x=parameter_key,
        y="spacing",
        showfliers=showfliers,
        ax=ax,
    )

    ax.set_xlabel(parameter_key.replace("_", " ").capitalize())
    ax.set_ylabel(
        r"Adjacent spacing $\sigma_i-\sigma_{i+1}$"
    )
    ax.set_title(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_spacing_heatmap(
    results: Iterable[Mapping[str, Any]],
    *,
    parameter_key: str = "periods",
    tol: float = 1e-10,
    n_grid: int = 500,
    log_scale: bool = True,
    title: str = "Adjacent singular-value spacings",
    cmap: str = "crest",
    figsize: tuple[float, float] = (10, 5),
    show: bool = True,
):
    """
    Plot interpolated spacing sequences on a normalized spectral-index grid.

    Each parameter value may have a different number of singular values.
    Interpolation preserves approximate spectral position before stacking.
    """
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    results = sorted(results, key=lambda result: result[parameter_key])
    x_common = np.linspace(0, 1, n_grid)

    spacing_rows = []
    parameter_labels = []

    for result in results:
        singular_values = extract_singular_values(
            result,
            tol=tol,
        )
        spacings = adjacent_singular_value_spacings(
            singular_values,
            tol=tol,
        )

        if spacings.size == 0:
            continue

        if spacings.size == 1:
            interpolated = np.full(n_grid, spacings[0])
        else:
            x_original = np.linspace(0, 1, spacings.size)
            interpolated = np.interp(
                x_common,
                x_original,
                spacings,
            )

        spacing_rows.append(interpolated)
        parameter_labels.append(result[parameter_key])

    if not spacing_rows:
        raise ValueError(
            "At least one result must contain two nonzero singular values."
        )

    spacing_matrix = np.asarray(spacing_rows)

    if log_scale:
        positive = spacing_matrix[spacing_matrix > 0]
        epsilon = (
            positive.min() * 0.1
            if positive.size
            else 1e-16
        )
        plotted_matrix = np.log10(spacing_matrix + epsilon)
        colorbar_label = (
            r"$\log_{10}(\sigma_i-\sigma_{i+1})$"
        )
    else:
        plotted_matrix = spacing_matrix
        colorbar_label = (
            r"$\sigma_i-\sigma_{i+1}$"
        )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plotted_matrix,
        cmap=cmap,
        yticklabels=parameter_labels,
        xticklabels=False,
        cbar_kws={"label": colorbar_label},
        ax=ax,
    )

    ax.set_xlabel(r"Normalized spectral index $i/(N-1)$")
    ax.set_ylabel(parameter_key.replace("_", " ").capitalize())
    ax.set_title(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax, spacing_matrix


# ---------------------------------------------------------------------------
# Rank-scaling fits and plots
# ---------------------------------------------------------------------------


def _rank_table(
    spectral_table: pd.DataFrame,
    *,
    parameter_key: str,
) -> pd.DataFrame:
    required = {parameter_key, "singval"}
    missing = required - set(spectral_table.columns)

    if missing:
        raise ValueError(
            f"spectral_table is missing columns: {sorted(missing)}"
        )

    return (
        spectral_table
        .groupby(parameter_key)
        .size()
        .rename("rank")
        .reset_index()
        .sort_values(parameter_key)
        .reset_index(drop=True)
    )


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual_sum_squares = np.sum((observed - predicted) ** 2)
    total_sum_squares = np.sum(
        (observed - np.mean(observed)) ** 2
    )

    if total_sum_squares == 0:
        return np.nan

    return float(
        1 - residual_sum_squares / total_sum_squares
    )


def fit_cubic_rank_scaling(
    spectral_table: pd.DataFrame,
    *,
    parameter_key: str = "periods",
) -> dict[str, Any]:
    """Fit a general cubic polynomial to rank versus sweep parameter."""
    counts = _rank_table(
        spectral_table,
        parameter_key=parameter_key,
    )

    x = counts[parameter_key].to_numpy(dtype=float)
    rank = counts["rank"].to_numpy(dtype=float)

    if x.size < 4:
        raise ValueError(
            "At least four parameter values are required for a cubic fit."
        )

    scaled_fit = np.polynomial.Polynomial.fit(
        x,
        rank,
        deg=3,
    )
    standard_fit = scaled_fit.convert()
    predicted = scaled_fit(x)

    return {
        "counts": counts,
        "fit": standard_fit,
        "r_squared": _r_squared(rank, predicted),
    }


def plot_cubic_rank_scaling(
    spectral_table: pd.DataFrame,
    *,
    parameter_key: str = "periods",
    title: str = "Rank by parameter value",
    figsize: tuple[float, float] = (7, 5),
    show: bool = True,
):
    """Plot observed rank as bars with a general cubic fit."""
    fit_result = fit_cubic_rank_scaling(
        spectral_table,
        parameter_key=parameter_key,
    )

    counts = fit_result["counts"]
    polynomial = fit_result["fit"]
    r_squared = fit_result["r_squared"]

    x = counts[parameter_key].to_numpy(dtype=float)
    rank = counts["rank"].to_numpy(dtype=float)

    coefficients = np.pad(
        polynomial.coef,
        (0, max(0, 4 - polynomial.coef.size)),
    )
    a0, a1, a2, a3 = coefficients[:4]

    x_fit = np.linspace(x.min(), x.max(), 500)
    rank_fit = polynomial(x_fit)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x,
        rank,
        label="Observed rank",
    )
    ax.plot(
        x_fit,
        rank_fit,
        linewidth=2,
        label=(
            rf"Cubic fit: $r(p)={a3:.3g}p^3"
            rf"{a2:+.3g}p^2"
            rf"{a1:+.3g}p"
            rf"{a0:+.3g}$"
            "\n"
            rf"$R^2={r_squared:.5f}$"
        ),
    )

    ax.set_xlabel(parameter_key.replace("_", " ").capitalize())
    ax.set_ylabel(r"Rank $r$")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax, fit_result


def fit_power_law_rank_scaling(
    spectral_table: pd.DataFrame,
    *,
    parameter_key: str = "periods",
    fixed_exponent: float = 3.0,
) -> dict[str, Any]:
    """
    Fit ``rank = A * p**alpha`` and a fixed-exponent comparison model.
    """
    counts = _rank_table(
        spectral_table,
        parameter_key=parameter_key,
    )

    parameter = counts[parameter_key].to_numpy(dtype=float)
    rank = counts["rank"].to_numpy(dtype=float)

    if parameter.size < 2:
        raise ValueError(
            "At least two parameter values are required for a power-law fit."
        )

    if np.any(parameter <= 0):
        raise ValueError(
            "Power-law fitting requires strictly positive parameter values."
        )

    if np.any(rank <= 0):
        raise ValueError(
            "Power-law fitting requires strictly positive ranks."
        )

    def power_law(p, prefactor, exponent):
        return prefactor * p**exponent

    initial_prefactor = (
        rank[0] / parameter[0] ** fixed_exponent
    )

    parameters, covariance = curve_fit(
        power_law,
        parameter,
        rank,
        p0=(initial_prefactor, fixed_exponent),
        maxfev=10_000,
    )

    prefactor, exponent = parameters
    parameter_errors = np.sqrt(np.diag(covariance))
    prefactor_error, exponent_error = parameter_errors

    fixed_power = parameter**fixed_exponent
    fixed_prefactor = float(
        np.dot(fixed_power, rank)
        / np.dot(fixed_power, fixed_power)
    )

    predicted = power_law(
        parameter,
        prefactor,
        exponent,
    )

    return {
        "counts": counts,
        "prefactor": float(prefactor),
        "prefactor_error": float(prefactor_error),
        "exponent": float(exponent),
        "exponent_error": float(exponent_error),
        "fixed_exponent": float(fixed_exponent),
        "fixed_prefactor": fixed_prefactor,
        "r_squared": _r_squared(rank, predicted),
    }


def plot_power_law_rank_scaling(
    spectral_table: pd.DataFrame,
    *,
    parameter_key: str = "periods",
    fixed_exponent: float = 3.0,
    title: str = "Scaling of ensnarlment rank",
    figsize: tuple[float, float] = (7, 5),
    print_fit: bool = True,
    show: bool = True,
):
    """Plot free and fixed-exponent power-law fits to rank."""
    fit_result = fit_power_law_rank_scaling(
        spectral_table,
        parameter_key=parameter_key,
        fixed_exponent=fixed_exponent,
    )

    counts = fit_result["counts"]
    parameter = counts[parameter_key].to_numpy(dtype=float)
    rank = counts["rank"].to_numpy(dtype=float)

    parameter_fit = np.linspace(
        parameter.min(),
        parameter.max(),
        500,
    )

    free_rank_fit = (
        fit_result["prefactor"]
        * parameter_fit ** fit_result["exponent"]
    )
    fixed_rank_fit = (
        fit_result["fixed_prefactor"]
        * parameter_fit ** fixed_exponent
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        parameter,
        rank,
        s=60,
        label="Observed rank",
    )
    ax.plot(
        parameter_fit,
        free_rank_fit,
        linewidth=2,
        label=(
            rf"Power-law fit: "
            rf"$r={fit_result['prefactor']:.3g}"
            rf"p^{{{fit_result['exponent']:.3f}}}$"
            "\n"
            rf"$R^2={fit_result['r_squared']:.5f}$"
        ),
    )
    ax.plot(
        parameter_fit,
        fixed_rank_fit,
        linestyle="--",
        linewidth=2,
        label=(
            rf"Fixed exponent: "
            rf"$r={fit_result['fixed_prefactor']:.3g}"
            rf"p^{{{fixed_exponent:g}}}$"
        ),
    )

    ax.set_xlabel(parameter_key.replace("_", " ").capitalize())
    ax.set_ylabel(r"Rank $r$")
    ax.set_title(title)
    ax.set_xticks(parameter)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if print_fit:
        print(
            "Free exponent: "
            f"alpha = {fit_result['exponent']:.6f} "
            f"± {fit_result['exponent_error']:.6f}"
        )
        print(
            "Free prefactor: "
            f"A = {fit_result['prefactor']:.6f} "
            f"± {fit_result['prefactor_error']:.6f}"
        )
        print(
            "Fixed-exponent prefactor: "
            f"A = {fit_result['fixed_prefactor']:.6f}"
        )
        print(
            "Power-law R²: "
            f"{fit_result['r_squared']:.6f}"
        )

    if show:
        plt.show()

    return fig, ax, fit_result

