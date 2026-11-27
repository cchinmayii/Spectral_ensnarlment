"""
Visualization helpers for dual ensnarled-network calculations.

The functions in this module operate on the dual-circuit object produced by
``network_generation.py`` and on arrays returned by ``compute_ensnarlment.py``.
They do not modify graph topology, recompute cycle bases, or recompute linkage
matrices.

Two singular-mode visualizations are provided:

- ``plot_singular_vectors`` displays one individual left/right singular-vector
  pair from the SVD of the ensnarlment operator.
- ``plot_degenerate_singular_subspace`` combines all singular vectors belonging
  to one unique nonzero singular value into a basis-independent edge
  participation weight.

Plotly network figures can be exported as both static PDF files and interactive
HTML files with ``export_network_plot``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


__all__ = [
    "EIGENVECTOR_DIR",
    "STATIC_DIR",
    "INTERACTIVE_DIR",
    "export_network_plot",
    "plot_networks",
    "plot_legacy_edge_priorities",
    "plot_singular_vectors",
    "plot_degenerate_singular_subspace",
    "plot_selected_basis_cycles",
    "update_plot_eigvecs",
    "update_meshCyclePlot",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_dual_circuit(D: Any) -> None:
    """Validate the minimal dual-circuit interface used by this module."""
    if not hasattr(D, "layer"):
        raise TypeError("D must provide a 'layer' attribute.")

    if not hasattr(D, "plot_circuit"):
        raise TypeError("D must provide a 'plot_circuit' method.")

    if len(D.layer) != 2:
        raise ValueError(
            "These visualization functions expect exactly two network layers."
        )


def _validate_operator_against_circuit(
    D: Any,
    ensnarlment_operator: np.ndarray,
) -> np.ndarray:
    """Validate an edge-edge operator against the two plotted network layers."""
    operator = np.asarray(ensnarlment_operator, dtype=float)

    if operator.ndim != 2:
        raise ValueError("ensnarlment_operator must be two-dimensional.")

    expected_edges = operator.shape
    actual_edges = tuple(
        len(layer.G.edges())
        for layer in D.layer
    )

    if actual_edges != expected_edges:
        raise ValueError(
            "Operator shape does not match the plotted edge counts: "
            f"operator={operator.shape}, layer edges={actual_edges}."
        )

    return operator


def _clean_singular_values(
    singular_values: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Return a copy with numerically tiny singular values set to zero."""
    cleaned = np.asarray(
        singular_values,
        dtype=float,
    ).copy()

    cleaned[np.abs(cleaned) <= tol] = 0.0
    return cleaned


def _plot_edge_values(
    D: Any,
    edge_values: Sequence[np.ndarray],
    *,
    attr_name: str,
    colorbar_label: str | None = None,
    plot_kwargs: dict[str, Any] | None = None,
):
    """Attach one edge-value array to each layer and create the circuit plot."""
    if len(edge_values) != len(D.layer):
        raise ValueError(
            "edge_values must contain one array for each network layer."
        )

    colors: list[np.ndarray] = []

    for layer_index, (layer, values) in enumerate(
        zip(D.layer, edge_values)
    ):
        values = np.asarray(values, dtype=float).reshape(-1)

        n_edges = len(layer.G.edges())
        if values.size != n_edges:
            raise ValueError(
                f"Layer {layer_index} has {n_edges} edges, but "
                f"{values.size} edge values were supplied."
            )

        layer.edges[attr_name] = values
        colors.append(values)

    kwargs: dict[str, Any] = {
        "color_nodes": ["#030512", "#030512"],
        "color_edges": colors,
        "colormap": ["BuGn", "RdPu"],
        "axis": True,
    }

    if colorbar_label is not None:
        kwargs["colorbar"] = {
            "show": True,
            "label": colorbar_label,
        }

    if plot_kwargs:
        kwargs.update(plot_kwargs)

    return D.plot_circuit(attr_name, **kwargs)


# ---------------------------------------------------------------------------
# Plot export
# ---------------------------------------------------------------------------


def export_network_plot(
    fig: Any,
    filename: str,
    output_dir: str | Path,
    *,
    width: int = 1600,
    height: int = 900,
) -> dict[str, Path]:
    """
    Export a Plotly network figure as PDF and interactive HTML.

    Files are written to:

    - ``<output_dir>/statics/<filename>.pdf``
    - ``<output_dir>/interactives/<filename>.html``

    Parameters
    ----------
    fig
        Plotly figure returned by the network-visualization functions.
        The object must provide ``write_image`` and ``write_html`` methods.
    filename
        Output filename or stem. Any supplied suffix is removed before the
        ``.pdf`` and ``.html`` suffixes are added.
    output_dir
        Base directory for this export. Two subdirectories, ``statics`` and
        ``interactives``, are created inside it.
    width, height
        Static-image dimensions in pixels.

    Returns
    -------
    dict
        ``{"pdf": pdf_path, "html": html_path}``.

    Notes
    -----
    Plotly static-image export requires a working image-export backend such as
    Kaleido in the active Python environment.
    """
    if not hasattr(fig, "write_image"):
        raise TypeError("fig must provide a 'write_image' method.")

    if not hasattr(fig, "write_html"):
        raise TypeError("fig must provide a 'write_html' method.")

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    stem = Path(filename).stem
    if not stem:
        raise ValueError("filename must contain a non-empty stem.")

    output_dir = Path(output_dir)
    static_dir = output_dir / "statics"
    interactive_dir = output_dir / "interactives"

    static_dir.mkdir(parents=True, exist_ok=True)
    interactive_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = static_dir / f"{stem}.pdf"
    html_path = interactive_dir / f"{stem}.html"

    fig.write_image(
        str(pdf_path),
        width=width,
        height=height,
    )

    fig.write_html(
        str(html_path),
        full_html=True,
        include_plotlyjs=True,
        auto_open=False,
    )

    return {
        "pdf": pdf_path,
        "html": html_path,
    }

# ---------------------------------------------------------------------------
# Network plots
# ---------------------------------------------------------------------------


def plot_networks(
    D: Any,
    *,
    show: bool = False,
    **plot_kwargs: Any,
):
    """
    Plot the two ensnarled networks without applying an edge attribute.

    Parameters
    ----------
    D
        Dual-circuit object with ``D.layer`` and ``D.plot_circuit``.
    show
        If True, call ``fig.show()`` before returning the figure.
    **plot_kwargs
        Additional keyword arguments forwarded to ``D.plot_circuit``.

    Returns
    -------
    fig
        Plotly figure returned by ``D.plot_circuit``.
    """
    _validate_dual_circuit(D)

    fig = D.plot_circuit("", **plot_kwargs)

    if show and hasattr(fig, "show"):
        fig.show()

    return fig


def plot_legacy_edge_priorities(
    D: Any,
    priority_plotter: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
):
    """
    Run the legacy SnarlPy edge-priority plotting function.

    The legacy priority routine is supplied as a callable rather than imported
    here because its module path and signature may differ across SnarlPy
    versions.

    Examples
    --------
    ``plot_legacy_edge_priorities(D, update_priorityPlot, operator)``

    Returns
    -------
    Any
        Whatever object is returned by the supplied legacy plotting function.
    """
    _validate_dual_circuit(D)

    if not callable(priority_plotter):
        raise TypeError("priority_plotter must be callable.")

    return priority_plotter(D, *args, **kwargs)


# ---------------------------------------------------------------------------
# Singular-vector plots
# ---------------------------------------------------------------------------


def plot_singular_vectors(
    D: Any,
    ensnarlment_operator: np.ndarray,
    singular_vector_index: int = 0,
    *,
    tol: float = 1e-10,
    magnitude: bool = True,
    attr_name: str = "singular_vector",
    verbose: bool = True,
    plot_kwargs: dict[str, Any] | None = None,
):
    """
    Color both networks by one left/right singular-vector pair.

    For

    ``Lambda = U @ diag(s) @ Vh``,

    layer 0 is colored by column ``singular_vector_index`` of ``U`` and layer 1
    by the corresponding column of ``V = Vh.T``.

    This function indexes individual SVD vectors. For a degenerate singular
    value, individual vectors are basis-dependent; use
    ``plot_degenerate_singular_subspace`` when a basis-independent
    visualization of the full degenerate eigenspace is required.

    Parameters
    ----------
    D
        Dual-circuit object containing exactly two layers.
    ensnarlment_operator
        Edge-edge ensnarlment operator with shape ``(E1, E2)``.
    singular_vector_index
        Zero-based index of the individual singular-vector pair to display.
    tol
        Singular values with absolute value at most ``tol`` are printed as
        zero.
    magnitude
        If True, plot absolute component magnitudes. If False, preserve signs.
    attr_name
        Edge attribute used internally for plotting.
    verbose
        If True, print cleaned singular values and SVD validation checks.
    plot_kwargs
        Optional overrides forwarded to ``D.plot_circuit``.

    Returns
    -------
    fig
        Plotly figure returned by ``D.plot_circuit``.
    """
    _validate_dual_circuit(D)

    operator = _validate_operator_against_circuit(
        D,
        ensnarlment_operator,
    )

    U, singular_values, Vh = np.linalg.svd(
        operator,
        full_matrices=False,
    )
    V = Vh.T

    cleaned_singular_values = _clean_singular_values(
        singular_values,
        tol,
    )

    n_vector_pairs = singular_values.size
    if not 0 <= singular_vector_index < n_vector_pairs:
        raise IndexError(
            f"singular_vector_index={singular_vector_index} is out of bounds; "
            f"the operator has {n_vector_pairs} singular-vector pairs."
        )

    singular_vectors = (
        U[:, singular_vector_index],
        V[:, singular_vector_index],
    )

    if magnitude:
        edge_values = tuple(
            np.abs(vector)
            for vector in singular_vectors
        )
        colorbar_label = "Singular-vector magnitude (|component|)"
    else:
        edge_values = tuple(
            np.asarray(vector, dtype=float)
            for vector in singular_vectors
        )
        colorbar_label = "Singular-vector component"

    if verbose:
        reconstructed = (U * singular_values) @ Vh

        print("Singular values of Lambda =")
        print(cleaned_singular_values)

        print(
            "\nSVD computed correctly:",
            np.allclose(operator, reconstructed),
        )
        print(
            "Singular values in descending order:",
            bool(
                np.all(
                    singular_values[:-1]
                    >= singular_values[1:]
                )
            ),
        )

    return _plot_edge_values(
        D,
        edge_values,
        attr_name=attr_name,
        colorbar_label=colorbar_label,
        plot_kwargs=plot_kwargs,
    )


def plot_degenerate_singular_subspace(
    D: Any,
    ensnarlment_operator: np.ndarray,
    singular_subspace_index: int = 0,
    *,
    singular_vector_index: int | None = None,
    zero_tol: float = 1e-10,
    degeneracy_atol: float = 1e-8,
    degeneracy_rtol: float = 1e-5,
    attr_name: str = "singular_subspace_weight",
    verbose: bool = True,
    plot_kwargs: dict[str, Any] | None = None,
):
    """
    Color both networks by one complete nonzero singular-value subspace.

    Singular values are first grouped into unique nonzero values, using
    ``degeneracy_atol`` and ``degeneracy_rtol``. The groups remain in the
    descending order returned by the SVD:

    - ``singular_subspace_index=0`` selects the principal singular subspace;
    - ``singular_subspace_index=1`` selects the subspace associated with the
      second unique nonzero singular value;
    - and so on.

    If the selected singular value has multiplicity ``k``, its ``k`` singular
    vectors are combined using the sum of squared components:

    ``w_e = sum_j u[e, j]**2`` for layer 0,

    ``w_e = sum_j v[e, j]**2`` for layer 1,

    where ``j`` runs over all SVD columns belonging to the selected degenerate
    subspace. These weights are invariant under an orthogonal change of basis
    within that singular subspace, unlike the components of an individual
    singular vector.

    Parameters
    ----------
    D
        Dual-circuit object containing exactly two layers.
    ensnarlment_operator
        Edge-edge ensnarlment operator with shape ``(E1, E2)``.
    singular_subspace_index
        Zero-based index of the UNIQUE NONZERO singular-value subspace to
        visualize. This does not index raw SVD columns.
    singular_vector_index
        Backward-compatible keyword alias for ``singular_subspace_index``.
        If supplied, it is interpreted as a unique-subspace index despite the
        legacy name.
    zero_tol
        Singular values with absolute value at most ``zero_tol`` are excluded
        from the list of nonzero singular subspaces.
    degeneracy_atol, degeneracy_rtol
        Absolute and relative tolerances used by ``numpy.isclose`` to group
        neighboring singular values into the same degenerate subspace.
    attr_name
        Edge attribute used internally for plotting.
    verbose
        If True, print the complete spectrum, unique nonzero singular values,
        selected SVD-column indices, multiplicity, and reconstruction check.
    plot_kwargs
        Optional overrides forwarded to ``D.plot_circuit``.

    Returns
    -------
    fig
        Plotly figure returned by ``D.plot_circuit``.
    """
    _validate_dual_circuit(D)

    if singular_vector_index is not None:
        if singular_subspace_index != 0:
            raise ValueError(
                "Specify only one of singular_subspace_index or "
                "singular_vector_index."
            )
        singular_subspace_index = singular_vector_index

    operator = _validate_operator_against_circuit(
        D,
        ensnarlment_operator,
    )

    U, singular_values, Vh = np.linalg.svd(
        operator,
        full_matrices=False,
    )
    V = Vh.T

    cleaned_singular_values = _clean_singular_values(
        singular_values,
        zero_tol,
    )

    nonzero_indices = np.flatnonzero(
        singular_values > zero_tol
    )

    singular_subspaces: list[list[int]] = []

    for index in nonzero_indices:
        if not singular_subspaces:
            singular_subspaces.append([int(index)])
            continue

        representative_index = singular_subspaces[-1][0]

        if np.isclose(
            singular_values[index],
            singular_values[representative_index],
            atol=degeneracy_atol,
            rtol=degeneracy_rtol,
        ):
            singular_subspaces[-1].append(int(index))
        else:
            singular_subspaces.append([int(index)])

    if not singular_subspaces:
        raise ValueError(
            "The operator has no nonzero singular values above zero_tol."
        )

    unique_nonzero_singular_values = np.asarray(
        [
            singular_values[group[0]]
            for group in singular_subspaces
        ],
        dtype=float,
    )

    n_subspaces = len(singular_subspaces)

    if not 0 <= singular_subspace_index < n_subspaces:
        raise IndexError(
            f"singular_subspace_index={singular_subspace_index} is out of "
            f"bounds; the operator has {n_subspaces} unique nonzero singular "
            "subspaces."
        )

    degenerate_indices = np.asarray(
        singular_subspaces[singular_subspace_index],
        dtype=int,
    )

    target_sigma = float(
        unique_nonzero_singular_values[
            singular_subspace_index
        ]
    )

    U_subspace = U[:, degenerate_indices]
    V_subspace = V[:, degenerate_indices]

    U_combined = np.sum(U_subspace**2, axis=1)
    V_combined = np.sum(V_subspace**2, axis=1)

    if verbose:
        reconstructed = (U * singular_values) @ Vh

        print("All singular values of Lambda =")
        print(cleaned_singular_values)

        print("\nUnique nonzero singular values =")
        print(unique_nonzero_singular_values)

        print(
            "\nNumber of unique nonzero singular subspaces =",
            n_subspaces,
        )
        print(
            "\nSelected subspace index =",
            singular_subspace_index,
        )
        print(
            "Selected singular value =",
            target_sigma,
        )
        print(
            "SVD vector indices in this subspace =",
            degenerate_indices.tolist(),
        )
        print(
            "Degeneracy multiplicity =",
            degenerate_indices.size,
        )
        print(
            "\nSVD computed correctly:",
            np.allclose(operator, reconstructed),
        )

    return _plot_edge_values(
        D,
        (U_combined, V_combined),
        attr_name=attr_name,
        colorbar_label="Degenerate-subspace participation weight",
        plot_kwargs=plot_kwargs,
    )


# ---------------------------------------------------------------------------
# Cycle-basis plots
# ---------------------------------------------------------------------------


def plot_selected_basis_cycles(
    D: Any,
    mesh_matrices: Sequence[np.ndarray],
    cycle_ids: int | None | Sequence[int | None],
    *,
    one_indexed: bool = True,
    orientation: str = "edges_rows",
    use_sign: bool = False,
    base_value: float = 0.0,
    highlight_value: float = 1.0,
    attr_name: str = "mesh_cycle",
    verbose: bool = True,
    plot_kwargs: dict[str, Any] | None = None,
):
    """
    Highlight selected basis cycles on the two network layers.

    Parameters
    ----------
    D
        Dual-circuit object containing exactly two layers.
    mesh_matrices
        One directed mesh matrix per layer. In the current ensnarlment pipeline
        these have shape ``(n_edges, n_cycles)``.
    cycle_ids
        One cycle index per layer. Use ``None`` for a layer that should remain
        unhighlighted. A single integer is applied to both layers.
    one_indexed
        If True, cycle 1 corresponds to matrix index 0.
    orientation
        ``"edges_rows"`` for matrices shaped ``(E, C)`` or
        ``"cycles_rows"`` for matrices shaped ``(C, E)``.
    use_sign
        If True, preserve directed-mesh signs. Otherwise plot only membership.
    base_value
        Edge value outside the selected cycle.
    highlight_value
        Edge value inside the selected cycle when ``use_sign=False``.
    attr_name
        Edge attribute used internally for plotting.
    verbose
        If True, print the selected edge indices for each displayed cycle.
    plot_kwargs
        Optional overrides forwarded to ``D.plot_circuit``.

    Returns
    -------
    fig
        Plotly figure returned by ``D.plot_circuit``.
    """
    _validate_dual_circuit(D)

    if len(mesh_matrices) != len(D.layer):
        raise ValueError(
            "mesh_matrices must contain one matrix for each network layer."
        )

    if isinstance(cycle_ids, (int, type(None))):
        normalized_cycle_ids: list[int | None] = [
            cycle_ids
        ] * len(D.layer)
    else:
        normalized_cycle_ids = list(cycle_ids)

        if len(normalized_cycle_ids) != len(D.layer):
            raise ValueError(
                "cycle_ids must contain one entry for each network layer."
            )

    if orientation not in {
        "edges_rows",
        "cycles_rows",
    }:
        raise ValueError(
            "orientation must be either 'edges_rows' or 'cycles_rows'."
        )

    colors: list[np.ndarray] = []

    for layer_index, (layer, matrix, cycle_id) in enumerate(
        zip(
            D.layer,
            mesh_matrices,
            normalized_cycle_ids,
        )
    ):
        M = np.asarray(matrix)

        if M.ndim != 2:
            raise ValueError(
                f"mesh_matrices[{layer_index}] must be two-dimensional."
            )

        if orientation == "edges_rows":
            n_edges = M.shape[0]
            n_cycles = M.shape[1]
        else:
            n_edges = M.shape[1]
            n_cycles = M.shape[0]

        actual_n_edges = len(layer.G.edges())

        if n_edges != actual_n_edges:
            raise ValueError(
                f"Layer {layer_index} has {actual_n_edges} edges, but its "
                f"mesh matrix represents {n_edges}."
            )

        if cycle_id is None:
            edge_colors = np.full(
                n_edges,
                base_value,
                dtype=float,
            )
            layer.edges[attr_name] = edge_colors
            colors.append(edge_colors)
            continue

        cycle_index = (
            cycle_id - 1
            if one_indexed
            else cycle_id
        )

        if cycle_index < 0 or cycle_index >= n_cycles:
            raise IndexError(
                f"Cycle {cycle_id} is out of bounds for layer {layer_index}; "
                f"the matrix contains {n_cycles} cycles."
            )

        if orientation == "edges_rows":
            cycle_vector = M[:, cycle_index]
        else:
            cycle_vector = M[cycle_index, :]

        if use_sign:
            edge_colors = np.asarray(
                cycle_vector,
                dtype=float,
            )
        else:
            edge_colors = np.where(
                cycle_vector != 0,
                highlight_value,
                base_value,
            ).astype(float)

        layer.edges[attr_name] = edge_colors
        colors.append(edge_colors)

        if verbose:
            edge_indices = np.flatnonzero(
                cycle_vector
            )
            print(
                f"Layer {layer_index}, cycle {cycle_id}: "
                f"edges {edge_indices.tolist()}"
            )

    kwargs: dict[str, Any] = {
        "color_nodes": ["#030512", "#030512"],
        "color_edges": colors,
        "colormap": ["BuGn", "RdPu"],
        "axis": True,
    }

    if plot_kwargs:
        kwargs.update(plot_kwargs)

    return D.plot_circuit(
        attr_name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Compatibility aliases for existing notebooks
# ---------------------------------------------------------------------------


def update_plot_eigvecs(
    D: Any,
    operator: np.ndarray,
    eigvec_idx: int,
):
    """Legacy notebook alias for ``plot_singular_vectors``."""
    return plot_singular_vectors(
        D,
        operator,
        singular_vector_index=eigvec_idx,
    )


def update_meshCyclePlot(
    D: Any,
    mesh_mats: Sequence[np.ndarray],
    cycle_ids: int | None | Sequence[int | None],
    *,
    one_indexed: bool = True,
    orientation: str = "edges_rows",
    use_sign: bool = False,
    base_value: float = 0.0,
    highlight_value: float = 1.0,
    attr_name: str = "mesh_cycle",
):
    """Legacy notebook alias for ``plot_selected_basis_cycles``."""
    return plot_selected_basis_cycles(
        D,
        mesh_matrices=mesh_mats,
        cycle_ids=cycle_ids,
        one_indexed=one_indexed,
        orientation=orientation,
        use_sign=use_sign,
        base_value=base_value,
        highlight_value=highlight_value,
        attr_name=attr_name,
    )

