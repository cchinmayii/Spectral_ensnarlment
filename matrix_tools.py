"""
Matrix manipulation and visualization utilities for ensnarlment analysis.

The directed mesh matrices used by the current pipeline have shape
(edges, cycles) by default. Functions that depend on matrix orientation expose
an ``orientation`` argument so older cycle-by-edge matrices can still be used.
"""

from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


MatrixOrientation = Literal["edges_rows", "cycles_rows"]
IncidenceAxis = Literal["edges", "cycles"]


__all__ = [
    "push_zero_rows_cols",
    "remove_zero_rows_cols",
    "reorder_rows_to_match",
    "rcm_incidence_permutation",
    "reorder_mesh_matrices_and_linkage",
    "reciprocal_sqrt_form",
    "plot_matrix",
]


def _validate_matrix(A: np.ndarray, *, name: str = "A") -> np.ndarray:
    """Return ``A`` as a two-dimensional NumPy array."""
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")

    return A


def _zero_mask(A: np.ndarray, *, axis: int, tol: float) -> np.ndarray:
    """Return a Boolean mask identifying numerically zero rows or columns."""
    if tol < 0:
        raise ValueError("tol must be non-negative.")

    if tol == 0:
        return np.all(A == 0, axis=axis)

    return np.all(np.isclose(A, 0, atol=tol, rtol=0), axis=axis)


def push_zero_rows_cols(
    A,
    tol: float = 0,
    *,
    move_rows: bool = True,
    move_cols: bool = True,
    return_permutations: bool = False,
):
    """
    Move zero rows to the bottom and zero columns to the right.

    Parameters
    ----------
    A : array_like
        Two-dimensional matrix.
    tol : float, default=0
        Numerical zero tolerance. ``tol=0`` uses exact comparison.
    move_rows : bool, default=True
        Move numerically zero rows to the bottom.
    move_cols : bool, default=True
        Move numerically zero columns to the right.
    return_permutations : bool, default=False
        If True, also return the row and column permutations.

    Returns
    -------
    reordered : ndarray
        Matrix after applying the requested zero-row and zero-column moves.
    row_permutation : ndarray, optional
        Returned when ``return_permutations=True``.
    column_permutation : ndarray, optional
        Returned when ``return_permutations=True``.
    """
    A = _validate_matrix(A)

    row_permutation = np.arange(A.shape[0])
    column_permutation = np.arange(A.shape[1])

    if move_rows:
        zero_rows = _zero_mask(A, axis=1, tol=tol)
        row_permutation = np.concatenate(
            [
                np.flatnonzero(~zero_rows),
                np.flatnonzero(zero_rows),
            ]
        )

    if move_cols:
        zero_cols = _zero_mask(A, axis=0, tol=tol)
        column_permutation = np.concatenate(
            [
                np.flatnonzero(~zero_cols),
                np.flatnonzero(zero_cols),
            ]
        )

    reordered = A[np.ix_(row_permutation, column_permutation)]

    if return_permutations:
        return reordered, row_permutation, column_permutation

    return reordered


def remove_zero_rows_cols(
    A,
    tol: float = 1e-12,
    *,
    return_indices: bool = False,
):
    """
    Remove numerically zero rows and columns.

    Entries with absolute value below ``tol`` are set to zero before rows and
    columns are tested.

    Parameters
    ----------
    A : array_like
        Two-dimensional matrix.
    tol : float, default=1e-12
        Numerical zero tolerance.
    return_indices : bool, default=False
        If True, also return the retained row and column indices.

    Returns
    -------
    reduced : ndarray
        Matrix with all-zero rows and columns removed.
    row_indices : ndarray, optional
        Original row indices retained in ``reduced``.
    column_indices : ndarray, optional
        Original column indices retained in ``reduced``.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative.")

    A = _validate_matrix(A).astype(float, copy=True)
    A[np.abs(A) < tol] = 0

    nonzero_rows = np.any(A != 0, axis=1)
    nonzero_cols = np.any(A != 0, axis=0)

    row_indices = np.flatnonzero(nonzero_rows)
    column_indices = np.flatnonzero(nonzero_cols)
    reduced = A[np.ix_(row_indices, column_indices)]

    if return_indices:
        return reduced, row_indices, column_indices

    return reduced


def reorder_rows_to_match(
    M1,
    M2,
    tol: float = 0,
    *,
    use_abs: bool = False,
):
    """
    Reorder rows of ``M1`` to match ``M2``, retaining duplicate rows correctly.

    Zero rows are first moved to the bottom of both matrices. Columns are not
    reordered.

    Parameters
    ----------
    M1, M2 : array_like
        Matrices with equal shape and matching row multisets.
    tol : float, default=0
        Numerical matching tolerance. ``tol=0`` uses exact matching.
    use_abs : bool, default=False
        Match absolute row values, ignoring signs.

    Returns
    -------
    M1_reordered : ndarray
        ``M1`` reordered to match the zero-row-ordered version of ``M2``.
    row_permutation : ndarray
        Row permutation relative to the zero-row-ordered version of ``M1``.
    """
    M1 = _validate_matrix(M1, name="M1")
    M2 = _validate_matrix(M2, name="M2")

    M1_ordered = push_zero_rows_cols(
        M1,
        tol=tol,
        move_rows=True,
        move_cols=False,
    )
    M2_ordered = push_zero_rows_cols(
        M2,
        tol=tol,
        move_rows=True,
        move_cols=False,
    )

    if M1_ordered.shape != M2_ordered.shape:
        raise ValueError(
            f"Shape mismatch: {M1_ordered.shape} versus {M2_ordered.shape}."
        )

    X = np.abs(M1_ordered) if use_abs else M1_ordered
    Y = np.abs(M2_ordered) if use_abs else M2_ordered

    if tol == 0:
        X_keys = [tuple(row) for row in X]
        Y_keys = [tuple(row) for row in Y]
    else:
        if tol >= 1:
            decimals = 0
        else:
            decimals = int(np.ceil(-np.log10(tol)))

        X_keys = [tuple(np.round(row, decimals)) for row in X]
        Y_keys = [tuple(np.round(row, decimals)) for row in Y]

    row_lookup = defaultdict(deque)

    for row_index, key in enumerate(X_keys):
        row_lookup[key].append(row_index)

    row_permutation = []

    for target_index, key in enumerate(Y_keys):
        if not row_lookup[key]:
            raise ValueError(
                "No matching row in M1 for "
                f"row {target_index} of the ordered M2."
            )

        row_permutation.append(row_lookup[key].popleft())

    row_permutation = np.asarray(row_permutation, dtype=int)
    M1_reordered = M1_ordered[row_permutation]

    if use_abs:
        equal = np.allclose(
            np.abs(M1_reordered),
            np.abs(M2_ordered),
            atol=tol,
            rtol=0,
        )
    else:
        equal = np.allclose(
            M1_reordered,
            M2_ordered,
            atol=tol,
            rtol=0,
        )

    if not equal:
        raise RuntimeError("Row reordering failed.")

    return M1_reordered, row_permutation


def rcm_incidence_permutation(
    M,
    axis: IncidenceAxis,
    tol: float = 1e-12,
    *,
    orientation: MatrixOrientation = "edges_rows",
):
    """
    Construct a reverse Cuthill-McKee permutation from an incidence matrix.

    Parameters
    ----------
    M : array_like
        Cycle-edge incidence matrix.
    axis : {"cycles", "edges"}
        Entity to reorder.
    tol : float, default=1e-12
        Entries with absolute value at most ``tol`` are treated as zero.
    orientation : {"edges_rows", "cycles_rows"}, default="edges_rows"
        ``"edges_rows"`` means shape ``(n_edges, n_cycles)``.
        ``"cycles_rows"`` means shape ``(n_cycles, n_edges)``.

    Returns
    -------
    permutation : ndarray
        Reverse Cuthill-McKee permutation for the requested entity.
    """
    M = _validate_matrix(M, name="M")

    if axis not in {"cycles", "edges"}:
        raise ValueError("axis must be either 'cycles' or 'edges'.")

    if orientation not in {"edges_rows", "cycles_rows"}:
        raise ValueError(
            "orientation must be either 'edges_rows' or 'cycles_rows'."
        )

    support = csr_matrix((np.abs(M) > tol).astype(np.int8))

    if orientation == "edges_rows":
        adjacency = (
            support.T @ support
            if axis == "cycles"
            else support @ support.T
        )
    else:
        adjacency = (
            support @ support.T
            if axis == "cycles"
            else support.T @ support
        )

    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    if adjacency.nnz == 0:
        return np.arange(adjacency.shape[0], dtype=int)

    adjacency.data[:] = 1

    return reverse_cuthill_mckee(
        adjacency,
        symmetric_mode=True,
    )


def reorder_mesh_matrices_and_linkage(
    M1,
    M2,
    L,
    *,
    shared_permutations: bool = True,
    tol: float = 1e-12,
    orientation: MatrixOrientation = "edges_rows",
):
    """
    Reorder two mesh matrices and their cycle-linkage matrix using RCM.

    Parameters
    ----------
    M1, M2 : array_like
        Directed mesh matrices.
    L : array_like
        Cycle-linkage matrix with shape ``(n_cycles1, n_cycles2)``.
    shared_permutations : bool, default=True
        Use identical edge and cycle permutations for both mesh matrices.
        This requires equal shapes and equal nonzero patterns.
    tol : float, default=1e-12
        Numerical zero tolerance.
    orientation : {"edges_rows", "cycles_rows"}, default="edges_rows"
        Mesh-matrix orientation.

    Returns
    -------
    result : dict
        Reordered matrices and the edge/cycle permutations.
    """
    M1 = _validate_matrix(M1, name="M1")
    M2 = _validate_matrix(M2, name="M2")
    L = _validate_matrix(L, name="L")

    if orientation == "edges_rows":
        n_edges1, n_cycles1 = M1.shape
        n_edges2, n_cycles2 = M2.shape
    elif orientation == "cycles_rows":
        n_cycles1, n_edges1 = M1.shape
        n_cycles2, n_edges2 = M2.shape
    else:
        raise ValueError(
            "orientation must be either 'edges_rows' or 'cycles_rows'."
        )

    if L.shape != (n_cycles1, n_cycles2):
        raise ValueError(
            f"L has shape {L.shape}, but expected "
            f"{(n_cycles1, n_cycles2)}."
        )

    if shared_permutations:
        if M1.shape != M2.shape:
            raise ValueError(
                "Shared permutations require M1 and M2 to have equal shape."
            )

        support1 = np.abs(M1) > tol
        support2 = np.abs(M2) > tol

        if not np.array_equal(support1, support2):
            raise ValueError(
                "Shared permutations require M1 and M2 to have the same "
                "nonzero pattern."
            )

        cycle_perm1 = rcm_incidence_permutation(
            M1,
            axis="cycles",
            tol=tol,
            orientation=orientation,
        )
        edge_perm1 = rcm_incidence_permutation(
            M1,
            axis="edges",
            tol=tol,
            orientation=orientation,
        )

        cycle_perm2 = cycle_perm1.copy()
        edge_perm2 = edge_perm1.copy()

    else:
        cycle_perm1 = rcm_incidence_permutation(
            M1,
            axis="cycles",
            tol=tol,
            orientation=orientation,
        )
        edge_perm1 = rcm_incidence_permutation(
            M1,
            axis="edges",
            tol=tol,
            orientation=orientation,
        )
        cycle_perm2 = rcm_incidence_permutation(
            M2,
            axis="cycles",
            tol=tol,
            orientation=orientation,
        )
        edge_perm2 = rcm_incidence_permutation(
            M2,
            axis="edges",
            tol=tol,
            orientation=orientation,
        )

    if orientation == "edges_rows":
        M1_reordered = M1[np.ix_(edge_perm1, cycle_perm1)]
        M2_reordered = M2[np.ix_(edge_perm2, cycle_perm2)]
    else:
        M1_reordered = M1[np.ix_(cycle_perm1, edge_perm1)]
        M2_reordered = M2[np.ix_(cycle_perm2, edge_perm2)]

    L_reordered = L[np.ix_(cycle_perm1, cycle_perm2)]

    return {
        "M1": M1_reordered,
        "M2": M2_reordered,
        "L": L_reordered,
        "cycle_perm1": cycle_perm1,
        "cycle_perm2": cycle_perm2,
        "edge_perm1": edge_perm1,
        "edge_perm2": edge_perm2,
    }


def reciprocal_sqrt_form(
    value: float,
    max_den: int = 10_000,
) -> float:
    """
    Recover ``x`` when ``value`` is approximately ``1 / sqrt(x)``.

    The returned value is obtained by rationalizing ``1 / value**2`` with a
    maximum denominator of ``max_den``.
    """
    if value <= 0:
        raise ValueError("value must be positive.")

    if max_den <= 0:
        raise ValueError("max_den must be positive.")

    return float(
        Fraction(1.0 / value**2).limit_denominator(max_den)
    )


def plot_matrix(
    A,
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    vmin=None,
    vmax=None,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (10, 6),
    aspect: str = "auto",
    show: bool = True,
):
    """
    Plot a matrix as a heatmap.

    Parameters
    ----------
    A : array_like
        Matrix to display.
    title, xlabel, ylabel : str
        Plot labels.
    vmin, vmax : float or None
        Optional fixed color limits.
    cmap : str, default="RdBu_r"
        Matplotlib colormap.
    figsize : tuple, default=(10, 6)
        Figure size.
    aspect : str, default="auto"
        Image aspect passed to ``matshow``.
    show : bool, default=True
        Call ``plt.show()`` before returning.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes.
    """
    A = _validate_matrix(A)

    fig, ax = plt.subplots(figsize=figsize)

    image = ax.matshow(
        A,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
    )

    fig.colorbar(image, ax=ax)

    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax

