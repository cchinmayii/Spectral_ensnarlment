"""
edge_cutting.py
===============

Edge-cutting algorithms for two spatially ensnarled networks.

The module implements four related workflows:

1. Recomputed-SVD cutting
   The residual edge-space operator is reconstructed after every accepted cut.
   Its leading singular subspace is recomputed and used to score the remaining
   edges.

2. Fixed-original-SVD cutting
   The SVD of the original edge-space operator is computed once. Edge scores
   remain fixed throughout the cutting process. Cuts may be drawn from the
   leading, second, or further distinct singular-value subspaces.

3. Two-sided cutting
   Rank-reducing single-edge cuts are alternated between graph 1 and graph 2.
   The residual operator and its principal singular subspaces are recomputed
   after every accepted cut.

4. Independent post-cut validation
   Selected edges are physically removed from copies of the NetworkX graphs.
   Fresh cycle bases, mesh matrices, and Gauss linking numbers are then
   recomputed independently.

Conventions
-----------
M1_dir has shape (E1, C1).
M2_dir has shape (E2, C2).
lk_mat has shape (C1, C2).

Cutting an edge imposes a linear constraint on cycle coefficients. If S1 is
the set of cut rows in M1_dir, then the surviving graph-1 cycle combinations
are represented by an orthonormal basis Q1 for

    null(M1_dir[S1, :]).

Similarly, Q2 spans the surviving cycle combinations in graph 2.

The reduced linkage matrix is

    L_res = Q1.T @ lk_mat @ Q2.

The corresponding edge-space operator is

    Lambda_res = (
        pinv(M1_dir @ Q1).T
        @ L_res
        @ pinv(M2_dir @ Q2)
    ).

The module records both rank(L_res) and rank(Lambda_res). These should agree when
the residual directed mesh matrices retain full column rank, but both are
stored explicitly so that any disagreement is visible.

Required packages
-----------------
numpy
scipy
networkx

Fresh geometric validation additionally requires the project-specific
`sampling`, `edge_algebra`, and `signature` modules, together with functions
for cycle rank and linkage-dictionary conversion.
"""

from __future__ import annotations

from itertools import combinations
import time
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
import scipy.linalg as spla

from plotly.colors import sample_colorscale


RankMeasure = Literal["linkage", "operator"]


__all__ = [
    # Basic numerical and residual-state utilities
    "numerical_rank",
    "normalize_cut_indices",
    "compute_residual_state",
    "compute_one_sided_residual_state",
    "group_nonzero_singular_subspaces",
    "singular_subspace_edge_scores",
    "principal_edge_scores",

    # Recomputed-SVD algorithms
    "find_recomputed_single_candidates",
    "find_recomputed_pair_candidates",
    "cut_with_recomputed_svd",

    # Fixed-original-SVD algorithms
    "cut_with_original_svd",
    "find_original_svd_pair_candidates",

    # Two-sided algorithm
    "find_two_sided_single_candidates",
    "cut_both_graphs_with_recomputed_svd",
    
    # Result-display and visualization helpers
    "print_cut_result",
    "print_two_sided_cut_result",
    "plot_cut_edges",

    # SnarlPy compatibility
    "run_snarlpy_legacy_cutting",

    # Graph manipulation and validation
    "extract_networkx_graph",
    "edge_indices_to_labels",
    "apply_edge_cuts",
    "compute_fresh_post_cut_linkage",
    "validate_edge_cut_set",
]


# ============================================================================ #
# General utilities
# ============================================================================ #


def numerical_rank(
    matrix: np.ndarray,
    tol: float = 1e-10,
) -> int:
    """
    Compute numerical matrix rank using an absolute singular-value tolerance.

    Parameters
    ----------
    matrix
        Input matrix.
    tol
        A singular value is counted as nonzero when it is strictly larger
        than this value.

    Returns
    -------
    int
        Number of singular values larger than ``tol``.

    Notes
    -----
    This function deliberately uses an absolute tolerance so that the same
    threshold is applied throughout the cutting pipeline.
    """
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError(
            f"matrix must be two-dimensional; received shape {matrix.shape}."
        )

    if tol < 0:
        raise ValueError("tol must be nonnegative.")

    if matrix.size == 0:
        return 0

    singular_values = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    return int(np.count_nonzero(singular_values > tol))


def normalize_cut_indices(
    cut_indices: Iterable[int] | None,
    number_of_edges: int,
    *,
    name: str = "cut_indices",
) -> tuple[int, ...]:
    """
    Convert edge indices to a sorted, duplicate-free tuple and validate them.

    Parameters
    ----------
    cut_indices
        Iterable of edge indices. ``None`` is treated as an empty collection.
    number_of_edges
        Total number of edges on the relevant graph.
    name
        Name used in error messages.

    Returns
    -------
    tuple of int
        Sorted unique edge indices.
    """
    if number_of_edges < 0:
        raise ValueError("number_of_edges must be nonnegative.")

    if cut_indices is None:
        return ()

    normalized = tuple(
        sorted(
            {
                int(edge_index)
                for edge_index in cut_indices
            }
        )
    )

    invalid = [
        edge_index
        for edge_index in normalized
        if edge_index < 0 or edge_index >= number_of_edges
    ]

    if invalid:
        raise IndexError(
            f"{name} contains out-of-range indices {invalid}. "
            f"Valid indices are 0 through {number_of_edges - 1}."
        )

    return normalized


def _validate_cutting_inputs(
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and convert the three core matrices."""
    M1_dir = np.asarray(M1_dir, dtype=float)
    M2_dir = np.asarray(M2_dir, dtype=float)
    lk_mat = np.asarray(lk_mat, dtype=float)

    if M1_dir.ndim != 2:
        raise ValueError(
            f"M1_dir must be two-dimensional; got shape {M1_dir.shape}."
        )

    if M2_dir.ndim != 2:
        raise ValueError(
            f"M2_dir must be two-dimensional; got shape {M2_dir.shape}."
        )

    if lk_mat.ndim != 2:
        raise ValueError(
            f"lk_mat must be two-dimensional; got shape {lk_mat.shape}."
        )

    expected_linkage_shape = (
        M1_dir.shape[1],
        M2_dir.shape[1],
    )

    if lk_mat.shape != expected_linkage_shape:
        raise ValueError(
            "Shape mismatch. Expected "
            f"lk_mat.shape == {expected_linkage_shape}, "
            f"but received {lk_mat.shape}."
        )

    if not (
        np.all(np.isfinite(M1_dir))
        and np.all(np.isfinite(M2_dir))
        and np.all(np.isfinite(lk_mat))
    ):
        raise ValueError(
            "M1_dir, M2_dir, and lk_mat must contain only finite values."
        )

    return M1_dir, M2_dir, lk_mat


def _null_space_after_edge_cuts(
    directed_mesh_matrix: np.ndarray,
    cut_indices: Sequence[int],
    rank_tol: float,
) -> np.ndarray:
    """
    Compute the surviving cycle-coordinate basis after imposing edge cuts.

    If no edges are cut, the original cycle-coordinate basis is returned.
    Otherwise, the columns of Q form an orthonormal basis for

        null(directed_mesh_matrix[cut_indices, :]).
    """
    number_of_cycles = directed_mesh_matrix.shape[1]

    if len(cut_indices) == 0:
        return np.eye(number_of_cycles, dtype=float)

    constraint_matrix = directed_mesh_matrix[
        np.asarray(cut_indices, dtype=int),
        :
    ]

    return spla.null_space(
        constraint_matrix,
        rcond=rank_tol,
    )


def _state_rank(
    state: Mapping[str, Any],
    rank_measure: RankMeasure,
) -> int:
    """Retrieve the selected residual-rank definition from a state."""
    if rank_measure == "linkage":
        return int(state["linkage_rank"])

    if rank_measure == "operator":
        return int(state["operator_rank"])

    raise ValueError(
        "rank_measure must be either 'linkage' or 'operator'."
    )


def _state_norm(
    state: Mapping[str, Any],
    rank_measure: RankMeasure,
) -> float:
    """Retrieve the norm corresponding to the selected rank measure."""
    if rank_measure == "linkage":
        return float(state["linkage_norm"])

    if rank_measure == "operator":
        return float(state["operator_norm"])

    raise ValueError(
        "rank_measure must be either 'linkage' or 'operator'."
    )


def compute_residual_state(
    cut_edges1: Iterable[int] | None,
    cut_edges2: Iterable[int] | None,
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    rank_tol: float = 1e-10,
) -> dict[str, Any]:
    """
    Reconstruct the residual system after cuts on either or both graphs.

    Parameters
    ----------
    cut_edges1
        Edge indices removed from graph 1.
    cut_edges2
        Edge indices removed from graph 2.
    M1_dir
        Directed edge-cycle matrix for graph 1, shape ``(E1, C1)``.
    M2_dir
        Directed edge-cycle matrix for graph 2, shape ``(E2, C2)``.
    lk_mat
        Original cycle-cycle linkage matrix, shape ``(C1, C2)``.
    rank_tol
        Relative singular-value cutoff used for null-space and pseudoinverse
        construction, and absolute cutoff used for numerical-rank evaluation.

    Returns
    -------
    dict
        Dictionary containing:

        ``cut_edges1``
            Sorted tuple of graph-1 cut indices.
        ``cut_edges2``
            Sorted tuple of graph-2 cut indices.
        ``Q1``, ``Q2``
            Surviving cycle-coordinate bases.
        ``M1_residual``, ``M2_residual``
            Residual directed mesh matrices.
        ``lk_residual``
            Reduced cycle-cycle linkage matrix.
        ``Lambda_residual``
            Residual edge-space operator.
        ``linkage_rank``
            Numerical rank of ``lk_residual``.
        ``operator_rank``
            Numerical rank of ``Lambda_residual``.
        ``linkage_norm``
            Frobenius norm of ``lk_residual``.
        ``operator_norm``
            Frobenius norm of ``Lambda_residual``.
        ``rank_agreement``
            Whether the two numerical ranks agree.

    Notes
    -----
    The edge-space operator remains expressed in the original edge coordinate
    spaces. Consequently, cut rows may still be present as coordinates, but
    the surviving cycle combinations satisfy the corresponding constraints.
    """
    if rank_tol < 0:
        raise ValueError("rank_tol must be nonnegative.")

    M1_dir, M2_dir, lk_mat = _validate_cutting_inputs(
        M1_dir,
        M2_dir,
        lk_mat,
    )

    number_of_edges1 = M1_dir.shape[0]
    number_of_edges2 = M2_dir.shape[0]

    cut_edges1 = normalize_cut_indices(
        cut_edges1,
        number_of_edges1,
        name="cut_edges1",
    )

    cut_edges2 = normalize_cut_indices(
        cut_edges2,
        number_of_edges2,
        name="cut_edges2",
    )

    Q1 = _null_space_after_edge_cuts(
        directed_mesh_matrix=M1_dir,
        cut_indices=cut_edges1,
        rank_tol=rank_tol,
    )

    Q2 = _null_space_after_edge_cuts(
        directed_mesh_matrix=M2_dir,
        cut_indices=cut_edges2,
        rank_tol=rank_tol,
    )

    M1_residual = M1_dir @ Q1
    M2_residual = M2_dir @ Q2

    lk_residual = Q1.T @ lk_mat @ Q2

    if lk_residual.size == 0:
        Lambda_residual = np.zeros(
            (M1_dir.shape[0], M2_dir.shape[0]),
            dtype=float,
        )
    else:
        M1_residual_pinv = np.linalg.pinv(
            M1_residual,
            rcond=rank_tol,
        )
    
        M2_residual_pinv = np.linalg.pinv(
            M2_residual,
            rcond=rank_tol,
        )
    
        Lambda_residual = (
            M1_residual_pinv.T
            @ lk_residual
            @ M2_residual_pinv
        )

    linkage_rank = numerical_rank(
        lk_residual,
        tol=rank_tol,
    )

    operator_rank = numerical_rank(
        Lambda_residual,
        tol=rank_tol,
    )

    linkage_norm = float(
        np.linalg.norm(
            lk_residual,
            ord="fro",
        )
    )

    operator_norm = float(
        np.linalg.norm(
            Lambda_residual,
            ord="fro",
        )
    )

    return {
        "cut_edges1": cut_edges1,
        "cut_edges2": cut_edges2,
        "Q1": Q1,
        "Q2": Q2,
        "M1_residual": M1_residual,
        "M2_residual": M2_residual,
        "lk_residual": lk_residual,
        "Lambda_residual": Lambda_residual,
        "linkage_rank": linkage_rank,
        "operator_rank": operator_rank,
        "linkage_norm": linkage_norm,
        "operator_norm": operator_norm,
        "rank_agreement": linkage_rank == operator_rank,

        # Compatibility aliases. These use linkage rank and norm because the
        # reduced linkage matrix directly represents surviving cycle linkage.
        "rank": linkage_rank,
        "norm": linkage_norm,
    }


def compute_one_sided_residual_state(
    cut_edges1: Iterable[int] | None,
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    rank_tol: float = 1e-10,
) -> dict[str, Any]:
    """
    Reconstruct the residual state when only graph 1 is cut.

    This is a convenience wrapper around :func:`compute_residual_state`.
    """
    return compute_residual_state(
        cut_edges1=cut_edges1,
        cut_edges2=(),
        M1_dir=M1_dir,
        M2_dir=M2_dir,
        lk_mat=lk_mat,
        rank_tol=rank_tol,
    )


# ============================================================================ #
# Singular-value grouping and edge scoring
# ============================================================================ #


def group_nonzero_singular_subspaces(
    singular_values: np.ndarray,
    singular_tol: float = 1e-10,
    degeneracy_rtol: float = 1e-8,
) -> list[np.ndarray]:
    """
    Group equal nonzero singular values into distinct singular subspaces.

    Parameters
    ----------
    singular_values
        One-dimensional singular-value array, normally returned in descending
        order by an SVD routine.
    singular_tol
        Values not larger than this threshold are treated as zero.
    degeneracy_rtol
        Relative tolerance used to identify equal singular values.

    Returns
    -------
    list of ndarray
        Each array contains the SVD column indices belonging to one distinct
        nonzero singular-value group. Groups retain descending spectral order.
    """
    singular_values = np.asarray(
        singular_values,
        dtype=float,
    )

    if singular_values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional.")

    nonzero_indices = np.flatnonzero(
        singular_values > singular_tol
    )

    groups: list[np.ndarray] = []
    assigned = np.zeros(
        len(singular_values),
        dtype=bool,
    )

    for singular_index in nonzero_indices:
        if assigned[singular_index]:
            continue

        group_mask = (
            (singular_values > singular_tol)
            & np.isclose(
                singular_values,
                singular_values[singular_index],
                rtol=degeneracy_rtol,
                atol=singular_tol,
            )
        )

        group = np.flatnonzero(group_mask)

        groups.append(group)
        assigned[group] = True

    return groups


def singular_subspace_edge_scores(
    singular_vectors: np.ndarray,
    subspace_indices: Sequence[int],
) -> np.ndarray:
    """
    Compute basis-invariant coordinate scores for a singular subspace.

    For an orthonormal basis ``W`` of the selected subspace, the score assigned
    to coordinate ``e`` is

        score[e] = sum_k |W[e, k]|^2.

    This quantity is invariant under orthogonal or unitary changes of basis
    within a degenerate singular subspace.

    Parameters
    ----------
    singular_vectors
        Matrix whose columns are singular vectors.
    subspace_indices
        Column indices defining the selected singular subspace.

    Returns
    -------
    ndarray
        One score per row coordinate.
    """
    singular_vectors = np.asarray(singular_vectors)

    if singular_vectors.ndim != 2:
        raise ValueError("singular_vectors must be two-dimensional.")

    subspace_indices = np.asarray(
        subspace_indices,
        dtype=int,
    )

    if subspace_indices.ndim != 1:
        raise ValueError("subspace_indices must be one-dimensional.")

    if len(subspace_indices) == 0:
        return np.zeros(
            singular_vectors.shape[0],
            dtype=float,
        )

    if (
        np.min(subspace_indices) < 0
        or np.max(subspace_indices) >= singular_vectors.shape[1]
    ):
        raise IndexError(
            "At least one subspace index is outside the singular-vector "
            "column range."
        )

    selected_vectors = singular_vectors[
        :,
        subspace_indices,
    ]

    return np.sum(
        np.abs(selected_vectors) ** 2,
        axis=1,
    ).real


def principal_edge_scores(
    Lambda_residual: np.ndarray,
    side: Literal[1, 2] = 1,
    singular_tol: float = 1e-10,
    degeneracy_rtol: float = 1e-7,
) -> np.ndarray:
    """
    Score edges from the complete leading singular subspace of an operator.

    Parameters
    ----------
    Lambda_residual
        Residual edge-space operator, shape ``(E1, E2)``.
    side
        ``1`` scores graph-1 edges using leading left singular vectors.
        ``2`` scores graph-2 edges using leading right singular vectors.
    singular_tol
        Threshold below which singular values are treated as zero.
    degeneracy_rtol
        Relative tolerance used to identify degeneracy of the leading
        singular value.

    Returns
    -------
    ndarray
        One score for each edge on the selected graph.
    """
    Lambda_residual = np.asarray(
        Lambda_residual,
        dtype=float,
    )

    if Lambda_residual.ndim != 2:
        raise ValueError("Lambda_residual must be two-dimensional.")

    if side not in (1, 2):
        raise ValueError("side must be either 1 or 2.")

    number_of_edges = Lambda_residual.shape[
        0 if side == 1 else 1
    ]

    if Lambda_residual.size == 0:
        return np.zeros(
            number_of_edges,
            dtype=float,
        )

    U, singular_values, Vh = np.linalg.svd(
        Lambda_residual,
        full_matrices=False,
    )

    nonzero_indices = np.flatnonzero(
        singular_values > singular_tol
    )

    if len(nonzero_indices) == 0:
        return np.zeros(
            number_of_edges,
            dtype=float,
        )

    leading_singular_value = singular_values[
        nonzero_indices[0]
    ]

    leading_indices = nonzero_indices[
        np.isclose(
            singular_values[nonzero_indices],
            leading_singular_value,
            rtol=degeneracy_rtol,
            atol=singular_tol,
        )
    ]

    if side == 1:
        singular_vectors = U
    else:
        # Right singular vectors are stored as rows of Vh.
        singular_vectors = Vh.T

    return singular_subspace_edge_scores(
        singular_vectors=singular_vectors,
        subspace_indices=leading_indices,
    )


def _edge_label(
    edge_index: int,
    edge_set: Sequence[Any] | None,
) -> Any:
    """Return a graph-edge label when available, otherwise its index."""
    if edge_set is None:
        return edge_index

    return edge_set[edge_index]


# ============================================================================ #
# Recomputed-SVD cutting on graph 1
# ============================================================================ #


def find_recomputed_single_candidates(
    current_state: Mapping[str, Any],
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    edge_set1: Sequence[Any] | None = None,
    rank_tol: float = 1e-10,
    rank_measure: RankMeasure = "linkage",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Exhaustively test every currently uncut graph-1 edge.

    Scores are computed from the leading left-singular subspace of the current
    residual edge-space operator. Every remaining edge is nevertheless tested;
    the spectral score controls ordering and tie-breaking, not admissibility.

    Parameters
    ----------
    current_state
        Residual state returned by :func:`compute_residual_state`.
    M1_dir, M2_dir, lk_mat
        Original system matrices.
    edge_set1
        Optional graph-1 edge labels in the same order as rows of ``M1_dir``.
    rank_tol
        Numerical tolerance.
    rank_measure
        ``"linkage"`` compares ranks of residual linkage matrices.
        ``"operator"`` compares ranks of residual edge-space operators.

    Returns
    -------
    candidates
        Rank-reducing candidates sorted by:

        1. largest rank drop;
        2. largest principal-subspace score;
        3. smallest corresponding residual norm;
        4. smallest edge index.

    candidate_log
        One record for every tested edge, including rejected edges.
    """
    current_cuts1 = set(current_state["cut_edges1"])
    current_cuts2 = current_state["cut_edges2"]
    current_rank = _state_rank(
        current_state,
        rank_measure,
    )

    scores = principal_edge_scores(
        current_state["Lambda_residual"],
        side=1,
        singular_tol=rank_tol,
    )

    remaining_edges = [
        edge_index
        for edge_index in range(M1_dir.shape[0])
        if edge_index not in current_cuts1
    ]

    remaining_edges.sort(
        key=lambda edge_index: (
            -scores[edge_index],
            edge_index,
        )
    )

    candidates: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []

    for score_rank, edge_index in enumerate(
        remaining_edges,
        start=1,
    ):
        trial_state = compute_residual_state(
            cut_edges1=current_cuts1 | {edge_index},
            cut_edges2=current_cuts2,
            M1_dir=M1_dir,
            M2_dir=M2_dir,
            lk_mat=lk_mat,
            rank_tol=rank_tol,
        )

        rank_after = _state_rank(
            trial_state,
            rank_measure,
        )

        residual_norm = _state_norm(
            trial_state,
            rank_measure,
        )

        rank_drop = current_rank - rank_after

        record = {
            "side": 1,
            "edge_idx": int(edge_index),
            "edge": _edge_label(
                edge_index,
                edge_set1,
            ),
            "score": float(scores[edge_index]),
            "score_rank": int(score_rank),
            "rank_before": int(current_rank),
            "rank_after": int(rank_after),
            "rank_drop": int(rank_drop),
            "rank_reducing": bool(rank_drop > 0),
            "selected": False,
            "residual_norm": float(residual_norm),
            "rank_measure": rank_measure,
        }

        candidate_log.append(record)

        if rank_drop > 0:
            candidates.append(
                {
                    **record,
                    "edges": (int(edge_index),),
                    "state": trial_state,
                }
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate["rank_drop"],
            -candidate["score"],
            candidate["residual_norm"],
            candidate["edge_idx"],
        )
    )

    return candidates, candidate_log


def find_recomputed_pair_candidates(
    current_state: Mapping[str, Any],
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    edge_set1: Sequence[Any] | None = None,
    rank_tol: float = 1e-10,
    rank_measure: RankMeasure = "linkage",
    pair_pool_size: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search graph-1 edge pairs using the current residual leading subspace.

    Parameters
    ----------
    current_state
        Current residual system.
    M1_dir, M2_dir, lk_mat
        Original system matrices.
    edge_set1
        Optional graph-1 edge labels.
    rank_tol
        Numerical tolerance.
    rank_measure
        Residual rank definition used to determine successful cuts.
    pair_pool_size
        If ``None``, all remaining edge pairs are tested.

        If an integer is supplied, only pairs among the highest-scoring
        ``pair_pool_size`` remaining edges are tested. This can substantially
        reduce runtime, but the pair search is then no longer exhaustive.

    Returns
    -------
    list of dict
        Rank-reducing pairs sorted by rank drop, score, residual norm, and
        lexicographic edge index.
    """
    current_cuts1 = set(current_state["cut_edges1"])
    current_cuts2 = current_state["cut_edges2"]

    current_rank = _state_rank(
        current_state,
        rank_measure,
    )

    scores = principal_edge_scores(
        current_state["Lambda_residual"],
        side=1,
        singular_tol=rank_tol,
    )

    remaining_edges = [
        edge_index
        for edge_index in range(M1_dir.shape[0])
        if edge_index not in current_cuts1
    ]

    remaining_edges.sort(
        key=lambda edge_index: (
            -scores[edge_index],
            edge_index,
        )
    )

    if pair_pool_size is not None:
        pair_pool_size = int(pair_pool_size)

        if pair_pool_size < 2:
            raise ValueError(
                "pair_pool_size must be at least 2 or None."
            )

        remaining_edges = remaining_edges[
            :min(pair_pool_size, len(remaining_edges))
        ]

    candidates: list[dict[str, Any]] = []

    for edge_i, edge_j in combinations(
        remaining_edges,
        2,
    ):
        trial_state = compute_residual_state(
            cut_edges1=current_cuts1 | {edge_i, edge_j},
            cut_edges2=current_cuts2,
            M1_dir=M1_dir,
            M2_dir=M2_dir,
            lk_mat=lk_mat,
            rank_tol=rank_tol,
        )

        rank_after = _state_rank(
            trial_state,
            rank_measure,
        )

        rank_drop = current_rank - rank_after

        if rank_drop <= 0:
            continue

        residual_norm = _state_norm(
            trial_state,
            rank_measure,
        )

        pair_score = (
            scores[edge_i]
            + scores[edge_j]
        )

        candidates.append(
            {
                "edges": (
                    int(edge_i),
                    int(edge_j),
                ),
                "edge_labels": (
                    _edge_label(edge_i, edge_set1),
                    _edge_label(edge_j, edge_set1),
                ),
                "score_i": float(scores[edge_i]),
                "score_j": float(scores[edge_j]),
                "score": float(pair_score),
                "rank_before": int(current_rank),
                "rank_after": int(rank_after),
                "rank_drop": int(rank_drop),
                "residual_norm": float(residual_norm),
                "rank_measure": rank_measure,
                "state": trial_state,
            }
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate["rank_drop"],
            -candidate["score"],
            candidate["residual_norm"],
            candidate["edges"],
        )
    )

    return candidates


def cut_with_recomputed_svd(
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    edge_set1: Sequence[Any] | None = None,
    initial_cut_edges1: Iterable[int] | None = None,
    rank_tol: float = 1e-10,
    rank_measure: RankMeasure = "linkage",
    allow_pair_cuts: bool = True,
    pair_pool_size: int | None = None,
    verbose: bool = True,
    print_rejected_singles: bool = False,
) -> dict[str, Any]:
    """
    Cut graph-1 edges while recomputing the residual SVD after every cut.

    Algorithm
    ---------
    At each iteration:

    1. Reconstruct the residual cycle spaces and residual operator.
    2. Compute edge scores from the complete leading left-singular subspace.
    3. Exhaustively test all remaining single edges.
    4. Select the best rank-reducing single edge.
    5. If no single edge works and ``allow_pairs=True``, test edge pairs.
    6. After accepting a pair, return to single-edge testing.
    7. Stop at rank zero or when no permitted cut reduces rank.

    Parameters
    ----------
    M1_dir, M2_dir, lk_mat
        Original system matrices.
    edge_set1
        Optional graph-1 edge labels.
    initial_cut_edges1
        Optional pre-existing graph-1 cuts.
    rank_tol
        Numerical tolerance.
    rank_measure
        ``"linkage"`` or ``"operator"``.
    allow_pair_cuts
        Whether pair cuts are attempted after single-edge irreducibility.
    pair_pool_size
        Optional restriction on the pair-search pool.
    verbose
        Print iteration details.
    print_rejected_singles
        Print every rejected single-edge test.

    Returns
    -------
    dict
        Final state, complete cut set, accepted-cut history, candidate log,
        initial ranks, and success flag.
    """
    M1_dir, M2_dir, lk_mat = _validate_cutting_inputs(
        M1_dir,
        M2_dir,
        lk_mat,
    )

    if edge_set1 is not None and len(edge_set1) != M1_dir.shape[0]:
        raise ValueError(
            "edge_set1 must have one entry per row of M1_dir."
        )

    initial_state = compute_one_sided_residual_state(
        cut_edges1=initial_cut_edges1,
        M1_dir=M1_dir,
        M2_dir=M2_dir,
        lk_mat=lk_mat,
        rank_tol=rank_tol,
    )

    current_state = initial_state
    initial_rank = _state_rank(
        initial_state,
        rank_measure,
    )

    history: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []
    iteration = 0
    stopping_reason = ""

    if verbose:
        print("Initial state")
        print(f"  rank measure: {rank_measure}")
        print(f"  residual rank: {initial_rank}")
        print(
            "  residual norm: "
            f"{_state_norm(current_state, rank_measure):.8e}"
        )
        print(
            "  residual linkage shape: "
            f"{current_state['lk_residual'].shape}"
        )
        print(
            "  residual edge-operator shape: "
            f"{current_state['Lambda_residual'].shape}"
        )
        print(
            "  linkage/operator ranks: "
            f"{current_state['linkage_rank']} / "
            f"{current_state['operator_rank']}"
        )

    while _state_rank(current_state, rank_measure) > 0:
        iteration += 1

        rank_before = _state_rank(
            current_state,
            rank_measure,
        )

        single_candidates, iteration_log = (
            find_recomputed_single_candidates(
                current_state=current_state,
                M1_dir=M1_dir,
                M2_dir=M2_dir,
                lk_mat=lk_mat,
                edge_set1=edge_set1,
                rank_tol=rank_tol,
                rank_measure=rank_measure,
            )
        )

        for row in iteration_log:
            row["iteration"] = iteration

        if single_candidates:
            best = single_candidates[0]
            chosen_edge = best["edge_idx"]

            for row in iteration_log:
                if row["edge_idx"] == chosen_edge:
                    row["selected"] = True
                    break

            candidate_log.extend(iteration_log)
            current_state = best["state"]

            number_tested = len(iteration_log)
            number_rank_reducing = len(single_candidates)
            number_rejected = (
                number_tested
                - number_rank_reducing
            )

            history.append(
                {
                    "iteration": iteration,
                    "cut_type": "single",
                    "edges": (chosen_edge,),
                    "edge_labels": (
                        _edge_label(chosen_edge, edge_set1),
                    ),
                    "rank_before": rank_before,
                    "rank_after": best["rank_after"],
                    "rank_drop": best["rank_drop"],
                    "score": best["score"],
                    "score_rank": best["score_rank"],
                    "residual_norm": best["residual_norm"],
                    "number_tested": number_tested,
                    "number_rank_reducing": number_rank_reducing,
                    "number_rejected": number_rejected,
                    "rank_measure": rank_measure,
                }
            )

            if verbose:
                higher_scoring_rejections = [
                    row
                    for row in iteration_log
                    if (
                        row["score_rank"] < best["score_rank"]
                        and not row["rank_reducing"]
                    )
                ]

                print(
                    f"\nIteration {iteration}: single cut"
                )
                print(
                    f"  selected edge {chosen_edge}: "
                    f"{_edge_label(chosen_edge, edge_set1)}"
                )
                print(f"  score: {best['score']:.8f}")
                print(
                    f"  score rank: {best['score_rank']} "
                    f"of {number_tested}"
                )
                print(
                    f"  rank: {rank_before} "
                    f"-> {best['rank_after']}"
                )
                print(
                    "  residual norm: "
                    f"{best['residual_norm']:.8e}"
                )
                print(
                    "  rank-reducing single edges: "
                    f"{number_rank_reducing}"
                )
                print(
                    "  rejected single edges: "
                    f"{number_rejected}"
                )
                print(
                    "  higher-scoring rejected edges: "
                    f"{len(higher_scoring_rejections)}"
                )

                if print_rejected_singles:
                    rejected_rows = [
                        row
                        for row in iteration_log
                        if not row["rank_reducing"]
                    ]

                    if rejected_rows:
                        print("  rejected single-edge tests:")

                    for row in rejected_rows:
                        print(
                            "    "
                            f"score rank {row['score_rank']}: "
                            f"edge {row['edge_idx']} "
                            f"{row['edge']}, "
                            f"score={row['score']:.8f}, "
                            f"rank {row['rank_before']} "
                            f"-> {row['rank_after']}"
                        )

            continue

        candidate_log.extend(iteration_log)

        if verbose:
            print(
                f"\nIteration {iteration}: "
                "no rank-reducing single edge"
            )
            print(
                f"  tested {len(iteration_log)} remaining edges"
            )

            if print_rejected_singles:
                for row in iteration_log:
                    print(
                        "    "
                        f"score rank {row['score_rank']}: "
                        f"edge {row['edge_idx']} "
                        f"{row['edge']}, "
                        f"score={row['score']:.8f}, "
                        f"rank {row['rank_before']} "
                        f"-> {row['rank_after']}"
                    )

        if not allow_pair_cuts:
            stopping_reason = (
                "No rank-reducing single edge remained and pair cuts "
                "were disabled."
            )
            break

        if verbose:
            print("  searching for edge pairs...", flush=True)

        pair_candidates = find_recomputed_pair_candidates(
            current_state=current_state,
            M1_dir=M1_dir,
            M2_dir=M2_dir,
            lk_mat=lk_mat,
            edge_set1=edge_set1,
            rank_tol=rank_tol,
            rank_measure=rank_measure,
            pair_pool_size=pair_pool_size,
        )

        if not pair_candidates:
            stopping_reason = (
                "Neither a single edge nor an allowed edge pair reduced "
                "the residual rank."
            )
            break

        best = pair_candidates[0]
        current_state = best["state"]

        history.append(
            {
                "iteration": iteration,
                "cut_type": "pair",
                "edges": best["edges"],
                "edge_labels": best["edge_labels"],
                "rank_before": rank_before,
                "rank_after": best["rank_after"],
                "rank_drop": best["rank_drop"],
                "score": best["score"],
                "score_i": best["score_i"],
                "score_j": best["score_j"],
                "residual_norm": best["residual_norm"],
                "number_of_candidates": len(pair_candidates),
                "rank_measure": rank_measure,
            }
        )

        if verbose:
            print(
                f"  selected pair: {best['edges']}"
            )
            print(
                f"  graph edges: {best['edge_labels']}"
            )
            print(
                "  individual scores: "
                f"{best['score_i']:.8f}, "
                f"{best['score_j']:.8f}"
            )
            print(
                f"  pair score: {best['score']:.8f}"
            )
            print(
                f"  rank: {rank_before} "
                f"-> {best['rank_after']}"
            )
            print(
                "  residual norm: "
                f"{best['residual_norm']:.8e}"
            )

    final_rank = _state_rank(
        current_state,
        rank_measure,
    )

    success = final_rank == 0

    if success:
        stopping_reason = "Residual rank reached zero."

    if verbose:
        if success:
            print("\nSuccess: residual rank reduced to zero.")
        else:
            print(f"\nStopped: {stopping_reason}")
            print(f"Remaining residual rank: {final_rank}")

    return {
        "method": "recomputed_svd",
        "rank_measure": rank_measure,
        "initial_state": initial_state,
        "final_state": current_state,
        "initial_rank": initial_rank,
        "final_rank": final_rank,
        "cut_edges1": list(current_state["cut_edges1"]),
        "cut_edges2": list(current_state["cut_edges2"]),
        "history": history,
        "candidate_log": candidate_log,
        "success": success,
        "stopping_reason": stopping_reason,
        "used_pair_cuts": any(
            row["cut_type"] == "pair"
            for row in history
        ),
    }


# ============================================================================ #
# Fixed-original-SVD cutting on graph 1
# ============================================================================ #


def _find_fixed_score_single_candidates(
    current_state: Mapping[str, Any],
    scores: np.ndarray,
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    edge_set1: Sequence[Any] | None,
    rank_tol: float,
    rank_measure: RankMeasure,
    score_tol: float,
) -> list[dict[str, Any]]:
    """
    Find single-edge cuts using a fixed score vector.

    Only edges with score larger than ``score_tol`` are considered.
    """
    current_cuts1 = set(current_state["cut_edges1"])
    current_cuts2 = current_state["cut_edges2"]

    current_rank = _state_rank(
        current_state,
        rank_measure,
    )

    available_edges = [
        edge_index
        for edge_index in range(M1_dir.shape[0])
        if (
            edge_index not in current_cuts1
            and scores[edge_index] > score_tol
        )
    ]

    candidates: list[dict[str, Any]] = []

    for edge_index in available_edges:
        trial_state = compute_residual_state(
            cut_edges1=current_cuts1 | {edge_index},
            cut_edges2=current_cuts2,
            M1_dir=M1_dir,
            M2_dir=M2_dir,
            lk_mat=lk_mat,
            rank_tol=rank_tol,
        )

        rank_after = _state_rank(
            trial_state,
            rank_measure,
        )

        rank_drop = current_rank - rank_after

        if rank_drop <= 0:
            continue

        candidates.append(
            {
                "edges": (int(edge_index),),
                "edge_labels": (
                    _edge_label(edge_index, edge_set1),
                ),
                "score": float(scores[edge_index]),
                "rank_before": int(current_rank),
                "rank_after": int(rank_after),
                "rank_drop": int(rank_drop),
                "residual_norm": _state_norm(
                    trial_state,
                    rank_measure,
                ),
                "state": trial_state,
            }
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate["rank_drop"],
            -candidate["score"],
            candidate["residual_norm"],
            candidate["edges"],
        )
    )

    return candidates


def find_original_svd_pair_candidates(
    current_state: Mapping[str, Any],
    scores: np.ndarray,
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    edge_set1: Sequence[Any] | None = None,
    rank_tol: float = 1e-10,
    rank_measure: RankMeasure = "linkage",
    score_tol: float = 1e-12,
    pair_pool_size: int | None = None,
    include_spectral_rank: bool = False,
) -> list[dict[str, Any]]:
    """
    Find rank-reducing pairs using scores from a fixed original subspace.

    Parameters
    ----------
    current_state
        Current residual state.
    scores
        Fixed graph-1 edge scores from one original singular subspace.
    M1_dir, M2_dir, lk_mat
        Original system matrices.
    edge_set1
        Optional graph-1 edge labels.
    rank_tol
        Numerical tolerance.
    rank_measure
        Rank definition used to determine success.
    score_tol
        Edges with scores not larger than this value are excluded.
    pair_pool_size
        Optional top-score restriction.
    include_spectral_rank
        If ``True``, each successful pair is assigned its score rank among
        every considered pair.

    Returns
    -------
    list of dict
        Sorted successful pairs.
    """
    scores = np.asarray(
        scores,
        dtype=float,
    )

    if scores.shape != (M1_dir.shape[0],):
        raise ValueError(
            "scores must contain one value per graph-1 edge."
        )

    current_cuts1 = set(current_state["cut_edges1"])
    current_cuts2 = current_state["cut_edges2"]

    current_rank = _state_rank(
        current_state,
        rank_measure,
    )

    available_edges = [
        edge_index
        for edge_index in range(M1_dir.shape[0])
        if (
            edge_index not in current_cuts1
            and scores[edge_index] > score_tol
        )
    ]

    available_edges.sort(
        key=lambda edge_index: (
            -scores[edge_index],
            edge_index,
        )
    )

    if pair_pool_size is not None:
        pair_pool_size = int(pair_pool_size)

        if pair_pool_size < 2:
            raise ValueError(
                "pair_pool_size must be at least 2 or None."
            )

        available_edges = available_edges[
            :min(pair_pool_size, len(available_edges))
        ]

    all_pairs = list(
        combinations(
            available_edges,
            2,
        )
    )

    spectral_rank_lookup: dict[
        tuple[int, int],
        int,
    ] = {}

    if include_spectral_rank:
        ranked_pairs = sorted(
            all_pairs,
            key=lambda pair: (
                -(
                    scores[pair[0]]
                    + scores[pair[1]]
                ),
                pair,
            ),
        )

        spectral_rank_lookup = {
            pair: rank
            for rank, pair in enumerate(
                ranked_pairs,
                start=1,
            )
        }

    candidates: list[dict[str, Any]] = []

    for edge_i, edge_j in all_pairs:
        trial_state = compute_residual_state(
            cut_edges1=current_cuts1 | {edge_i, edge_j},
            cut_edges2=current_cuts2,
            M1_dir=M1_dir,
            M2_dir=M2_dir,
            lk_mat=lk_mat,
            rank_tol=rank_tol,
        )

        rank_after = _state_rank(
            trial_state,
            rank_measure,
        )

        rank_drop = current_rank - rank_after

        if rank_drop <= 0:
            continue

        pair_score = (
            scores[edge_i]
            + scores[edge_j]
        )

        record = {
            "edges": (
                int(edge_i),
                int(edge_j),
            ),
            "edge_labels": (
                _edge_label(edge_i, edge_set1),
                _edge_label(edge_j, edge_set1),
            ),
            "score_i": float(scores[edge_i]),
            "score_j": float(scores[edge_j]),
            "score": float(pair_score),
            "rank_before": int(current_rank),
            "rank_after": int(rank_after),
            "rank_drop": int(rank_drop),
            "residual_norm": _state_norm(
                trial_state,
                rank_measure,
            ),
            "state": trial_state,
        }

        if include_spectral_rank:
            record["spectral_rank"] = spectral_rank_lookup[
                (edge_i, edge_j)
            ]
            record["number_of_ranked_pairs"] = len(all_pairs)

        candidates.append(record)

    candidates.sort(
        key=lambda candidate: (
            -candidate["rank_drop"],
            -candidate["score"],
            candidate["residual_norm"],
            candidate["edges"],
        )
    )

    return candidates


def cut_with_original_svd(
    Lambda_original: np.ndarray,
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    edge_set1: Sequence[Any] | None = None,
    initial_cut_edges1: Iterable[int] | None = None,
    subspace_numbers: Sequence[int] | None = (1, 2),
    rank_tol: float = 1e-10,
    score_tol: float = 1e-12,
    degeneracy_rtol: float = 1e-8,
    rank_measure: RankMeasure = "linkage",
    allow_single_cuts: bool = True,
    allow_pair_cuts: bool = True,
    pair_pool_size: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Cut graph-1 edges using singular subspaces of the original operator.

    The SVD of ``Lambda_original`` is computed exactly once. Scores do not change
    after cuts. For each requested distinct singular subspace, the routine
    repeatedly accepts rank-reducing single edges and then rank-reducing pairs.

    Parameters
    ----------
    Lambda_original
        Original edge-space operator whose left singular vectors score graph-1
        edges.
    M1_dir, M2_dir, lk_mat
        Original system matrices used to reconstruct residual states.
    edge_set1
        Optional graph-1 edge labels.
    initial_cut_edges1
        Optional cuts present before this routine begins.
    subspace_numbers
        One-based distinct singular-subspace numbers to process.

        Examples:

        ``(1,)``
            Leading original singular subspace only.

        ``(1, 2)``
            Leading and second distinct original singular subspaces.

        ``None``
            Process every nonzero distinct singular subspace.
    rank_tol
        Numerical tolerance.
    score_tol
        Minimum non-negligible edge score.
    degeneracy_rtol
        Relative tolerance for grouping equal singular values.
    rank_measure
        ``"linkage"`` or ``"operator"``.
    allow_single_cuts
        Whether individual edge cuts are attempted.
    allow_pair_cuts
        Whether edge pairs are attempted after single-edge irreducibility in
        each selected subspace.
    pair_pool_size
        Optional pair-search restriction.
    verbose
        Print progress.

    Returns
    -------
    dict
        Complete cut set, cuts grouped by original singular subspace, original
        SVD information, accepted history, and final residual state.
    """
    M1_dir, M2_dir, lk_mat = _validate_cutting_inputs(
        M1_dir,
        M2_dir,
        lk_mat,
    )

    Lambda_original = np.asarray(
        Lambda_original,
        dtype=float,
    )

    expected_operator_shape = (
        M1_dir.shape[0],
        M2_dir.shape[0],
    )

    if Lambda_original.shape != expected_operator_shape:
        raise ValueError(
            "Lambda_original must have shape "
            f"{expected_operator_shape}; got {Lambda_original.shape}."
        )

    if edge_set1 is not None and len(edge_set1) != M1_dir.shape[0]:
        raise ValueError(
            "edge_set1 must have one entry per row of M1_dir."
        )

    U_original, singular_values, Vh_original = np.linalg.svd(
        Lambda_original,
        full_matrices=False,
    )

    singular_groups = group_nonzero_singular_subspaces(
        singular_values=singular_values,
        singular_tol=rank_tol,
        degeneracy_rtol=degeneracy_rtol,
    )

    if subspace_numbers is None:
        selected_subspace_numbers = tuple(
            range(1, len(singular_groups) + 1)
        )
    else:
        selected_subspace_numbers = tuple(
            int(number)
            for number in subspace_numbers
        )

    invalid_subspaces = [
        number
        for number in selected_subspace_numbers
        if number < 1 or number > len(singular_groups)
    ]

    if invalid_subspaces:
        raise IndexError(
            f"Invalid distinct subspace numbers {invalid_subspaces}. "
            f"The original operator has {len(singular_groups)} "
            "nonzero distinct singular-value groups."
        )

    initial_state = compute_one_sided_residual_state(
        cut_edges1=initial_cut_edges1,
        M1_dir=M1_dir,
        M2_dir=M2_dir,
        lk_mat=lk_mat,
        rank_tol=rank_tol,
    )

    current_state = initial_state

    history: list[dict[str, Any]] = []
    subspace_results: list[dict[str, Any]] = []
    iteration = 0

    if verbose:
        print(
            "Number of nonzero singular values: "
            f"{np.count_nonzero(singular_values > rank_tol)}"
        )
        print("\nDistinct nonzero singular-value groups:")

        for group_number, group in enumerate(
            singular_groups,
            start=1,
        ):
            print(
                f"  group {group_number}: "
                f"sigma={singular_values[group[0]]:.8f}, "
                f"multiplicity={len(group)}"
            )

    for subspace_number in selected_subspace_numbers:
        if _state_rank(current_state, rank_measure) == 0:
            break

        group = singular_groups[
            subspace_number - 1
        ]

        scores = singular_subspace_edge_scores(
            singular_vectors=U_original,
            subspace_indices=group,
        )

        accepted_single_edges: list[int] = []
        accepted_pairs: list[tuple[int, int]] = []

        if verbose:
            print(
                f"\nOriginal singular subspace {subspace_number}"
            )
            print(
                "  singular value: "
                f"{singular_values[group[0]]:.8f}"
            )
            print(f"  multiplicity: {len(group)}")
            print(
                "  starting residual rank: "
                f"{_state_rank(current_state, rank_measure)}"
            )
            print(
                "  starting residual norm: "
                f"{_state_norm(current_state, rank_measure):.8e}"
            )

        if allow_single_cuts:
            while _state_rank(current_state, rank_measure) > 0:
                candidates = _find_fixed_score_single_candidates(
                    current_state=current_state,
                    scores=scores,
                    M1_dir=M1_dir,
                    M2_dir=M2_dir,
                    lk_mat=lk_mat,
                    edge_set1=edge_set1,
                    rank_tol=rank_tol,
                    rank_measure=rank_measure,
                    score_tol=score_tol,
                )

                if not candidates:
                    if verbose:
                        print(
                            "  no remaining single edge from this "
                            "subspace reduces rank"
                        )
                    break

                best = candidates[0]
                iteration += 1
                current_state = best["state"]

                edge_index = best["edges"][0]
                accepted_single_edges.append(edge_index)

                history.append(
                    {
                        "iteration": iteration,
                        "method": "original_svd",
                        "subspace_number": subspace_number,
                        "singular_value": float(
                            singular_values[group[0]]
                        ),
                        "multiplicity": len(group),
                        "cut_type": "single",
                        "edges": best["edges"],
                        "edge_labels": best["edge_labels"],
                        "score": best["score"],
                        "rank_before": best["rank_before"],
                        "rank_after": best["rank_after"],
                        "rank_drop": best["rank_drop"],
                        "residual_norm": best["residual_norm"],
                    }
                )

                if verbose:
                    print(
                        f"  cut edge {edge_index} "
                        f"{best['edge_labels'][0]}, "
                        f"score={best['score']:.8g}, "
                        f"rank {best['rank_before']} "
                        f"-> {best['rank_after']}, "
                        f"norm={best['residual_norm']:.8e}"
                    )

        if (
            allow_pair_cuts
            and _state_rank(current_state, rank_measure) > 0
        ):
            while _state_rank(current_state, rank_measure) > 0:
                if verbose:
                    print("Searching for edge pairs...", flush=True)

                pair_candidates = (
                    find_original_svd_pair_candidates(
                        current_state=current_state,
                        scores=scores,
                        M1_dir=M1_dir,
                        M2_dir=M2_dir,
                        lk_mat=lk_mat,
                        edge_set1=edge_set1,
                        rank_tol=rank_tol,
                        rank_measure=rank_measure,
                        score_tol=score_tol,
                        pair_pool_size=pair_pool_size,
                    )
                )

                if not pair_candidates:
                    if verbose:
                        print(
                            "  no remaining pair from this subspace "
                            "reduces rank"
                        )
                    break

                best = pair_candidates[0]
                iteration += 1
                current_state = best["state"]

                accepted_pairs.append(
                    best["edges"]
                )

                history.append(
                    {
                        "iteration": iteration,
                        "method": "original_svd",
                        "subspace_number": subspace_number,
                        "singular_value": float(
                            singular_values[group[0]]
                        ),
                        "multiplicity": len(group),
                        "cut_type": "pair",
                        "edges": best["edges"],
                        "edge_labels": best["edge_labels"],
                        "score": best["score"],
                        "score_i": best["score_i"],
                        "score_j": best["score_j"],
                        "rank_before": best["rank_before"],
                        "rank_after": best["rank_after"],
                        "rank_drop": best["rank_drop"],
                        "residual_norm": best["residual_norm"],
                    }
                )

                if verbose:
                    print(
                        f"  cut pair {best['edges']} "
                        f"{best['edge_labels']}, "
                        f"score={best['score']:.8g}, "
                        f"rank {best['rank_before']} "
                        f"-> {best['rank_after']}, "
                        f"norm={best['residual_norm']:.8e}"
                    )

                # Return to single-edge testing within the same fixed
                # original subspace. A pair cut can make new singles viable.
                if allow_single_cuts:
                    while _state_rank(
                        current_state,
                        rank_measure,
                    ) > 0:
                        new_single_candidates = (
                            _find_fixed_score_single_candidates(
                                current_state=current_state,
                                scores=scores,
                                M1_dir=M1_dir,
                                M2_dir=M2_dir,
                                lk_mat=lk_mat,
                                edge_set1=edge_set1,
                                rank_tol=rank_tol,
                                rank_measure=rank_measure,
                                score_tol=score_tol,
                            )
                        )

                        if not new_single_candidates:
                            break

                        new_best = new_single_candidates[0]
                        iteration += 1
                        current_state = new_best["state"]

                        edge_index = new_best["edges"][0]
                        accepted_single_edges.append(edge_index)

                        history.append(
                            {
                                "iteration": iteration,
                                "method": "original_svd",
                                "subspace_number": subspace_number,
                                "singular_value": float(
                                    singular_values[group[0]]
                                ),
                                "multiplicity": len(group),
                                "cut_type": "single",
                                "edges": new_best["edges"],
                                "edge_labels": new_best[
                                    "edge_labels"
                                ],
                                "score": new_best["score"],
                                "rank_before": new_best[
                                    "rank_before"
                                ],
                                "rank_after": new_best[
                                    "rank_after"
                                ],
                                "rank_drop": new_best[
                                    "rank_drop"
                                ],
                                "residual_norm": new_best[
                                    "residual_norm"
                                ],
                            }
                        )

                        if verbose:
                            print(
                                f"  cut edge {edge_index} "
                                f"{new_best['edge_labels'][0]}, "
                                f"score={new_best['score']:.8g}, "
                                f"rank {new_best['rank_before']} "
                                f"-> {new_best['rank_after']}, "
                                f"norm="
                                f"{new_best['residual_norm']:.8e}"
                            )

        subspace_results.append(
            {
                "subspace_number": subspace_number,
                "singular_value": float(
                    singular_values[group[0]]
                ),
                "multiplicity": len(group),
                "svd_indices": group.copy(),
                "scores": scores,
                "single_cuts": accepted_single_edges,
                "pair_cuts": accepted_pairs,
                "pair_cut_edges": [
                    edge_index
                    for pair in accepted_pairs
                    for edge_index in pair
                ],
                "all_edges_added": (
                    accepted_single_edges
                    + [
                        edge_index
                        for pair in accepted_pairs
                        for edge_index in pair
                    ]
                ),
            }
        )

    final_rank = _state_rank(
        current_state,
        rank_measure,
    )

    result = {
        "method": "original_svd",
        "rank_measure": rank_measure,
        "Lambda_original": Lambda_original,
        "U_original": U_original,
        "singular_values": singular_values,
        "Vh_original": Vh_original,
        "singular_groups": singular_groups,
        "selected_subspace_numbers": selected_subspace_numbers,
        "initial_state": initial_state,
        "final_state": current_state,
        "initial_rank": _state_rank(
            initial_state,
            rank_measure,
        ),
        "final_rank": final_rank,
        "cut_edges1": list(current_state["cut_edges1"]),
        "cut_edges2": list(current_state["cut_edges2"]),
        "history": history,
        "subspace_results": subspace_results,
        "success": final_rank == 0,
    }

    # Convenience fields for the commonly used first and second subspaces.
    for subspace_result in subspace_results:
        number = subspace_result["subspace_number"]

        if number == 1:
            result["leading_single_cuts"] = subspace_result[
                "single_cuts"
            ]
            result["leading_pair_cuts"] = subspace_result[
                "pair_cuts"
            ]

        elif number == 2:
            result["second_single_cuts"] = subspace_result[
                "single_cuts"
            ]
            result["second_pair_cuts"] = subspace_result[
                "pair_cuts"
            ]

    result.setdefault(
        "leading_single_cuts",
        [],
    )
    result.setdefault(
        "leading_pair_cuts",
        [],
    )
    result.setdefault(
        "second_single_cuts",
        [],
    )
    result.setdefault(
        "second_pair_cuts",
        [],
    )

    if verbose:
        print("\nFinal result")
        print(f"  success: {result['success']}")
        print(f"  complete cut set: {result['cut_edges1']}")
        print(
            f"  total number of cuts: "
            f"{len(result['cut_edges1'])}"
        )
        print(f"  initial rank: {result['initial_rank']}")
        print(f"  final residual rank: {result['final_rank']}")
        print(
            "  final residual norm: "
            f"{_state_norm(current_state, rank_measure):.8e}"
        )

    return result


# ============================================================================ #
# Recomputed-SVD cutting on both graphs
# ============================================================================ #


def find_two_sided_single_candidates(
    current_state: Mapping[str, Any],
    side: Literal[1, 2],
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    edge_set: Sequence[Any] | None = None,
    rank_tol: float = 1e-10,
    rank_measure: RankMeasure = "linkage",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Exhaustively test all currently uncut edges on one graph.

    The current leading left singular subspace scores graph 1. The current
    leading right singular subspace scores graph 2.

    Parameters
    ----------
    current_state
        Current two-sided residual state.
    side
        ``1`` for graph 1 or ``2`` for graph 2.
    M1_dir, M2_dir, lk_mat
        Original system matrices.
    edge_set
        Optional labels for the selected graph.
    rank_tol
        Numerical tolerance.
    rank_measure
        Residual rank definition.

    Returns
    -------
    candidates, candidate_log
        Successful candidates and all tested candidates.
    """
    if side not in (1, 2):
        raise ValueError("side must be either 1 or 2.")

    if side == 1:
        number_of_edges = M1_dir.shape[0]
        current_cuts = set(
            current_state["cut_edges1"]
        )
    else:
        number_of_edges = M2_dir.shape[0]
        current_cuts = set(
            current_state["cut_edges2"]
        )

    if edge_set is not None and len(edge_set) != number_of_edges:
        raise ValueError(
            "edge_set must contain one label per edge on the selected side."
        )

    current_rank = _state_rank(
        current_state,
        rank_measure,
    )

    scores = principal_edge_scores(
        Lambda_residual=current_state["Lambda_residual"],
        side=side,
        singular_tol=rank_tol,
    )

    remaining_edges = [
        edge_index
        for edge_index in range(number_of_edges)
        if edge_index not in current_cuts
    ]

    remaining_edges.sort(
        key=lambda edge_index: (
            -scores[edge_index],
            edge_index,
        )
    )

    candidates: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []

    for score_rank, edge_index in enumerate(
        remaining_edges,
        start=1,
    ):
        if side == 1:
            trial_cuts1 = (
                set(current_state["cut_edges1"])
                | {edge_index}
            )
            trial_cuts2 = current_state["cut_edges2"]
        else:
            trial_cuts1 = current_state["cut_edges1"]
            trial_cuts2 = (
                set(current_state["cut_edges2"])
                | {edge_index}
            )

        trial_state = compute_residual_state(
            cut_edges1=trial_cuts1,
            cut_edges2=trial_cuts2,
            M1_dir=M1_dir,
            M2_dir=M2_dir,
            lk_mat=lk_mat,
            rank_tol=rank_tol,
        )

        rank_after = _state_rank(
            trial_state,
            rank_measure,
        )

        rank_drop = current_rank - rank_after

        record = {
            "side": side,
            "graph": f"G{side}",
            "edge_idx": int(edge_index),
            "edge": _edge_label(
                edge_index,
                edge_set,
            ),
            "score": float(scores[edge_index]),
            "score_rank": int(score_rank),
            "rank_before": int(current_rank),
            "rank_after": int(rank_after),
            "rank_drop": int(rank_drop),
            "rank_reducing": bool(rank_drop > 0),
            "residual_norm": _state_norm(
                trial_state,
                rank_measure,
            ),
        }

        candidate_log.append(record)

        if rank_drop > 0:
            candidates.append(
                {
                    **record,
                    "state": trial_state,
                }
            )

    candidates.sort(
        key=lambda candidate: (
            -candidate["rank_drop"],
            -candidate["score"],
            candidate["residual_norm"],
            candidate["edge_idx"],
        )
    )

    return candidates, candidate_log


def cut_both_graphs_with_recomputed_svd(
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    lk_mat: np.ndarray,
    edge_set1: Sequence[Any] | None = None,
    edge_set2: Sequence[Any] | None = None,
    initial_cut_edges1: Iterable[int] | None = None,
    initial_cut_edges2: Iterable[int] | None = None,
    side_order: Sequence[Literal[1, 2]] = (1, 2),
    rank_tol: float = 1e-10,
    rank_measure: RankMeasure = "linkage",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Alternate recomputed-SVD single-edge cuts between both graphs.

    During one alternation cycle, each side in ``side_order`` is processed
    until it is single-edge irreducible. The routine then starts another full
    alternation cycle because cuts on one graph may make new cuts possible on
    the other graph.

    Parameters
    ----------
    M1_dir, M2_dir, lk_mat
        Original system matrices.
    edge_set1, edge_set2
        Optional graph-edge labels.
    initial_cut_edges1, initial_cut_edges2
        Optional initial cuts.
    side_order
        Order in which graph sides are processed. The default is graph 1
        followed by graph 2. ``(2, 1)`` reverses the priority.
    rank_tol
        Numerical tolerance.
    rank_measure
        ``"linkage"`` or ``"operator"``.
    verbose
        Print progress.

    Returns
    -------
    dict
        Final state, graph-specific cut sets, history, candidate log, and
        success flag.
    """
    M1_dir, M2_dir, lk_mat = _validate_cutting_inputs(
        M1_dir,
        M2_dir,
        lk_mat,
    )

    side_order = tuple(side_order)

    if sorted(side_order) != [1, 2]:
        raise ValueError(
            "side_order must contain graph sides 1 and 2 exactly once."
        )

    if edge_set1 is not None and len(edge_set1) != M1_dir.shape[0]:
        raise ValueError(
            "edge_set1 must have one entry per graph-1 edge."
        )

    if edge_set2 is not None and len(edge_set2) != M2_dir.shape[0]:
        raise ValueError(
            "edge_set2 must have one entry per graph-2 edge."
        )

    initial_state = compute_residual_state(
        cut_edges1=initial_cut_edges1,
        cut_edges2=initial_cut_edges2,
        M1_dir=M1_dir,
        M2_dir=M2_dir,
        lk_mat=lk_mat,
        rank_tol=rank_tol,
    )

    current_state = initial_state

    history: list[dict[str, Any]] = []
    candidate_log: list[dict[str, Any]] = []

    iteration = 0
    alternation_cycle = 0

    if verbose:
        print("Initial state")
        print(f"  rank measure: {rank_measure}")
        print(
            "  residual rank: "
            f"{_state_rank(current_state, rank_measure)}"
        )
        print(
            "  residual norm: "
            f"{_state_norm(current_state, rank_measure):.8e}"
        )

    while _state_rank(current_state, rank_measure) > 0:
        alternation_cycle += 1
        progress_this_cycle = False

        for side in side_order:
            edge_set = (
                edge_set1
                if side == 1
                else edge_set2
            )

            if verbose:
                print(
                    f"\nAlternation cycle {alternation_cycle}: "
                    f"testing graph {side}"
                )

            while _state_rank(
                current_state,
                rank_measure,
            ) > 0:
                candidates, tests = (
                    find_two_sided_single_candidates(
                        current_state=current_state,
                        side=side,
                        M1_dir=M1_dir,
                        M2_dir=M2_dir,
                        lk_mat=lk_mat,
                        edge_set=edge_set,
                        rank_tol=rank_tol,
                        rank_measure=rank_measure,
                    )
                )

                for row in tests:
                    row["alternation_cycle"] = alternation_cycle
                    row["prospective_iteration"] = iteration + 1

                candidate_log.extend(tests)

                if not candidates:
                    if verbose:
                        print(
                            f"  graph {side} is "
                            "single-edge irreducible"
                        )
                    break

                best = candidates[0]
                rank_before = _state_rank(
                    current_state,
                    rank_measure,
                )

                current_state = best["state"]
                iteration += 1
                progress_this_cycle = True

                history.append(
                    {
                        "iteration": iteration,
                        "alternation_cycle": alternation_cycle,
                        "side": side,
                        "graph": f"G{side}",
                        "edge_idx": best["edge_idx"],
                        "edge": best["edge"],
                        "score": best["score"],
                        "score_rank": best["score_rank"],
                        "rank_before": rank_before,
                        "rank_after": best["rank_after"],
                        "rank_drop": best["rank_drop"],
                        "residual_norm": best["residual_norm"],
                        "number_of_candidates": len(candidates),
                    }
                )

                if verbose:
                    print(
                        f"  cut G{side} edge "
                        f"{best['edge_idx']}: {best['edge']}"
                    )
                    print(
                        f"    score rank: "
                        f"{best['score_rank']} of {len(tests)}"
                    )
                    print(
                        f"    rank: {rank_before} "
                        f"-> {best['rank_after']}"
                    )

            if _state_rank(
                current_state,
                rank_measure,
            ) == 0:
                break

        if not progress_this_cycle:
            break

    final_rank = _state_rank(
        current_state,
        rank_measure,
    )

    success = final_rank == 0

    if verbose:
        if success:
            print(
                "\nSuccess: residual rank reduced to zero "
                "using two-sided single-edge cuts."
            )
        else:
            print(
                "\nStopped: neither graph has a "
                "rank-reducing single edge."
            )
            print(
                f"Remaining residual rank: {final_rank}"
            )

    return {
        "method": "two_sided_recomputed_svd",
        "rank_measure": rank_measure,
        "side_order": side_order,
        "initial_state": initial_state,
        "final_state": current_state,
        "initial_rank": _state_rank(
            initial_state,
            rank_measure,
        ),
        "final_rank": final_rank,
        "cut_edges1": list(current_state["cut_edges1"]),
        "cut_edges2": list(current_state["cut_edges2"]),
        "number_of_cuts1": len(
            current_state["cut_edges1"]
        ),
        "number_of_cuts2": len(
            current_state["cut_edges2"]
        ),
        "total_number_of_cuts": (
            len(current_state["cut_edges1"])
            + len(current_state["cut_edges2"])
        ),
        "history": history,
        "candidate_log": candidate_log,
        "success": success,
    }


# ============================================================================ #
# Result-printing helpers
# ============================================================================ #


def print_cut_result(
    result: Mapping[str, Any],
    edge_set1: Sequence[Any] | None = None,
) -> None:
    """
    Print a standard summary for a graph-1 cutting result.

    Works with results from ``cut_with_recomputed_svd`` and
    ``cut_with_original_svd``.
    """
    print("\n" + "=" * 68)
    print("EDGE-CUTTING RESULT")
    print("=" * 68)

    print("Method:", result.get("method"))
    print("Rank measure:", result.get("rank_measure"))
    print("Complete unlink success:", result["success"])
    print("Initial rank:", result["initial_rank"])
    print("Final residual rank:", result["final_rank"])

    cut_edges1 = list(
        result.get("cut_edges1", [])
    )

    print("Number of graph-1 cuts:", len(cut_edges1))
    print(
        "Cut number equals initial rank:",
        len(cut_edges1) == result["initial_rank"],
    )

    print("\nGraph-1 cut indices:")
    print(cut_edges1)

    print("\nGraph-1 cut edges:")

    for edge_index in cut_edges1:
        print(
            f"  {edge_index}: "
            f"{_edge_label(edge_index, edge_set1)}"
        )

    print("\nCut sequence:")

    for step in result.get("history", []):
        edges = tuple(step.get("edges", ()))
        labels = tuple(
            _edge_label(edge_index, edge_set1)
            for edge_index in edges
        )

        subspace_text = ""

        if "subspace_number" in step:
            subspace_text = (
                f" | original subspace="
                f"{step['subspace_number']}"
            )

        print(
            f"  {step['iteration']:2d}. "
            f"{step['cut_type']:6s} | "
            f"indices={edges} | "
            f"edges={labels} | "
            f"rank {step['rank_before']} "
            f"-> {step['rank_after']}"
            f"{subspace_text}"
        )


def print_two_sided_cut_result(
    result: Mapping[str, Any],
) -> None:
    """Print a standard summary for a two-sided cutting result."""
    print("\n" + "=" * 68)
    print("TWO-GRAPH SINGLE-EDGE RESULT")
    print("=" * 68)

    print("Complete unlink success:", result["success"])
    print("Rank measure:", result["rank_measure"])
    print("Initial rank:", result["initial_rank"])
    print("Final residual rank:", result["final_rank"])
    print("Graph-1 cuts:", result["number_of_cuts1"])
    print("Graph-2 cuts:", result["number_of_cuts2"])
    print("Total cuts:", result["total_number_of_cuts"])
    print(
        "Cut number equals initial rank:",
        (
            result["total_number_of_cuts"]
            == result["initial_rank"]
        ),
    )

    print("\nGraph-1 cut indices:")
    print(result["cut_edges1"])

    print("\nGraph-2 cut indices:")
    print(result["cut_edges2"])

    print("\nCut sequence:")

    for row in result["history"]:
        print(
            f"  {row['iteration']:2d}. "
            f"{row['graph']} | "
            f"index={row['edge_idx']:3d} | "
            f"edge={row['edge']} | "
            f"score rank={row['score_rank']:3d} | "
            f"rank {row['rank_before']} "
            f"-> {row['rank_after']}"
        )


# ============================================================================ #
# Cut-set visualization
# ============================================================================ #

def plot_cut_edges(
    D,
    cut_edges1,
    cut_edges2=None,
    *,
    color_nodes=("#030512", "#030512"),
    colormap=("RdPu", "BuGn"),
    uncut_value=0.25,
    axis=True,
    show=True,
):
    """
    Visualize cut edges on a two-layer Kirchhoff circuit.

    Cut edges are highlighted using the upper end of each layer's colormap,
    while uncut edges remain visible in a lighter version of the same color.

    The ``priority`` edge attribute remains binary:
    1.0 identifies a cut edge and 0.0 identifies an uncut edge.

    Parameters
    ----------
    D
        Two-layer Kirchhoff circuit object. The function expects
        ``D.layer[0]`` and ``D.layer[1]`` to exist.
    cut_edges1 : iterable of int
        Indices of cut edges in graph 1.
    cut_edges2 : iterable of int or None, optional
        Indices of cut edges in graph 2. If omitted, graph 2 is displayed
        entirely in its lighter baseline color.
    color_nodes : sequence of str, optional
        Node colors for graph 1 and graph 2.
    colormap : sequence of str, optional
        Edge colormaps for graph 1 and graph 2.
    uncut_value : float, optional
        Position on each colormap used for uncut edges. Must lie in
        ``[0, 1)``. Larger values give darker uncut edges.
    axis : bool, optional
        Whether the circuit plot displays its axes.
    show : bool, optional
        If True, immediately display the figure.

    Returns
    -------
    fig
        Plotly figure returned by ``D.plot_circuit``.

    Notes
    -----
    Edge indices must use the same ordering as the rows of ``M1_dir`` and
    ``M2_dir`` and as the edge tables stored in ``D.layer``.

    Kirchhoff rescales numerical edge-color arrays by their maximum value.
    Therefore, when a layer contains no cuts, a constant numerical array
    would be mapped to the darkest end of its colormap. In that case, the
    desired baseline color is sampled explicitly from the colormap and
    passed directly as a fixed color.
    """
    if not hasattr(D, "layer") or len(D.layer) < 2:
        raise TypeError(
            "D must contain at least two layers in D.layer."
        )

    if not 0.0 <= uncut_value < 1.0:
        raise ValueError(
            "uncut_value must satisfy 0 <= uncut_value < 1."
        )

    number_of_edges1 = len(D.layer[0].edges)
    number_of_edges2 = len(D.layer[1].edges)

    cut_edges1 = normalize_cut_indices(
        cut_indices=cut_edges1,
        number_of_edges=number_of_edges1,
        name="cut_edges1",
    )

    cut_edges2 = normalize_cut_indices(
        cut_indices=cut_edges2,
        number_of_edges=number_of_edges2,
        name="cut_edges2",
    )

    # ------------------------------------------------------------------ #
    # Binary priority attributes
    # ------------------------------------------------------------------ #

    cut_scores1 = np.zeros(
        number_of_edges1,
        dtype=float,
    )

    cut_scores2 = np.zeros(
        number_of_edges2,
        dtype=float,
    )

    if cut_edges1:
        cut_scores1[
            np.asarray(cut_edges1, dtype=int)
        ] = 1.0

    if cut_edges2:
        cut_scores2[
            np.asarray(cut_edges2, dtype=int)
        ] = 1.0

    D.layer[0].edges["priority"] = cut_scores1
    D.layer[1].edges["priority"] = cut_scores2

    # ------------------------------------------------------------------ #
    # Plotting colors
    # ------------------------------------------------------------------ #

    def make_edge_colors(
        cut_edges,
        number_of_edges,
        cmap,
    ):
        """
        Construct edge colors for one layer.

        If the layer contains cuts, numerical values are used:
            uncut -> uncut_value
            cut   -> 1.0

        If there are no cuts, the baseline color is sampled explicitly
        to prevent Kirchhoff from rescaling a constant array to 1.0.
        """
        if not cut_edges:
            return sample_colorscale(
                cmap,
                [uncut_value],
            )[0]

        plot_scores = np.full(
            number_of_edges,
            uncut_value,
            dtype=float,
        )

        plot_scores[
            np.asarray(cut_edges, dtype=int)
        ] = 1.0

        return plot_scores

    edge_colors1 = make_edge_colors(
        cut_edges=cut_edges1,
        number_of_edges=number_of_edges1,
        cmap=colormap[0],
    )

    edge_colors2 = make_edge_colors(
        cut_edges=cut_edges2,
        number_of_edges=number_of_edges2,
        cmap=colormap[1],
    )

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #

    fig = D.plot_circuit(
        "priority",
        color_nodes=list(color_nodes),
        color_edges=[
            edge_colors1,
            edge_colors2,
        ],
        colormap=list(colormap),
        axis=axis,
    )

    if show:
        fig.show()

    return fig


# ============================================================================ #
# SnarlPy legacy wrapper
# ============================================================================ #


def run_snarlpy_legacy_cutting(
    D: Any | None = None,
    G1: nx.Graph | None = None,
    G2: nx.Graph | None = None,
    *,
    snarlpy_module: Any,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run SnarlPy's legacy ``cuttingEdgeAlgorithm``.

    Parameters
    ----------
    D
        Optional two-layer object. If supplied, graph 1 and graph 2 are
        extracted from ``D.layer[0]`` and ``D.layer[1]``.
    G1, G2
        Explicit NetworkX graphs. These are used when ``D`` is not supplied.
    snarlpy_module
        Module or object exposing ``cuttingEdgeAlgorithm``.
        In the existing notebooks this is normally ``spe``.
    verbose
        Print the returned edge set.

    Returns
    -------
    dict
        Legacy cut edges and cut-set size.
    """
    if D is not None:
        if not hasattr(D, "layer") or len(D.layer) < 2:
            raise TypeError(
                "D must contain at least two entries in D.layer."
            )

        G1 = extract_networkx_graph(D.layer[0])
        G2 = extract_networkx_graph(D.layer[1])

    if G1 is None or G2 is None:
        raise ValueError(
            "Supply either D or both G1 and G2."
        )

    if not hasattr(
        snarlpy_module,
        "cuttingEdgeAlgorithm",
    ):
        raise AttributeError(
            "snarlpy_module has no cuttingEdgeAlgorithm attribute."
        )

    cut_edges = list(
        snarlpy_module.cuttingEdgeAlgorithm(
            cut_graph=G1,
            ref_graph=G2,
        )
    )

    result = {
        "method": "snarlpy_legacy",
        "cut_edges": cut_edges,
        "cut_set_size": len(cut_edges),
    }

    if verbose:
        print(
            "SnarlPy cut-set size:",
            result["cut_set_size"],
        )
        print("SnarlPy cut edges:")

        for number, edge in enumerate(
            cut_edges,
            start=1,
        ):
            print(f"  {number:2d}. {edge}")

    return result


# ============================================================================ #
# Graph extraction, physical edge removal, and fresh validation
# ============================================================================ #


def extract_networkx_graph(
    layer: Any,
) -> nx.Graph:
    """
    Extract a NetworkX graph from a graph, Kirchhoff layer, or circuit object.

    The attributes ``G``, ``network``, and ``graph`` are checked in that order.
    """
    if isinstance(layer, nx.Graph):
        return layer

    for attribute_name in (
        "G",
        "network",
        "graph",
    ):
        if hasattr(layer, attribute_name):
            candidate = getattr(
                layer,
                attribute_name,
            )

            if isinstance(candidate, nx.Graph):
                return candidate

    raise TypeError(
        "Could not locate a NetworkX graph in the supplied object. "
        "Checked the object itself and attributes 'G', 'network', and "
        f"'graph'. Object type: {type(layer).__name__}."
    )


def edge_indices_to_labels(
    cut_indices: Iterable[int],
    edge_set: Sequence[Any],
) -> list[Any]:
    """
    Convert matrix-row edge indices into graph-edge labels.
    """
    normalized = normalize_cut_indices(
        cut_indices,
        len(edge_set),
        name="cut_indices",
    )

    return [
        edge_set[edge_index]
        for edge_index in normalized
    ]


def apply_edge_cuts(
    graph: nx.Graph,
    cut_indices: Iterable[int],
    edge_set: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """
    Remove selected edges from a copy of a NetworkX graph.

    Parameters
    ----------
    graph
        Original graph. It is not modified.
    cut_indices
        Indices into ``edge_set``.
    edge_set
        Edge ordering associated with the rows of the relevant directed mesh
        matrix. If omitted, ``list(graph.edges())`` is used.

    Returns
    -------
    dict
        Original graph reference, cut graph copy, edge ordering, normalized
        cut indices, edge labels, and edge counts.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")

    if edge_set is None:
        edge_set = list(graph.edges())
    else:
        edge_set = list(edge_set)

    normalized_indices = normalize_cut_indices(
        cut_indices,
        len(edge_set),
        name="cut_indices",
    )

    cut_edge_labels = [
        edge_set[edge_index]
        for edge_index in normalized_indices
    ]

    missing_edges = [
        edge
        for edge in cut_edge_labels
        if not graph.has_edge(*edge)
    ]

    if missing_edges:
        raise RuntimeError(
            "Some selected edge labels are absent from the graph: "
            f"{missing_edges}"
        )

    cut_graph = graph.copy()
    cut_graph.remove_edges_from(
        cut_edge_labels
    )

    still_present = [
        edge
        for edge in cut_edge_labels
        if cut_graph.has_edge(*edge)
    ]

    if still_present:
        raise RuntimeError(
            "Some selected edges remain after removal: "
            f"{still_present}"
        )

    return {
        "original_graph": graph,
        "cut_graph": cut_graph,
        "edge_set": edge_set,
        "cut_indices": list(normalized_indices),
        "cut_edge_labels": cut_edge_labels,
        "original_edge_count": graph.number_of_edges(),
        "cut_edge_count": cut_graph.number_of_edges(),
        "number_removed": len(cut_edge_labels),
    }


def compute_fresh_post_cut_linkage(
    G1: nx.Graph,
    G2: nx.Graph,
    cycle_basis_type: Literal[
        "minimum",
        "fundamental",
    ] = "minimum",
    *,
    sampling_module: Any,
    edge_algebra_module: Any,
    signature_module: Any,
    cycle_rank_function: Callable[[nx.Graph], int],
    linkage_dict_to_matrix_function: Callable[..., np.ndarray],
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Independently recompute cycle bases, mesh matrices, and Gauss linkage.

    This validation does not assume that the two graphs are isomorphic, have
    equal edge counts, use corresponding cycle bases, or have equal mesh
    matrices.

    Parameters
    ----------
    G1, G2
        Embedded NetworkX graphs after physical edge removal.
    cycle_basis_type
        ``"minimum"`` or ``"fundamental"``.
    sampling_module
        Project module exposing:

        - ``calc_cycle_minimum_basis``
        - ``calc_cycle_basis``
        - ``calc_basis_linkage``

    edge_algebra_module
        Project module exposing:

        - ``generate_edge_matrix_binary``
        - ``generate_edge_matrix``

    signature_module
        Project module exposing ``get_edge_direction``.
    cycle_rank_function
        Function computing graph cycle rank.
    linkage_dict_to_matrix_function
        Function converting linkage results to a matrix. It must support
        ``shape=...`` and ``rounded=...`` keyword arguments.
    verbose
        Print stage timing and dimensions.

    Returns
    -------
    dict
        Fresh bases, edge sets, mesh matrices, linkage matrices, ranks, and
        timing information.
    """
    if cycle_basis_type not in (
        "minimum",
        "fundamental",
    ):
        raise ValueError(
            "cycle_basis_type must be 'minimum' or 'fundamental'."
        )

    total_start = time.perf_counter()

    def compute_basis(
        graph: nx.Graph,
        graph_name: str,
    ) -> tuple[list[Any], int, float]:
        stage_start = time.perf_counter()

        if cycle_basis_type == "minimum":
            basis = (
                sampling_module
                .calc_cycle_minimum_basis(graph)
            )
        else:
            basis = sampling_module.calc_cycle_basis(
                graph
            )

        elapsed = (
            time.perf_counter()
            - stage_start
        )

        expected_rank = int(
            cycle_rank_function(graph)
        )

        if len(basis) != expected_rank:
            raise RuntimeError(
                f"Fresh {graph_name} basis has the wrong size: "
                f"expected {expected_rank}, obtained {len(basis)}."
            )

        if verbose:
            print(
                f"{graph_name} {cycle_basis_type} basis: "
                f"{len(basis)} cycles in {elapsed:.2f} s"
            )

        return basis, expected_rank, elapsed

    basis1, cycle_rank1, basis1_time = compute_basis(
        G1,
        "G1",
    )

    basis2, cycle_rank2, basis2_time = compute_basis(
        G2,
        "G2",
    )

    mesh_start = time.perf_counter()

    edge_set1 = list(G1.edges())
    edge_set2 = list(G2.edges())

    M1_undir = (
        edge_algebra_module
        .generate_edge_matrix_binary(
            basis1,
            edge_set1,
        )
    )

    M2_undir = (
        edge_algebra_module
        .generate_edge_matrix_binary(
            basis2,
            edge_set2,
        )
    )

    signature1 = (
        signature_module
        .get_edge_direction(edge_set1)
    )

    signature2 = (
        signature_module
        .get_edge_direction(edge_set2)
    )

    M1_dir = (
        edge_algebra_module
        .generate_edge_matrix(
            basis1,
            edge_set1,
            signature1,
        )
    )

    M2_dir = (
        edge_algebra_module
        .generate_edge_matrix(
            basis2,
            edge_set2,
            signature2,
        )
    )

    mesh_time = (
        time.perf_counter()
        - mesh_start
    )

    if verbose:
        print(
            f"Fresh mesh matrices computed in {mesh_time:.2f} s"
        )
        print("Fresh M1_dir shape:", M1_dir.shape)
        print("Fresh M2_dir shape:", M2_dir.shape)

    linkage_start = time.perf_counter()

    linkage_shape = (
        len(basis1),
        len(basis2),
    )

    if len(basis1) == 0 or len(basis2) == 0:
        numeric_result: Mapping[Any, Any] = {}

        lk_mat_raw = np.zeros(
            linkage_shape,
            dtype=float,
        )

        lk_mat_rounded = np.zeros(
            linkage_shape,
            dtype=float,
        )

        if verbose:
            print(
                "At least one graph is acyclic; the fresh linkage "
                "matrix has an empty cycle dimension."
            )
    else:
        numeric_result = (
            sampling_module
            .calc_basis_linkage(
                basis1,
                basis2,
            )
        )

        lk_mat_raw = (
            linkage_dict_to_matrix_function(
                numeric_result,
                shape=linkage_shape,
                rounded=False,
            )
        )

        lk_mat_rounded = (
            linkage_dict_to_matrix_function(
                numeric_result,
                shape=linkage_shape,
                rounded=True,
            )
        )

    linkage_time = (
        time.perf_counter()
        - linkage_start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    if verbose:
        print(
            "Fresh Gauss linkage computed in "
            f"{linkage_time:.2f} s"
        )
        print(
            "Total fresh validation time: "
            f"{total_time:.2f} s"
        )

    return {
        "G1": G1,
        "G2": G2,
        "basis1": basis1,
        "basis2": basis2,
        "edge_set1": edge_set1,
        "edge_set2": edge_set2,
        "M1_undir": M1_undir,
        "M2_undir": M2_undir,
        "M1_dir": M1_dir,
        "M2_dir": M2_dir,
        "lk_mat": np.asarray(
            lk_mat_rounded,
            dtype=float,
        ),
        "lk_mat_raw": np.asarray(
            lk_mat_raw,
            dtype=float,
        ),
        "numeric_result": numeric_result,
        "cycle_rank1": cycle_rank1,
        "cycle_rank2": cycle_rank2,
        "cycle_basis_type": cycle_basis_type,
        "basis1_time_s": basis1_time,
        "basis2_time_s": basis2_time,
        "mesh_time_s": mesh_time,
        "linkage_time_s": linkage_time,
        "total_time_s": total_time,
    }


def validate_edge_cut_set(
    G1: nx.Graph,
    G2: nx.Graph,
    cut_edges1: Iterable[int] | None = None,
    cut_edges2: Iterable[int] | None = None,
    edge_set1: Sequence[Any] | None = None,
    edge_set2: Sequence[Any] | None = None,
    cycle_basis_type: Literal[
        "minimum",
        "fundamental",
    ] = "minimum",
    raw_linkage_atol: float = 1e-7,
    rank_tol: float = 1e-10,
    *,
    sampling_module: Any,
    edge_algebra_module: Any,
    signature_module: Any,
    cycle_rank_function: Callable[[nx.Graph], int],
    linkage_dict_to_matrix_function: Callable[..., np.ndarray],
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Physically apply a cut set and independently test whether linkage vanishes.

    Parameters
    ----------
    G1, G2
        Original embedded graphs.
    cut_edges1, cut_edges2
        Edge indices removed from graph 1 and graph 2.
    edge_set1, edge_set2
        Edge orderings corresponding to the original directed mesh matrices.
        If omitted, current NetworkX edge orderings are used.
    cycle_basis_type
        Fresh basis type.
    raw_linkage_atol
        Absolute tolerance used to test raw Gauss-linking values against zero.
    rank_tol
        Numerical rank tolerance.
    sampling_module, edge_algebra_module, signature_module
        Project-specific modules.
    cycle_rank_function
        Project cycle-rank function.
    linkage_dict_to_matrix_function
        Project linkage-conversion function.
    verbose
        Print validation information.

    Returns
    -------
    dict
        Physical graph-cut information, fresh linkage information, and Boolean
        validation tests.
    """
    graph1_cut_info = apply_edge_cuts(
        graph=G1,
        cut_indices=(
            ()
            if cut_edges1 is None
            else cut_edges1
        ),
        edge_set=edge_set1,
    )

    graph2_cut_info = apply_edge_cuts(
        graph=G2,
        cut_indices=(
            ()
            if cut_edges2 is None
            else cut_edges2
        ),
        edge_set=edge_set2,
    )

    G1_cut = graph1_cut_info["cut_graph"]
    G2_cut = graph2_cut_info["cut_graph"]

    if verbose:
        print("Original G1 edge count:", G1.number_of_edges())
        print("Post-cut G1 edge count:", G1_cut.number_of_edges())
        print(
            "Number of G1 edges removed:",
            graph1_cut_info["number_removed"],
        )

        print("\nOriginal G2 edge count:", G2.number_of_edges())
        print("Post-cut G2 edge count:", G2_cut.number_of_edges())
        print(
            "Number of G2 edges removed:",
            graph2_cut_info["number_removed"],
        )

    fresh = compute_fresh_post_cut_linkage(
        G1=G1_cut,
        G2=G2_cut,
        cycle_basis_type=cycle_basis_type,
        sampling_module=sampling_module,
        edge_algebra_module=edge_algebra_module,
        signature_module=signature_module,
        cycle_rank_function=cycle_rank_function,
        linkage_dict_to_matrix_function=(
            linkage_dict_to_matrix_function
        ),
        verbose=verbose,
    )

    fresh_lk_mat = fresh["lk_mat"]
    fresh_lk_mat_raw = fresh["lk_mat_raw"]

    rounded_zero = bool(
        np.all(fresh_lk_mat == 0)
    )

    raw_zero = bool(
        np.allclose(
            fresh_lk_mat_raw,
            0.0,
            atol=raw_linkage_atol,
            rtol=0.0,
        )
    )

    fresh_linkage_rank = numerical_rank(
        fresh_lk_mat,
        tol=rank_tol,
    )

    maximum_absolute_raw_linkage = (
        float(
            np.max(
                np.abs(fresh_lk_mat_raw)
            )
        )
        if fresh_lk_mat_raw.size
        else 0.0
    )

    result = {
        "G1_original": G1,
        "G2_original": G2,
        "G1_cut": G1_cut,
        "G2_cut": G2_cut,
        "graph1_cut_info": graph1_cut_info,
        "graph2_cut_info": graph2_cut_info,
        "fresh": fresh,
        "rounded_linkage_is_zero": rounded_zero,
        "raw_linkage_is_numerically_zero": raw_zero,
        "fresh_linkage_rank": fresh_linkage_rank,
        "maximum_absolute_raw_linkage": (
            maximum_absolute_raw_linkage
        ),
        "success": (
            rounded_zero
            and raw_zero
            and fresh_linkage_rank == 0
        ),
    }

    if verbose:
        print("\n" + "=" * 68)
        print("FRESH POST-CUT VALIDATION")
        print("=" * 68)

        print("Fresh G1 cycle rank:", fresh["cycle_rank1"])
        print("Fresh G2 cycle rank:", fresh["cycle_rank2"])
        print("Fresh G1 basis size:", len(fresh["basis1"]))
        print("Fresh G2 basis size:", len(fresh["basis2"]))
        print(
            "Rounded matrix is exactly zero:",
            rounded_zero,
        )
        print(
            "Raw matrix is numerically zero:",
            raw_zero,
        )
        print(
            "Maximum absolute raw linking value:",
            maximum_absolute_raw_linkage,
        )
        print("Fresh linkage rank:", fresh_linkage_rank)
        print("Validation success:", result["success"])

    return result


# ============================================================================ #
# Notebook usage examples
# ============================================================================ #

# ---------------------------------------------------------------------------
# A. Recomputed residual SVD, singles followed by pairs
# ---------------------------------------------------------------------------
#
# recomputed_result = cut_with_recomputed_svd(
#     M1_dir=M1_dir,
#     M2_dir=M2_dir,
#     lk_mat=lk_mat,
#     edge_set1=edge_set1,
#     initial_cut_edges1=[],
#     rank_tol=1e-10,
#     rank_measure="linkage",
#     allow_pair_cuts=True,
#     pair_pool_size=None,       # Exhaustive pair search
#     verbose=True,
# )
#
# print_cut_result(
#     recomputed_result,
#     edge_set1=edge_set1,
# )


# ---------------------------------------------------------------------------
# B. Recomputed residual SVD, single-edge cuts only
# ---------------------------------------------------------------------------
#
# recomputed_single_result = cut_with_recomputed_svd(
#     M1_dir=M1_dir,
#     M2_dir=M2_dir,
#     lk_mat=lk_mat,
#     edge_set1=edge_set1,
#     rank_measure="linkage",
#     allow_pair_cuts=False,
#     verbose=True,
# )


# ---------------------------------------------------------------------------
# C. Original SVD, leading and second distinct subspaces
# ---------------------------------------------------------------------------
#
# original_svd_result = cut_with_original_svd(
#     Lambda_original=Lambda_new,
#     M1_dir=M1_dir,
#     M2_dir=M2_dir,
#     lk_mat=lk_mat,
#     edge_set1=edge_set1,
#     initial_cut_edges1=[],
#     subspace_numbers=(1, 2),
#     rank_tol=1e-10,
#     score_tol=1e-12,
#     rank_measure="linkage",
#     allow_single_cuts=True,
#     allow_pair_cuts=True,
#     pair_pool_size=None,
#     verbose=True,
# )
#
# print_cut_result(
#     original_svd_result,
#     edge_set1=edge_set1,
# )
#
# leading_cut_set = original_svd_result["leading_single_cuts"]
# second_cut_set = original_svd_result["second_single_cuts"]
# leading_pair_cut_set = original_svd_result["leading_pair_cuts"]
# second_pair_cut_set = original_svd_result["second_pair_cuts"]
# cut_edges1 = original_svd_result["cut_edges1"]


# ---------------------------------------------------------------------------
# D. Inspect every successful pair and its fixed-score spectral rank
# ---------------------------------------------------------------------------
#
# state_after_singles = original_svd_result["final_state"]
#
# leading_group = original_svd_result["singular_groups"][0]
#
# leading_scores = singular_subspace_edge_scores(
#     singular_vectors=original_svd_result["U_original"],
#     subspace_indices=leading_group,
# )
#
# pair_candidates = find_original_svd_pair_candidates(
#     current_state=state_after_singles,
#     scores=leading_scores,
#     M1_dir=M1_dir,
#     M2_dir=M2_dir,
#     lk_mat=lk_mat,
#     edge_set1=edge_set1,
#     rank_tol=1e-10,
#     rank_measure="linkage",
#     include_spectral_rank=True,
# )
#
# for candidate in pair_candidates:
#     print(
#         f"edges={candidate['edges']} "
#         f"{candidate['edge_labels']}, "
#         f"spectral rank={candidate['spectral_rank']} "
#         f"of {candidate['number_of_ranked_pairs']}, "
#         f"score={candidate['score']:.8f}, "
#         f"rank {candidate['rank_before']} "
#         f"-> {candidate['rank_after']}"
#     )


# ---------------------------------------------------------------------------
# E. Alternate single-edge cuts between graph 1 and graph 2
# ---------------------------------------------------------------------------
#
# two_sided_result = cut_both_graphs_with_recomputed_svd(
#     M1_dir=M1_dir,
#     M2_dir=M2_dir,
#     lk_mat=lk_mat,
#     edge_set1=edge_set1,
#     edge_set2=edge_set2,
#     initial_cut_edges1=[],
#     initial_cut_edges2=[],
#     side_order=(1, 2),
#     rank_tol=1e-10,
#     rank_measure="linkage",
#     verbose=True,
# )
#
# print_two_sided_cut_result(two_sided_result)


# ---------------------------------------------------------------------------
# F. SnarlPy legacy baseline
# ---------------------------------------------------------------------------
#
# snarlpy_result = run_snarlpy_legacy_cutting(
#     D=D,
#     snarlpy_module=spe,
#     verbose=True,
# )


# ---------------------------------------------------------------------------
# G. Fresh physical validation of a graph-1 cut set
# ---------------------------------------------------------------------------
#
# G1_original = extract_networkx_graph(D.layer[0])
# G2_original = extract_networkx_graph(D.layer[1])
#
# # IMPORTANT:
# # Use the exact edge ordering that was used to build M1_dir and M2_dir.
# #
# # edge_set1 = ...
# # edge_set2 = ...
#
# validation = validate_edge_cut_set(
#     G1=G1_original,
#     G2=G2_original,
#     cut_edges1=recomputed_result["cut_edges1"],
#     cut_edges2=[],
#     edge_set1=edge_set1,
#     edge_set2=edge_set2,
#     cycle_basis_type=cycle_basis_type,
#     raw_linkage_atol=1e-7,
#     rank_tol=1e-10,
#     sampling_module=sampling,
#     edge_algebra_module=edge_algebra,
#     signature_module=signature,
#     cycle_rank_function=cycle_rank,
#     linkage_dict_to_matrix_function=linkage_dict_to_matrix,
#     verbose=True,
# )