"""
compute_ensnarlment.py
======================

Construct the ensnarlment operator for a pair of embedded spatial networks.

The public :func:`compute_ensnarlment` pipeline performs only the operations
needed to construct the operator used in the paper:

1. Extract the two embedded NetworkX graphs.
2. Compute cycle bases independently, or compute a basis on graph 1 and map it
   to graph 2 through a graph isomorphism.
3. Construct the directed edge-cycle matrices ``M1_dir`` and ``M2_dir``.
4. Compute and immediately round the cycle-cycle Gauss-linking matrix ``L``.
5. Construct the ensnarlment operator

       Lambda = pinv(M1_dir).T @ L @ pinv(M2_dir).

Objects intentionally not constructed or retained
-----------------------------------------------
- unsigned mesh matrices;
- raw, unrounded linkage matrices;
- Gram matrices or inverse Gram matrices;
- pseudoinverses after Lambda has been assembled;
- Lambda @ Lambda.T or Lambda.T @ Lambda;
- singular values, singular vectors, or other spectral results.

Spectral analysis should be performed in a separate module directly from the
SVD of ``ensnarlment_operator``.

External project dependencies
-----------------------------
The cycle, mesh, and linking operations use the existing project modules named
``sampling``, ``edge_algebra``, and ``signature``. The module attempts to find
these dependencies automatically. They may also be supplied explicitly to
:func:`compute_ensnarlment`, which avoids assumptions about their import path.
"""

from __future__ import annotations

__version__ = "0.1.0"

import importlib
import time
import warnings
from typing import Any, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
import scipy.linalg as spla
from networkx.algorithms import isomorphism as iso


CycleBasisType = Literal["minimum", "fundamental"]
RoundingAction = Literal["raise", "warn", "ignore"]


__all__ = [
    "cycle_rank",
    "extract_graph_pair",
    "compute_cycle_basis",
    "find_graph_isomorphism",
    "map_cycle_to_graph",
    "map_edge_to_graph",
    "build_directed_mesh_matrix",
    "compute_linkage_matrix",
    "compute_ensnarlment_operator",
    "compute_ensnarlment",
    "print_ensnarlment_summary",
]


# ============================================================================
# Dependency resolution
# ============================================================================


def _import_first_available(module_names: Sequence[str]) -> Any | None:
    """Import and return the first available module from ``module_names``."""
    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue

    return None


def _resolve_project_modules(
    sampling_module: Any | None,
    edge_algebra_module: Any | None,
    signature_module: Any | None,
) -> tuple[Any, Any, Any]:
    """
    Resolve the three legacy/project-specific modules used by the pipeline.

    Explicitly supplied modules take precedence. Automatic lookup supports
    common top-level and package-qualified import layouts.
    """
    if sampling_module is None:
        sampling_module = _import_first_available(
            (
                "sampling",
                "snarlpy.sampling",
                "SnarlPy.sampling",
            )
        )

    if edge_algebra_module is None:
        edge_algebra_module = _import_first_available(
            (
                "edge_algebra",
                "snarlpy.edge_algebra",
                "SnarlPy.edge_algebra",
            )
        )

    if signature_module is None:
        signature_module = _import_first_available(
            (
                "signature",
                "snarlpy.signature",
                "SnarlPy.signature",
            )
        )

    missing = []

    if sampling_module is None:
        missing.append("sampling")

    if edge_algebra_module is None:
        missing.append("edge_algebra")

    if signature_module is None:
        missing.append("signature")

    if missing:
        raise ImportError(
            "Could not import the following project modules: "
            f"{', '.join(missing)}. Either make them importable in the "
            "current environment or pass sampling_module=..., "
            "edge_algebra_module=..., and signature_module=... explicitly."
        )

    required_attributes = {
        "sampling": (
            sampling_module,
            (
                "calc_cycle_minimum_basis",
                "calc_cycle_basis",
                "calc_basis_linkage",
            ),
        ),
        "edge_algebra": (
            edge_algebra_module,
            ("generate_edge_matrix",),
        ),
        "signature": (
            signature_module,
            ("get_edge_direction",),
        ),
    }

    for module_name, (module, attribute_names) in required_attributes.items():
        missing_attributes = [
            attribute_name
            for attribute_name in attribute_names
            if not hasattr(module, attribute_name)
        ]

        if missing_attributes:
            raise AttributeError(
                f"The supplied {module_name} module is missing required "
                f"attributes: {missing_attributes}."
            )

    return (
        sampling_module,
        edge_algebra_module,
        signature_module,
    )


# ============================================================================
# Graph and edge-ordering helpers
# ============================================================================


def _extract_networkx_graph(layer: Any) -> nx.Graph:
    """Extract a NetworkX graph from a graph or Kirchhoff layer object."""
    if isinstance(layer, nx.Graph):
        return layer

    for attribute_name in ("G", "network", "graph"):
        if hasattr(layer, attribute_name):
            candidate = getattr(layer, attribute_name)

            if isinstance(candidate, nx.Graph):
                return candidate

    raise TypeError(
        "Could not locate a NetworkX graph in the supplied layer. "
        f"Received object type {type(layer).__name__}."
    )


def extract_graph_pair(network_pair: Any) -> tuple[nx.Graph, nx.Graph]:
    """
    Extract two NetworkX graphs from a supported network-pair container.

    Supported inputs
    ----------------
    - a Kirchhoff dual circuit with ``network_pair.layer``;
    - a two-element sequence of graphs or graph-containing layer objects.
    """
    if hasattr(network_pair, "layer"):
        layers = network_pair.layer
    elif isinstance(network_pair, Sequence) and not isinstance(
        network_pair,
        (str, bytes),
    ):
        layers = network_pair
    else:
        raise TypeError(
            "network_pair must have a 'layer' attribute or be a "
            "two-element sequence."
        )

    if len(layers) != 2:
        raise ValueError(
            "The ensnarlment pipeline requires exactly two network layers; "
            f"received {len(layers)}."
        )

    return (
        _extract_networkx_graph(layers[0]),
        _extract_networkx_graph(layers[1]),
    )


def _canonical_undirected_edge(edge: Sequence[Any]) -> frozenset[Any]:
    """Return an orientation-independent representation of an edge."""
    if len(edge) != 2:
        raise ValueError(f"Expected a two-node edge, received {edge!r}.")

    return frozenset(edge)


def _validate_edge_set(
    graph: nx.Graph,
    edge_set: Sequence[Sequence[Any]],
    *,
    name: str,
) -> list[tuple[Any, Any]]:
    """Validate that an ordered edge list represents every graph edge once."""
    edge_set = [tuple(edge) for edge in edge_set]

    if len(edge_set) != graph.number_of_edges():
        raise ValueError(
            f"{name} contains {len(edge_set)} edges, but the graph has "
            f"{graph.number_of_edges()} edges."
        )

    missing = [
        edge
        for edge in edge_set
        if not graph.has_edge(*edge)
    ]

    if missing:
        raise ValueError(
            f"{name} contains edges that are absent from the graph: "
            f"{missing[:10]}"
        )

    canonical_edges = [
        _canonical_undirected_edge(edge)
        for edge in edge_set
    ]

    if len(set(canonical_edges)) != len(canonical_edges):
        raise ValueError(f"{name} contains duplicate undirected edges.")

    graph_edges = {
        _canonical_undirected_edge(edge)
        for edge in graph.edges()
    }

    if set(canonical_edges) != graph_edges:
        raise ValueError(
            f"{name} does not represent exactly the graph's edge set."
        )

    return edge_set


def _native_edge_sets(
    network_pair: Any,
    G1: nx.Graph,
    G2: nx.Graph,
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """
    Retrieve stable edge orderings from the network container when available.

    The network-generation module stores these as ``edge_set1`` and
    ``edge_set2``. If they are unavailable, native NetworkX insertion order is
    used.
    """
    edge_set1 = getattr(
        network_pair,
        "edge_set1",
        list(G1.edges()),
    )

    edge_set2 = getattr(
        network_pair,
        "edge_set2",
        list(G2.edges()),
    )

    return (
        _validate_edge_set(G1, edge_set1, name="edge_set1"),
        _validate_edge_set(G2, edge_set2, name="edge_set2"),
    )


# ============================================================================
# Cycle bases and graph correspondence
# ============================================================================


def cycle_rank(G: nx.Graph) -> int:
    """Return the dimension of the undirected cycle space of ``G``."""
    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a NetworkX graph.")

    return int(
        G.number_of_edges()
        - G.number_of_nodes()
        + nx.number_connected_components(G)
    )


def compute_cycle_basis(
    G: nx.Graph,
    cycle_basis_type: CycleBasisType = "minimum",
    *,
    sampling_module: Any,
) -> list[nx.Graph]:
    """
    Compute a minimum or fundamental cycle basis using the project backend.
    """
    if cycle_basis_type == "minimum":
        basis = sampling_module.calc_cycle_minimum_basis(G)
    elif cycle_basis_type == "fundamental":
        basis = sampling_module.calc_cycle_basis(G)
    else:
        raise ValueError(
            "cycle_basis_type must be either 'minimum' or 'fundamental'."
        )

    basis = list(basis)
    expected_cycle_rank = cycle_rank(G)

    if len(basis) != expected_cycle_rank:
        raise RuntimeError(
            "The computed cycle basis has the wrong size: expected "
            f"{expected_cycle_rank}, obtained {len(basis)}."
        )

    return basis


def find_graph_isomorphism(
    G1: nx.Graph,
    G2: nx.Graph,
) -> dict[Any, Any]:
    """
    Return one topology-only node isomorphism from ``G1`` to ``G2``.

    Spatial positions and other attributes are intentionally ignored.
    """
    matcher = iso.GraphMatcher(G1, G2)

    if not matcher.is_isomorphic():
        raise ValueError("G1 and G2 are not isomorphic.")

    return dict(next(matcher.isomorphisms_iter()))


def map_cycle_to_graph(
    cycle_G1: nx.Graph,
    G2: nx.Graph,
    node_isomorphism: Mapping[Any, Any],
) -> nx.Graph:
    """
    Map one cycle from graph 1 into graph 2 and copy graph-2 attributes.

    Copying graph-2 node attributes is essential because the linkage
    calculation must use graph 2's actual spatial coordinates.
    """
    cycle_G2 = nx.Graph()

    for node in cycle_G1.nodes():
        if node not in node_isomorphism:
            raise KeyError(
                f"Node {node!r} is absent from the node isomorphism."
            )

    for u, v in cycle_G1.edges():
        u2 = node_isomorphism[u]
        v2 = node_isomorphism[v]

        if not G2.has_edge(u2, v2):
            raise ValueError(
                f"Mapped edge {(u2, v2)} is not present in G2."
            )

        cycle_G2.add_edge(u2, v2)

    for node in cycle_G2.nodes():
        cycle_G2.nodes[node].update(G2.nodes[node])

    for u, v in cycle_G2.edges():
        cycle_G2.edges[u, v].update(G2.edges[u, v])

    return cycle_G2


def map_edge_to_graph(
    edge_G1: Sequence[Any],
    G2: nx.Graph,
    node_isomorphism: Mapping[Any, Any],
) -> tuple[Any, Any]:
    """Map one graph-1 edge into graph 2 using ``node_isomorphism``."""
    u, v = edge_G1

    try:
        u2 = node_isomorphism[u]
        v2 = node_isomorphism[v]
    except KeyError as error:
        raise KeyError(
            f"An endpoint of edge {(u, v)} is absent from the node "
            "isomorphism."
        ) from error

    if not G2.has_edge(u2, v2):
        raise ValueError(
            f"Mapped edge {(u2, v2)} is not present in G2."
        )

    return (u2, v2)


# ============================================================================
# Directed mesh matrices and linkage matrix
# ============================================================================


def build_directed_mesh_matrix(
    basis: Sequence[nx.Graph],
    edge_set: Sequence[Sequence[Any]],
    *,
    edge_algebra_module: Any,
    signature_module: Any,
) -> np.ndarray:
    """
    Construct a directed edge-cycle matrix with shape ``(E, C)``.

    No unsigned mesh matrix is constructed.
    """
    edge_set = [tuple(edge) for edge in edge_set]
    edge_signatures = signature_module.get_edge_direction(edge_set)

    directed_mesh_matrix = np.asarray(
        edge_algebra_module.generate_edge_matrix(
            basis,
            edge_set,
            edge_signatures,
        ),
        dtype=float,
    )

    expected_shape = (
        len(edge_set),
        len(basis),
    )

    if directed_mesh_matrix.shape != expected_shape:
        raise RuntimeError(
            "The directed mesh matrix has an unexpected orientation or "
            f"shape. Expected E x C = {expected_shape}, obtained "
            f"{directed_mesh_matrix.shape}."
        )

    if not np.all(np.isfinite(directed_mesh_matrix)):
        raise ValueError(
            "The directed mesh matrix contains non-finite values."
        )

    return directed_mesh_matrix


def _validate_rounding_action(
    rounding_action: RoundingAction,
) -> None:
    if rounding_action not in ("raise", "warn", "ignore"):
        raise ValueError(
            "rounding_action must be 'raise', 'warn', or 'ignore'."
        )


def compute_linkage_matrix(
    basis1: Sequence[nx.Graph],
    basis2: Sequence[nx.Graph],
    *,
    sampling_module: Any,
    rounding_tolerance: float = 1e-5,
    rounding_action: RoundingAction = "warn",
) -> tuple[np.ndarray, float]:
    """
    Compute the rounded integer cycle-cycle linkage matrix.

    The raw Gauss-linking matrix is never assembled or retained. Raw values
    are inspected one at a time only to quantify their distance from the
    nearest integer before being written into the rounded matrix.

    Returns
    -------
    linkage_matrix
        Integer array of shape ``(len(basis1), len(basis2))``.
    maximum_rounding_error
        Largest absolute difference between a raw linking value and its
        nearest integer.
    """
    if rounding_tolerance < 0:
        raise ValueError("rounding_tolerance must be nonnegative.")

    _validate_rounding_action(rounding_action)

    linkage_shape = (
        len(basis1),
        len(basis2),
    )

    linkage_matrix = np.zeros(
        linkage_shape,
        dtype=int,
    )

    if len(basis1) == 0 or len(basis2) == 0:
        return linkage_matrix, 0.0

    numeric_result = sampling_module.calc_basis_linkage(
        basis1,
        basis2,
    )

    expected_entries = {
        (i, j)
        for i in range(len(basis1))
        for j in range(len(basis2))
    }

    returned_entries = set(numeric_result.keys())
    unexpected_entries = returned_entries - expected_entries

    if unexpected_entries:
        raise IndexError(
            "The linkage backend returned indices outside the expected "
            f"matrix shape {linkage_shape}: {sorted(unexpected_entries)[:10]}"
        )

    maximum_rounding_error = 0.0

    for (i, j), raw_value in numeric_result.items():
        raw_value = float(raw_value)

        if not np.isfinite(raw_value):
            raise ValueError(
                f"Non-finite linking value returned for cycle pair {(i, j)}."
            )

        rounded_value = int(np.rint(raw_value))
        rounding_error = abs(raw_value - rounded_value)

        maximum_rounding_error = max(
            maximum_rounding_error,
            rounding_error,
        )

        linkage_matrix[i, j] = rounded_value

    if maximum_rounding_error > rounding_tolerance:
        message = (
            "At least one Gauss-linking value is not sufficiently close to "
            "an integer. Maximum rounding error: "
            f"{maximum_rounding_error:.6e}; tolerance: "
            f"{rounding_tolerance:.6e}."
        )

        if rounding_action == "raise":
            raise RuntimeError(message)

        if rounding_action == "warn":
            warnings.warn(
                message,
                RuntimeWarning,
                stacklevel=2,
            )

    return linkage_matrix, maximum_rounding_error


# ============================================================================
# Ensnarlment operator
# ============================================================================


def compute_ensnarlment_operator(
    M1_dir: np.ndarray,
    M2_dir: np.ndarray,
    linkage_matrix: np.ndarray,
    *,
    pinv_atol: float | None = None,
    pinv_rtol: float | None = None,
    check_finite: bool = True,
) -> np.ndarray:
    """
    Construct the ensnarlment operator using Moore-Penrose pseudoinverses.

    With directed edge-cycle matrices of shapes ``(E1, C1)`` and ``(E2, C2)``
    and a linkage matrix of shape ``(C1, C2)``, the operator is

        Lambda = pinv(M1_dir).T @ linkage_matrix @ pinv(M2_dir),

    and therefore has shape ``(E1, E2)``.

    The pseudoinverses are temporary and are not returned or retained.
    """
    M1_dir = np.asarray(M1_dir, dtype=float)
    M2_dir = np.asarray(M2_dir, dtype=float)
    linkage_matrix = np.asarray(linkage_matrix, dtype=float)

    if M1_dir.ndim != 2 or M2_dir.ndim != 2:
        raise ValueError("M1_dir and M2_dir must be two-dimensional.")

    expected_linkage_shape = (
        M1_dir.shape[1],
        M2_dir.shape[1],
    )

    if linkage_matrix.shape != expected_linkage_shape:
        raise ValueError(
            "linkage_matrix has shape "
            f"{linkage_matrix.shape}; expected {expected_linkage_shape}."
        )

    if pinv_atol is not None and pinv_atol < 0:
        raise ValueError("pinv_atol must be nonnegative or None.")

    if pinv_rtol is not None and pinv_rtol < 0:
        raise ValueError("pinv_rtol must be nonnegative or None.")

    pinv_kwargs: dict[str, Any] = {
        "check_finite": check_finite,
    }

    if pinv_atol is not None:
        pinv_kwargs["atol"] = pinv_atol

    if pinv_rtol is not None:
        pinv_kwargs["rtol"] = pinv_rtol

    M1_pinv = spla.pinv(
        M1_dir,
        **pinv_kwargs,
    )

    M2_pinv = spla.pinv(
        M2_dir,
        **pinv_kwargs,
    )

    ensnarlment_operator = (
        M1_pinv.T
        @ linkage_matrix
        @ M2_pinv
    )

    expected_operator_shape = (
        M1_dir.shape[0],
        M2_dir.shape[0],
    )

    if ensnarlment_operator.shape != expected_operator_shape:
        raise RuntimeError(
            "The ensnarlment operator has shape "
            f"{ensnarlment_operator.shape}; expected "
            f"{expected_operator_shape}."
        )

    if not np.all(np.isfinite(ensnarlment_operator)):
        raise ValueError(
            "The ensnarlment operator contains non-finite values."
        )

    return ensnarlment_operator


# ============================================================================
# Complete construction pipeline
# ============================================================================


def compute_ensnarlment(
    network_pair: Any,
    *,
    cycle_basis_type: CycleBasisType = "minimum",
    use_isomorphism: bool = False,
    node_isomorphism: Mapping[Any, Any] | None = None,
    edge_set1: Sequence[Sequence[Any]] | None = None,
    edge_set2: Sequence[Sequence[Any]] | None = None,
    linkage_rounding_tolerance: float = 1e-5,
    linkage_rounding_action: RoundingAction = "warn",
    pinv_atol: float | None = None,
    pinv_rtol: float | None = None,
    check_finite: bool = True,
    sampling_module: Any | None = None,
    edge_algebra_module: Any | None = None,
    signature_module: Any | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute cycle bases, directed mesh matrices, linkage, and Lambda.

    Parameters
    ----------
    network_pair
        Kirchhoff dual circuit, or a two-element sequence containing the two
        embedded NetworkX graphs.
    cycle_basis_type
        ``"minimum"`` uses ``calc_cycle_minimum_basis``.
        ``"fundamental"`` uses ``calc_cycle_basis``.
    use_isomorphism
        If ``False``, compute cycle bases independently on both graphs.

        If ``True``, compute a basis only on graph 1, map it to graph 2 through
        a topology-only node isomorphism, and map graph 1's edge ordering to
        graph 2. This gives corresponding bases and edge coordinates.
    node_isomorphism
        Optional precomputed mapping from graph-1 nodes to graph-2 nodes.
        Valid only when ``use_isomorphism=True``.
    edge_set1, edge_set2
        Optional exact edge orderings. If omitted, edge orderings stored on the
        generated network pair are used when available; otherwise NetworkX
        insertion order is used.

        When ``use_isomorphism=True``, ``edge_set2`` must be omitted because it
        is generated by mapping ``edge_set1`` through the node isomorphism.
    linkage_rounding_tolerance
        Maximum accepted distance between a raw Gauss-linking value and its
        nearest integer.
    linkage_rounding_action
        ``"raise"``, ``"warn"``, or ``"ignore"`` when the rounding tolerance
        is exceeded.
    pinv_atol, pinv_rtol
        Optional absolute and relative thresholds passed to
        ``scipy.linalg.pinv``. ``None`` uses SciPy's default for that threshold.
    check_finite
        Forwarded to ``scipy.linalg.pinv``.
    sampling_module, edge_algebra_module, signature_module
        Optional explicit project modules. Supplying them is useful when the
        package is not installed under one of the automatically searched names.
    verbose
        Print a concise construction summary and stage timings.

    Returns
    -------
    dict
        Contains only the retained construction objects and metadata:

        ``basis1``, ``basis2``
            Cycle bases.
        ``edge_set1``, ``edge_set2``
            Exact edge orderings used by the directed mesh matrices.
        ``M1_dir``, ``M2_dir``
            Directed edge-cycle matrices with shapes ``(E1, C1)`` and
            ``(E2, C2)``.
        ``linkage_matrix``
            Rounded integer cycle-cycle linkage matrix.
        ``ensnarlment_operator``
            The edge-edge ensnarlment operator Lambda.
        ``node_isomorphism``
            Mapping used for corresponding bases, or ``None``.
        construction metadata and timings.

    Notes
    -----
    This function does not compute the SVD or numerical rank of Lambda.
    """
    total_start = time.perf_counter()

    (
        sampling_module,
        edge_algebra_module,
        signature_module,
    ) = _resolve_project_modules(
        sampling_module=sampling_module,
        edge_algebra_module=edge_algebra_module,
        signature_module=signature_module,
    )

    G1, G2 = extract_graph_pair(network_pair)

    if edge_set1 is None or edge_set2 is None:
        native_edge_set1, native_edge_set2 = _native_edge_sets(
            network_pair,
            G1,
            G2,
        )
    else:
        native_edge_set1 = []
        native_edge_set2 = []

    if edge_set1 is None:
        edge_set1 = native_edge_set1
    else:
        edge_set1 = _validate_edge_set(
            G1,
            edge_set1,
            name="edge_set1",
        )

    if use_isomorphism:
        if edge_set2 is not None:
            raise ValueError(
                "Do not supply edge_set2 when use_isomorphism=True. "
                "The corresponding graph-2 ordering is generated by mapping "
                "edge_set1 through the node isomorphism."
            )
    else:
        if node_isomorphism is not None:
            raise ValueError(
                "node_isomorphism is only valid when use_isomorphism=True."
            )

        if edge_set2 is None:
            edge_set2 = native_edge_set2
        else:
            edge_set2 = _validate_edge_set(
                G2,
                edge_set2,
                name="edge_set2",
            )

    basis_start = time.perf_counter()

    if use_isomorphism:
        isomorphism_start = time.perf_counter()

        if node_isomorphism is None:
            node_isomorphism = find_graph_isomorphism(
                G1,
                G2,
            )
        else:
            node_isomorphism = dict(node_isomorphism)

            if set(node_isomorphism.keys()) != set(G1.nodes()):
                raise ValueError(
                    "node_isomorphism must map every graph-1 node exactly "
                    "once."
                )

            if set(node_isomorphism.values()) != set(G2.nodes()):
                raise ValueError(
                    "node_isomorphism values must cover every graph-2 node "
                    "exactly once."
                )

            for u, v in G1.edges():
                if not G2.has_edge(
                    node_isomorphism[u],
                    node_isomorphism[v],
                ):
                    raise ValueError(
                        "The supplied node_isomorphism does not preserve "
                        f"graph-1 edge {(u, v)}."
                    )

        isomorphism_time = (
            time.perf_counter()
            - isomorphism_start
        )

        basis1 = compute_cycle_basis(
            G1,
            cycle_basis_type=cycle_basis_type,
            sampling_module=sampling_module,
        )

        basis2 = [
            map_cycle_to_graph(
                cycle,
                G2,
                node_isomorphism,
            )
            for cycle in basis1
        ]

        edge_set2 = [
            map_edge_to_graph(
                edge,
                G2,
                node_isomorphism,
            )
            for edge in edge_set1
        ]

        edge_set2 = _validate_edge_set(
            G2,
            edge_set2,
            name="mapped edge_set2",
        )

        basis_note = (
            "Cycle basis computed on graph 1 and mapped to graph 2 through "
            "a graph isomorphism."
        )
    else:
        isomorphism_time = 0.0
        node_isomorphism = None

        basis1 = compute_cycle_basis(
            G1,
            cycle_basis_type=cycle_basis_type,
            sampling_module=sampling_module,
        )

        basis2 = compute_cycle_basis(
            G2,
            cycle_basis_type=cycle_basis_type,
            sampling_module=sampling_module,
        )

        basis_note = (
            "Cycle bases computed independently on graph 1 and graph 2."
        )

    basis_time = (
        time.perf_counter()
        - basis_start
    )

    mesh_start = time.perf_counter()

    M1_dir = build_directed_mesh_matrix(
        basis=basis1,
        edge_set=edge_set1,
        edge_algebra_module=edge_algebra_module,
        signature_module=signature_module,
    )

    M2_dir = build_directed_mesh_matrix(
        basis=basis2,
        edge_set=edge_set2,
        edge_algebra_module=edge_algebra_module,
        signature_module=signature_module,
    )

    mesh_time = (
        time.perf_counter()
        - mesh_start
    )

    linkage_start = time.perf_counter()

    linkage_matrix, maximum_rounding_error = compute_linkage_matrix(
        basis1=basis1,
        basis2=basis2,
        sampling_module=sampling_module,
        rounding_tolerance=linkage_rounding_tolerance,
        rounding_action=linkage_rounding_action,
    )

    linkage_time = (
        time.perf_counter()
        - linkage_start
    )

    operator_start = time.perf_counter()

    ensnarlment_operator = compute_ensnarlment_operator(
        M1_dir=M1_dir,
        M2_dir=M2_dir,
        linkage_matrix=linkage_matrix,
        pinv_atol=pinv_atol,
        pinv_rtol=pinv_rtol,
        check_finite=check_finite,
    )

    operator_time = (
        time.perf_counter()
        - operator_start
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    result = {
        "basis1": basis1,
        "basis2": basis2,
        "edge_set1": list(edge_set1),
        "edge_set2": list(edge_set2),
        "M1_dir": M1_dir,
        "M2_dir": M2_dir,
        "linkage_matrix": linkage_matrix,
        "ensnarlment_operator": ensnarlment_operator,
        "node_isomorphism": node_isomorphism,
        "use_isomorphism": bool(use_isomorphism),
        "isomorphic_check": bool(nx.is_isomorphic(G1, G2)),
        "cycle_basis_type": cycle_basis_type,
        "cycle_basis_is_minimal": cycle_basis_type == "minimum",
        "basis_note": basis_note,
        "cycle_rank1": cycle_rank(G1),
        "cycle_rank2": cycle_rank(G2),
        "number_of_cycles1": len(basis1),
        "number_of_cycles2": len(basis2),
        "number_of_edges1": G1.number_of_edges(),
        "number_of_edges2": G2.number_of_edges(),
        "maximum_linkage_rounding_error": maximum_rounding_error,
        "pinv_atol": pinv_atol,
        "pinv_rtol": pinv_rtol,
        "isomorphism_time_s": isomorphism_time,
        "basis_time_s": basis_time,
        "mesh_time_s": mesh_time,
        "linkage_time_s": linkage_time,
        "operator_time_s": operator_time,
        "total_time_s": total_time,
    }

    if verbose:
        print_ensnarlment_summary(result)

    return result


# ============================================================================
# Reporting
# ============================================================================


def print_ensnarlment_summary(
    result: Mapping[str, Any],
) -> None:
    """Print a concise summary of a completed ensnarlment construction."""
    print("\n" + "=" * 68)
    print("ENSNARLMENT CONSTRUCTION")
    print("=" * 68)

    print("Cycle-basis type:", result["cycle_basis_type"])
    print("Uses mapped isomorphic basis:", result["use_isomorphism"])
    print(
        "Graph cycle ranks:",
        (result["cycle_rank1"], result["cycle_rank2"]),
    )
    print(
        "Basis sizes:",
        (
            result["number_of_cycles1"],
            result["number_of_cycles2"],
        ),
    )
    print(
        "Directed mesh shapes:",
        (
            result["M1_dir"].shape,
            result["M2_dir"].shape,
        ),
    )
    print(
        "Linkage-matrix shape:",
        result["linkage_matrix"].shape,
    )
    print(
        "Ensnarlment-operator shape:",
        result["ensnarlment_operator"].shape,
    )
    print(
        "Maximum linkage rounding error:",
        f"{result['maximum_linkage_rounding_error']:.6e}",
    )

    print("\nStage timings")
    print(
        "  graph isomorphism:",
        f"{result['isomorphism_time_s']:.3f} s",
    )
    print(
        "  cycle bases and mapping:",
        f"{result['basis_time_s']:.3f} s",
    )
    print(
        "  directed mesh matrices:",
        f"{result['mesh_time_s']:.3f} s",
    )
    print(
        "  cycle linkage:",
        f"{result['linkage_time_s']:.3f} s",
    )
    print(
        "  ensnarlment operator:",
        f"{result['operator_time_s']:.3f} s",
    )
    print(
        "  total:",
        f"{result['total_time_s']:.3f} s",
    )


# ============================================================================
# Notebook usage examples
# ============================================================================

# During active development:
#
#     %load_ext autoreload
#     %autoreload 2
#
#     import ensnarlment_utils.compute_ensnarlment as ce


# ----------------------------------------------------------------------------
# Independent cycle bases
# ----------------------------------------------------------------------------
#
# Use this for non-isomorphic network pairs, or when each graph should use its
# own independently computed cycle basis and native edge ordering.
#
#     result = ce.compute_ensnarlment(
#         network_pair=D,
#         cycle_basis_type="minimum",
#         use_isomorphism=False,
#         node_isomorphism=None,
#         edge_set1=None,
#         edge_set2=None,
#         linkage_rounding_tolerance=1e-5,
#         linkage_rounding_action="warn",
#         pinv_atol=None,
#         pinv_rtol=None,
#         check_finite=True,
#         sampling_module=None,
#         edge_algebra_module=None,
#         signature_module=None,
#         verbose=True,
#     )


# ----------------------------------------------------------------------------
# Corresponding bases for isomorphic graphs
# ----------------------------------------------------------------------------
#
# A basis is computed only on graph 1 and mapped to graph 2. Graph 2's edge
# ordering is also mapped from graph 1, giving corresponding edge coordinates.
#
#     result = ce.compute_ensnarlment(
#         network_pair=D,
#         cycle_basis_type="minimum",
#         use_isomorphism=True,
#         node_isomorphism=None,
#         edge_set1=None,
#         edge_set2=None,
#         linkage_rounding_tolerance=1e-5,
#         linkage_rounding_action="warn",
#         pinv_atol=None,
#         pinv_rtol=None,
#         check_finite=True,
#         sampling_module=None,
#         edge_algebra_module=None,
#         signature_module=None,
#         verbose=True,
#     )


# ----------------------------------------------------------------------------
# Normal compact calls
# ----------------------------------------------------------------------------
#
# Independent minimum bases:
#
#     result = ce.compute_ensnarlment(
#         D,
#         cycle_basis_type="minimum",
#     )
#
# Corresponding minimum bases for an isomorphic pair:
#
#     result = ce.compute_ensnarlment(
#         D,
#         cycle_basis_type="minimum",
#         use_isomorphism=True,
#     )
#
# Faster fundamental bases:
#
#     result = ce.compute_ensnarlment(
#         D,
#         cycle_basis_type="fundamental",
#         use_isomorphism=True,
#     )


# ----------------------------------------------------------------------------
# Access retained objects
# ----------------------------------------------------------------------------
#
#     basis1 = result["basis1"]
#     basis2 = result["basis2"]
#
#     edge_set1 = result["edge_set1"]
#     edge_set2 = result["edge_set2"]
#
#     M1_dir = result["M1_dir"]
#     M2_dir = result["M2_dir"]
#
#     linkage_matrix = result["linkage_matrix"]
#     Lambda = result["ensnarlment_operator"]


# ----------------------------------------------------------------------------
# Pass the result into the edge-cutting module
# ----------------------------------------------------------------------------
#
#     cut_result = ec.cut_with_original_svd(
#         P_original=result["ensnarlment_operator"],
#         M1_dir=result["M1_dir"],
#         M2_dir=result["M2_dir"],
#         lk_mat=result["linkage_matrix"],
#         edge_set1=result["edge_set1"],
#     )
#
# ``P_original`` and ``lk_mat`` remain parameter names in the current
# edge-cutting module. They can be renamed separately without retaining P or
# P_new anywhere in the ensnarlment-construction API.
