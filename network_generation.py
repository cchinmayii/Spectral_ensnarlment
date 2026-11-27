"""
network_generation.py
=====================

Reusable construction utilities for the spatial network pairs used in the
ensnarlment notebooks.

The module separates three stages that were previously repeated across many
notebook cells:

1. Build one or two embedded NetworkX graphs.
2. Wrap those graphs in a Kirchhoff ``DualCircuit``.
3. Assign stable node/edge labels and retain the exact edge orderings used by
   downstream mesh-matrix and edge-cutting code.

Public constructors
-------------------
``create_hopf_link``
    Two polygonal loops embedded as a Hopf link.

``create_symmetric_ladders``
    Two linked ladder-like networks.

``create_dual_prism``
    Two regular prism skeletons with a rigid transformation applied to the
    first prism.

``create_grids``
    Two isomorphic rectangular grids with alternating out-of-plane node
    displacements. The second grid can be phase-inverted, rotated, and
    translated relative to the first.

``create_cubic``
    Two finite simple-cubic lattices with independently specified side
    lengths. The first lattice is translated by a user-specified offset.

``create_isomorphic_diamond``
    Two exactly isomorphic diamond networks generated either as conventional
    cubic supercells or as FCC-indexed finite clusters.

``create_isomorphic_laves``
    A repeated right- or left-handed Laves skeleton and an exact translated
    copy with the same handedness.

``create_double_gyroid_laves``
    A repeated Laves skeleton and a reflected copy with opposite handedness.

``create_voronoi``
    Two independently generated random Voronoi networks, or one Voronoi
    network and a topology-preserving re-embedding of it.

Important edge-ordering rule
----------------------------
Every returned Kirchhoff dual circuit ``D`` receives the attributes
``D.edge_set1`` and ``D.edge_set2``. These are the exact ordered edge lists
that should be used when constructing ``M1_dir`` and ``M2_dir`` and when
mapping cut indices back to graph edges.

The Laves constructors are intentionally isolated behind a small public API.
The provisional unit-cell geometry can therefore be refined later without
changing notebook calls.
"""

from __future__ import annotations

__version__ = "0.5.0-grids"

import copy
from itertools import combinations
import math
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism as iso
from scipy.spatial import cKDTree

import kirchhoff.circuit_dual as kcd
import kirchhoff.init_dual as kid
import kirchhoff.init_random as kir


PeriodSpecification = int | Sequence[int]
Chirality = Literal["right", "left", "R", "L"]
DiamondConstruction = Literal["conventional", "fcc_cluster"]


__all__ = [
    # Shared validation and transformation helpers
    "normalize_periods",
    "cycle_rank",
    "rotation_matrix_3d",
    "transform_graph_copy",
    "translated_graph_copy",
    "reflected_graph_copy",
    "prune_to_cycle_core",
    "assign_graph_labels",
    "set_print_labels",
    "extract_layer_graphs",
    "get_edge_sets",
    "validate_network_pair",
    "construct_dual_circuit",

    # Hopf-linked polygon loops
    "build_polygon_loop_graph",
    "build_hopf_link_graphs",
    "create_hopf_link",

    # Symmetric ladders
    "build_symmetric_ladder_graphs",
    "create_symmetric_ladders",

    # Rectangular grids
    "build_grid_graph",
    "build_grid_graphs",
    "create_grids",

    # Prisms
    "prism_vertex_coordinates",
    "build_prism_graph",
    "build_dual_prism_graphs",
    "create_dual_prism",

    # Cubic lattices
    "build_cubic_lattice",
    "build_dual_cubic_graphs",
    "create_cubic",

    # Diamond lattices
    "build_diamond_unit_cell",
    "diamond_fcc_translation_vectors",
    "build_diamond_conventional_supercell",
    "build_diamond_fcc_cluster",
    "build_diamond_lattice",
    "build_isomorphic_diamond_graphs",
    "create_isomorphic_diamond",

    # Laves / gyroid-skeleton lattices
    "build_laves_skeleton",
    "build_isomorphic_laves_graphs",
    "create_isomorphic_laves",
    "build_double_gyroid_laves_graphs",
    "create_double_gyroid_laves",

    # Random Voronoi networks
    "default_isomorphic_voronoi_embedding",
    "build_voronoi_graphs",
    "NetworkxDualVoronoi",
    "create_voronoi",
    "init_dual_voronoi",
    "create_label_voronoi",

    # Isomorphism and corresponding-cycle helpers
    "find_graph_isomorphism",
    "map_cycle_to_graph",
    "mapped_edge_in_graph2",
    "construct_corresponding_cycle_bases",
]


# ============================================================================
# Constants
# ============================================================================


DIAMOND_NEAREST_NEIGHBOUR_DISTANCE = np.sqrt(3.0) / 2.0
DIAMOND_TRANSLATION_LENGTH = 2.0

LAVES_NEAREST_NEIGHBOUR_DISTANCE = np.sqrt(2.0)
LAVES_TRANSLATION_LENGTH = 4.0

# Fundamental points used by Kirchhoff's Laves skeleton generator.
LAVES_FUNDAMENTAL_POINTS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 2.0, 1.0],
        [0.0, 3.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 2.0],
        [3.0, 0.0, 3.0],
        [2.0, 1.0, 3.0],
    ],
    dtype=float,
)


# ============================================================================
# General utilities
# ============================================================================


def normalize_periods(periods: PeriodSpecification) -> tuple[int, int, int]:
    """
    Convert a repetition specification to ``(px, py, pz)``.

    Parameters
    ----------
    periods
        Either one positive integer, used along all three axes, or a sequence
        of exactly three positive integers.

    Examples
    --------
    ``1`` becomes ``(1, 1, 1)``.
    ``(2, 3, 1)`` remains ``(2, 3, 1)``.
    """
    if isinstance(periods, (int, np.integer)):
        normalized = (int(periods),) * 3
    else:
        normalized = tuple(int(value) for value in periods)

    if len(normalized) != 3:
        raise ValueError(
            "periods must be a positive integer or a length-three sequence."
        )

    if any(value < 1 for value in normalized):
        raise ValueError("Every period count must be at least 1.")

    return normalized


def cycle_rank(graph: nx.Graph) -> int:
    """
    Return the dimension of an undirected graph's cycle space.

    For a graph with ``E`` edges, ``V`` vertices, and ``c`` connected
    components, the cycle rank is ``E - V + c``.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")

    return (
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )


def _as_vector3(values: Sequence[float], *, name: str) -> np.ndarray:
    """Convert an input to a finite three-dimensional float vector."""
    vector = np.asarray(values, dtype=float)

    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")

    return vector


def rotation_matrix_3d(
    rotation: Sequence[float],
) -> np.ndarray:
    """
    Construct a 3D Euler rotation matrix.

    Parameters
    ----------
    rotation
        Rotation angles ``(x, y, z)`` in degrees.

    Returns
    -------
    ndarray, shape (3, 3)
        Matrix applying rotations around x, then y, then z.
    """
    rotation = _as_vector3(rotation, name="rotation")
    rotation_x, rotation_y, rotation_z = np.radians(rotation)

    rotation_x_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rotation_x), -np.sin(rotation_x)],
            [0.0, np.sin(rotation_x), np.cos(rotation_x)],
        ]
    )

    rotation_y_matrix = np.asarray(
        [
            [np.cos(rotation_y), 0.0, np.sin(rotation_y)],
            [0.0, 1.0, 0.0],
            [-np.sin(rotation_y), 0.0, np.cos(rotation_y)],
        ]
    )

    rotation_z_matrix = np.asarray(
        [
            [np.cos(rotation_z), -np.sin(rotation_z), 0.0],
            [np.sin(rotation_z), np.cos(rotation_z), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rotation_z_matrix @ rotation_y_matrix @ rotation_x_matrix


def assign_graph_labels(graph: nx.Graph) -> nx.Graph:
    """
    Assign deterministic node and edge labels in current iteration order.

    The graph is modified in place and returned for convenience.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")

    for node_index, node in enumerate(graph.nodes()):
        graph.nodes[node]["label"] = node_index

        if "pos" in graph.nodes[node]:
            graph.nodes[node]["pos"] = np.asarray(
                graph.nodes[node]["pos"],
                dtype=float,
            )

    for edge_index, edge in enumerate(graph.edges()):
        graph.edges[edge]["label"] = edge_index

    return graph


def transform_graph_copy(
    graph: nx.Graph,
    matrix: np.ndarray | None = None,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> nx.Graph:
    """
    Return a topology-preserving affine copy of an embedded graph.

    Node IDs, node insertion order, edge insertion order, and non-geometric
    attributes are retained. Positions are transformed according to

    ``new_position = matrix @ old_position + offset``.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")

    if matrix is None:
        matrix = np.eye(3, dtype=float)
    else:
        matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3).")

    offset = _as_vector3(offset, name="offset")

    transformed = nx.Graph()
    transformed.graph.update(graph.graph)

    for node, node_data in graph.nodes(data=True):
        if "pos" not in node_data:
            raise KeyError(f"Node {node!r} has no 'pos' attribute.")

        copied_data = dict(node_data)
        old_position = _as_vector3(
            node_data["pos"],
            name=f"position of node {node!r}",
        )
        copied_data["pos"] = matrix @ old_position + offset
        transformed.add_node(node, **copied_data)

    # Add edges in the original graph's iteration order. This is important for
    # preserving row-to-edge correspondence in later mesh matrices.
    for u, v, edge_data in graph.edges(data=True):
        transformed.add_edge(u, v, **dict(edge_data))

    return transformed


def translated_graph_copy(
    graph: nx.Graph,
    coords_offset: Sequence[float],
) -> nx.Graph:
    """Return an exact translated copy of an embedded graph."""
    return transform_graph_copy(
        graph,
        matrix=np.eye(3),
        offset=coords_offset,
    )


def reflected_graph_copy(
    graph: nx.Graph,
    axis: Literal["x", "y", "z"] = "x",
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> nx.Graph:
    """
    Return a reflected, topology-preserving copy of an embedded graph.

    A reflection has determinant ``-1`` and therefore reverses handedness.
    """
    reflection_matrices = {
        "x": np.diag([-1.0, 1.0, 1.0]),
        "y": np.diag([1.0, -1.0, 1.0]),
        "z": np.diag([1.0, 1.0, -1.0]),
    }

    if axis not in reflection_matrices:
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    return transform_graph_copy(
        graph,
        matrix=reflection_matrices[axis],
        offset=offset,
    )


def prune_to_cycle_core(graph: nx.Graph) -> nx.Graph:
    """
    Return the graph's 2-core by repeatedly removing degree-0 and degree-1 nodes.

    The original graph is not modified. Surviving node and edge identifiers and
    attributes are retained.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")

    core = graph.copy()

    while True:
        removable_nodes = [
            node
            for node, degree in core.degree()
            if degree <= 1
        ]

        if not removable_nodes:
            break

        core.remove_nodes_from(removable_nodes)

    return assign_graph_labels(core)


def _unique_points(
    points: Iterable[Sequence[float]],
    decimals: int = 12,
) -> list[np.ndarray]:
    """Deduplicate points while preserving first-occurrence order."""
    unique: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()

    for point in points:
        array = _as_vector3(point, name="point")
        key = tuple(np.round(array, decimals=decimals))

        if key in seen:
            continue

        seen.add(key)
        unique.append(array)

    return unique


def _graph_from_points_at_distance(
    points: Iterable[Sequence[float]],
    target_distance: float,
    tolerance: float = 1e-10,
) -> nx.Graph:
    """Construct a graph by joining point pairs at one target distance."""
    if target_distance <= 0:
        raise ValueError("target_distance must be positive.")

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")

    unique_points = _unique_points(points)
    graph = nx.Graph()

    for node_index, position in enumerate(unique_points):
        graph.add_node(
            node_index,
            pos=np.asarray(position, dtype=float),
            label=node_index,
        )

    if len(unique_points) < 2:
        return graph

    positions = np.asarray(unique_points, dtype=float)
    tree = cKDTree(positions)

    candidate_pairs = tree.query_pairs(
        r=target_distance + tolerance,
        output_type="set",
    )

    edge_index = 0

    for node_i, node_j in sorted(candidate_pairs):
        distance = np.linalg.norm(
            positions[node_i] - positions[node_j]
        )

        if np.isclose(
            distance,
            target_distance,
            rtol=0.0,
            atol=tolerance,
        ):
            graph.add_edge(
                node_i,
                node_j,
                label=edge_index,
            )
            edge_index += 1

    return graph


def _graph_from_points_within_distance(
    points: Iterable[Sequence[float]],
    maximum_distance: float,
    tolerance: float = 1e-10,
) -> nx.Graph:
    """Construct a graph by joining distinct points within a cutoff distance."""
    if maximum_distance <= 0:
        raise ValueError("maximum_distance must be positive.")

    unique_points = _unique_points(points)
    graph = nx.Graph()

    for node_index, position in enumerate(unique_points):
        graph.add_node(
            node_index,
            pos=np.asarray(position, dtype=float),
            label=node_index,
        )

    if len(unique_points) < 2:
        return graph

    positions = np.asarray(unique_points, dtype=float)
    tree = cKDTree(positions)
    candidate_pairs = tree.query_pairs(
        r=maximum_distance + tolerance,
        output_type="set",
    )

    edge_index = 0

    for node_i, node_j in sorted(candidate_pairs):
        distance = np.linalg.norm(
            positions[node_i] - positions[node_j]
        )

        if tolerance < distance <= maximum_distance + tolerance:
            graph.add_edge(
                node_i,
                node_j,
                label=edge_index,
            )
            edge_index += 1

    return graph


# ============================================================================
# Kirchhoff wrapping and metadata
# ============================================================================


class _NetworkxGraphPair(kid.NetworkxDual):
    """Minimal Kirchhoff graph-set wrapper around two completed graphs."""

    def __init__(self, graph1: nx.Graph, graph2: nx.Graph):
        super().__init__()
        self.layer = [graph1, graph2]


def set_print_labels(D: Any) -> Any:
    """
    Synchronize graph labels, Kirchhoff table labels, and stored edge orderings.

    The returned object receives:

    ``D.edge_set1``
        Ordered edges from layer 1.
    ``D.edge_set2``
        Ordered edges from layer 2.
    ``D.graph_sets``
        The two underlying NetworkX graphs.
    """
    if not hasattr(D, "layer") or len(D.layer) != 2:
        raise TypeError("D must contain exactly two layers.")

    edge_sets: list[list[Any]] = []
    graph_sets: list[nx.Graph] = []

    for layer_index, circuit_layer in enumerate(D.layer):
        if not hasattr(circuit_layer, "G"):
            raise TypeError(
                f"D.layer[{layer_index}] does not expose a NetworkX graph as G."
            )

        graph = circuit_layer.G
        assign_graph_labels(graph)

        node_order = list(graph.nodes())
        edge_order = list(graph.edges())

        if len(circuit_layer.nodes) != len(node_order):
            raise RuntimeError(
                f"Layer {layer_index} node-table length does not match its graph."
            )

        if len(circuit_layer.edges) != len(edge_order):
            raise RuntimeError(
                f"Layer {layer_index} edge-table length does not match its graph."
            )

        circuit_layer.nodes["label"] = node_order
        circuit_layer.edges["label"] = edge_order

        graph_sets.append(graph)
        edge_sets.append(edge_order)

    D.graph_sets = graph_sets
    D.edge_set1 = edge_sets[0]
    D.edge_set2 = edge_sets[1]

    return D


def extract_layer_graphs(D: Any) -> tuple[nx.Graph, nx.Graph]:
    """Return the two NetworkX graphs stored in a Kirchhoff dual circuit."""
    if not hasattr(D, "layer") or len(D.layer) != 2:
        raise TypeError("D must contain exactly two layers.")

    graph1 = D.layer[0].G
    graph2 = D.layer[1].G

    if not isinstance(graph1, nx.Graph) or not isinstance(graph2, nx.Graph):
        raise TypeError("Both layers must contain NetworkX graphs.")

    return graph1, graph2


def get_edge_sets(D: Any) -> tuple[list[Any], list[Any]]:
    """
    Return stable edge orderings for the two circuit layers.

    If the metadata has not yet been attached, it is created by calling
    :func:`set_print_labels`.
    """
    if not hasattr(D, "edge_set1") or not hasattr(D, "edge_set2"):
        set_print_labels(D)

    return list(D.edge_set1), list(D.edge_set2)


def validate_network_pair(
    graph1: nx.Graph,
    graph2: nx.Graph,
    *,
    require_isomorphic: bool = False,
    require_positions: bool = True,
) -> dict[str, Any]:
    """Validate common structural properties of a generated graph pair."""
    for graph_number, graph in enumerate((graph1, graph2), start=1):
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"graph{graph_number} must be a NetworkX graph.")

        if require_positions:
            missing_positions = [
                node
                for node, data in graph.nodes(data=True)
                if "pos" not in data
            ]

            if missing_positions:
                raise ValueError(
                    f"graph{graph_number} has nodes without positions: "
                    f"{missing_positions[:10]}"
                )

    isomorphic = nx.is_isomorphic(graph1, graph2)

    if require_isomorphic and not isomorphic:
        raise ValueError("The generated graph pair is not isomorphic.")

    return {
        "isomorphic": isomorphic,
        "equal_node_count": (
            graph1.number_of_nodes() == graph2.number_of_nodes()
        ),
        "equal_edge_count": (
            graph1.number_of_edges() == graph2.number_of_edges()
        ),
        "connected1": (
            nx.is_connected(graph1) if graph1.number_of_nodes() else False
        ),
        "connected2": (
            nx.is_connected(graph2) if graph2.number_of_nodes() else False
        ),
        "cycle_rank1": cycle_rank(graph1),
        "cycle_rank2": cycle_rank(graph2),
        "number_of_nodes1": graph1.number_of_nodes(),
        "number_of_nodes2": graph2.number_of_nodes(),
        "number_of_edges1": graph1.number_of_edges(),
        "number_of_edges2": graph2.number_of_edges(),
    }


def construct_dual_circuit(
    graph1: nx.Graph,
    graph2: nx.Graph,
    *,
    network_type: str,
    circuit_type: str = "default",
    generation_parameters: Mapping[str, Any] | None = None,
    require_isomorphic: bool = False,
) -> Any:
    """
    Wrap two completed embedded graphs in a Kirchhoff ``DualCircuit``.
    """
    assign_graph_labels(graph1)
    assign_graph_labels(graph2)

    validation = validate_network_pair(
        graph1,
        graph2,
        require_isomorphic=require_isomorphic,
    )

    graph_set = _NetworkxGraphPair(graph1, graph2)
    circuit_set = kcd.construct_from_graphSet(
        graph_set,
        circuit_type=circuit_type,
    )

    for circuit_layer, graph in zip(circuit_set, graph_set.layer):
        circuit_layer.info = circuit_layer.set_info(
            graph,
            network_type,
        )

    D = kcd.DualCircuit(circuit_set)
    D.network_type = network_type
    D.generation_parameters = dict(generation_parameters or {})
    D.generation_validation = validation

    return set_print_labels(D)


# ============================================================================
# Hopf-linked polygon loops
# ============================================================================


def build_polygon_loop_graph(
    number_of_edges: int,
    *,
    radius: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
) -> nx.Graph:
    """
    Build one regular polygonal loop embedded in three dimensions.

    The polygon is first constructed counterclockwise in the xy-plane,
    centered at the origin. A rigid Euler rotation and translation are then
    applied to every vertex.

    Parameters
    ----------
    number_of_edges
        Number of polygon edges and vertices. Must be at least three.
    radius
        Circumradius of the polygon.
    center
        Final center of the polygon as ``(x, y, z)``.
    rotation
        Euler rotation angles ``(x, y, z)`` in degrees. Rotations are applied
        around x, then y, then z.

    Returns
    -------
    networkx.Graph
        Embedded cycle graph with deterministic node and edge labels.
    """
    number_of_edges = int(number_of_edges)
    radius = float(radius)

    if number_of_edges < 3:
        raise ValueError(
            "number_of_edges must be at least 3 for a polygonal loop."
        )

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be a positive finite number.")

    center_vector = _as_vector3(center, name="center")
    rotation_matrix = rotation_matrix_3d(rotation)

    graph = nx.Graph()

    for node_index in range(number_of_edges):
        angle = 2.0 * np.pi * node_index / number_of_edges

        local_position = np.asarray(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0,
            ],
            dtype=float,
        )

        position = rotation_matrix @ local_position + center_vector

        graph.add_node(
            node_index,
            pos=position,
            label=node_index,
        )

    for edge_index in range(number_of_edges):
        graph.add_edge(
            edge_index,
            (edge_index + 1) % number_of_edges,
            label=edge_index,
        )

    return graph


def build_hopf_link_graphs(
    n_edges: int | Sequence[int],
    *,
    radius1: float = 1.0,
    radius2: float = 1.0,
    center1: Sequence[float] = (0.0, 0.0, 0.0),
    center2: Sequence[float] = (1.0, 0.0, 0.0),
    rotation1: Sequence[float] = (0.0, 0.0, 0.0),
    rotation2: Sequence[float] = (90.0, 0.0, 0.0),
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build two polygonal loops in the standard Hopf-link configuration.

    By default, graph 1 is a unit-radius polygon in the xy-plane centered at
    the origin. Graph 2 is a unit-radius polygon centered at ``(1, 0, 0)`` and
    rotated by 90 degrees around the x-axis, placing it in the xz-plane.

    Parameters
    ----------
    n_edges
        Either one integer, used for both loops, or a length-two sequence
        ``(n_edges1, n_edges2)``.
    radius1, radius2
        Circumradii of the two polygonal loops.
    center1, center2
        Loop centers in three-dimensional coordinates.
    rotation1, rotation2
        Euler rotations in degrees for the two loops.

    Returns
    -------
    graph1, graph2
        Two independently embedded cycle graphs.

    Notes
    -----
    The default geometry reproduces the Hopf-link construction previously
    used in the notebooks, but exposes the geometry as explicit parameters.
    """
    if isinstance(n_edges, (int, np.integer)):
        edge_counts = (int(n_edges), int(n_edges))
    else:
        edge_counts = tuple(int(value) for value in n_edges)

    if len(edge_counts) != 2:
        raise ValueError(
            "n_edges must be an integer or a length-two sequence."
        )

    if any(number_of_edges < 3 for number_of_edges in edge_counts):
        raise ValueError("Each Hopf-link loop must have at least 3 edges.")

    graph1 = build_polygon_loop_graph(
        edge_counts[0],
        radius=radius1,
        center=center1,
        rotation=rotation1,
    )

    graph2 = build_polygon_loop_graph(
        edge_counts[1],
        radius=radius2,
        center=center2,
        rotation=rotation2,
    )

    return graph1, graph2


def create_hopf_link(
    n_edges: int | Sequence[int],
    *,
    radius1: float = 1.0,
    radius2: float = 1.0,
    center1: Sequence[float] = (0.0, 0.0, 0.0),
    center2: Sequence[float] = (1.0, 0.0, 0.0),
    rotation1: Sequence[float] = (0.0, 0.0, 0.0),
    rotation2: Sequence[float] = (90.0, 0.0, 0.0),
    circuit_type: str = "default",
) -> Any:
    """Create a labeled Kirchhoff dual circuit containing one Hopf link."""
    graph1, graph2 = build_hopf_link_graphs(
        n_edges=n_edges,
        radius1=radius1,
        radius2=radius2,
        center1=center1,
        center2=center2,
        rotation1=rotation1,
        rotation2=rotation2,
    )

    edge_counts = (
        graph1.number_of_edges(),
        graph2.number_of_edges(),
    )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type="hopf_link",
        circuit_type=circuit_type,
        generation_parameters={
            "n_edges": edge_counts,
            "radius1": float(radius1),
            "radius2": float(radius2),
            "center1": tuple(float(value) for value in center1),
            "center2": tuple(float(value) for value in center2),
            "rotation1": tuple(float(value) for value in rotation1),
            "rotation2": tuple(float(value) for value in rotation2),
        },
        require_isomorphic=edge_counts[0] == edge_counts[1],
    )


# ============================================================================
# Symmetric ladders
# ============================================================================


def _build_symmetric_ladder_layer(
    *,
    unit_base: np.ndarray,
    unit_full: np.ndarray,
    unit_half: np.ndarray,
    number_of_period_blocks: int,
    period_vector: np.ndarray,
    use_half_final_block: bool,
    edge_length: float,
    tolerance: float,
) -> nx.Graph:
    """Build one ladder layer from its repeated coordinate motifs."""
    points: list[np.ndarray] = [
        np.asarray(point, dtype=float)
        for point in unit_base
    ]

    for period_index in range(number_of_period_blocks):
        shift = period_index * period_vector

        if use_half_final_block and period_index == number_of_period_blocks - 1:
            unit = unit_half
        else:
            unit = unit_full

        points.extend(
            np.asarray(point, dtype=float) + shift
            for point in unit
        )

    return _graph_from_points_within_distance(
        points,
        maximum_distance=edge_length,
        tolerance=tolerance,
    )


def build_symmetric_ladder_graphs(
    num_loops: int,
    edge_length: float = 1.0,
    tolerance: float = 1e-4,
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build the two linked symmetric ladder graphs.

    ``num_loops`` may be odd or even. For odd values, the final repeated motif
    is truncated to produce the requested number of loops.
    """
    num_loops = int(num_loops)

    if num_loops < 1:
        raise ValueError("num_loops must be at least 1.")

    number_of_period_blocks = (num_loops + 1) // 2
    use_half_final_block = num_loops % 2 == 1

    unit_a_base = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    unit_a_full = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    unit_a_half = unit_a_full[:2]

    unit_b_base = np.asarray(
        [[0.5, 0.5, -0.5], [0.5, 1.5, -0.5]]
    )
    unit_b_full = np.asarray(
        [
            [0.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
            [1.5, 0.5, 0.5],
            [1.5, 1.5, 0.5],
        ]
    )
    unit_b_half = unit_b_full[:2]

    period_vector = np.asarray([1.0, 0.0, 1.0])

    graph1 = _build_symmetric_ladder_layer(
        unit_base=unit_a_base,
        unit_full=unit_a_full,
        unit_half=unit_a_half,
        number_of_period_blocks=number_of_period_blocks,
        period_vector=period_vector,
        use_half_final_block=use_half_final_block,
        edge_length=edge_length,
        tolerance=tolerance,
    )

    graph2 = _build_symmetric_ladder_layer(
        unit_base=unit_b_base,
        unit_full=unit_b_full,
        unit_half=unit_b_half,
        number_of_period_blocks=number_of_period_blocks,
        period_vector=period_vector,
        use_half_final_block=use_half_final_block,
        edge_length=edge_length,
        tolerance=tolerance,
    )

    return graph1, graph2


def create_symmetric_ladders(
    num_loops: int,
    edge_length: float = 1.0,
    tolerance: float = 1e-4,
    circuit_type: str = "default",
) -> Any:
    """Create a labeled Kirchhoff dual circuit of symmetric ladders."""
    graph1, graph2 = build_symmetric_ladder_graphs(
        num_loops=num_loops,
        edge_length=edge_length,
        tolerance=tolerance,
    )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type="symmetric_ladders",
        circuit_type=circuit_type,
        generation_parameters={
            "num_loops": int(num_loops),
            "edge_length": float(edge_length),
            "tolerance": float(tolerance),
        },
    )


# ============================================================================
# Prisms
# ============================================================================


def prism_vertex_coordinates(n: int) -> list[np.ndarray]:
    """
    Return the vertices of a regular unit-edge prism with ``n``-gonal ends.
    """
    n = int(n)

    if n < 3:
        raise ValueError("A prism requires at least three vertices per face.")

    angular_step = 2.0 * np.pi / n
    radius = 1.0 / (2.0 * np.sin(angular_step / 2.0))

    coordinates: list[np.ndarray] = []

    for z_coordinate in (0.5, -0.5):
        for vertex_index in range(n):
            angle = vertex_index * angular_step + angular_step / 2.0
            coordinates.append(
                np.asarray(
                    [
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        z_coordinate,
                    ],
                    dtype=float,
                )
            )

    return coordinates


def build_prism_graph(n: int) -> nx.Graph:
    """Build the skeleton graph of a regular ``n``-gonal prism."""
    n = int(n)
    coordinates = prism_vertex_coordinates(n)
    graph = nx.Graph()

    for node_index, position in enumerate(coordinates):
        graph.add_node(
            node_index,
            pos=position,
            label=node_index,
        )

    ordered_edges: list[tuple[int, int]] = []

    # Top and bottom polygonal faces.
    for face_start in (0, n):
        for vertex_index in range(n):
            ordered_edges.append(
                (
                    face_start + vertex_index,
                    face_start + ((vertex_index + 1) % n),
                )
            )

    # Vertical edges joining corresponding face vertices.
    ordered_edges.extend(
        (vertex_index, vertex_index + n)
        for vertex_index in range(n)
    )

    for edge_index, (u, v) in enumerate(ordered_edges):
        graph.add_edge(u, v, label=edge_index)

    return graph


def build_dual_prism_graphs(
    size: Sequence[int],
    coords_offset: Sequence[float],
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build two prism skeletons and transform the first prism.

    This preserves the behavior of the original notebook implementation:
    rotation and translation are applied to graph 1, while graph 2 remains in
    its canonical orientation at the origin.
    """
    if len(size) != 2:
        raise ValueError("size must contain two prism face sizes.")

    prism_size1, prism_size2 = (int(value) for value in size)

    graph1_base = build_prism_graph(prism_size1)
    graph2 = build_prism_graph(prism_size2)

    graph1 = transform_graph_copy(
        graph1_base,
        matrix=rotation_matrix_3d(rotation),
        offset=coords_offset,
    )

    assign_graph_labels(graph1)
    assign_graph_labels(graph2)

    return graph1, graph2


def create_dual_prism(
    size: Sequence[int],
    coords_offset: Sequence[float],
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    circuit_type: str = "default",
) -> Any:
    """Create a labeled Kirchhoff dual circuit containing two prisms."""
    graph1, graph2 = build_dual_prism_graphs(
        size=size,
        coords_offset=coords_offset,
        rotation=rotation,
    )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type="dual_prism",
        circuit_type=circuit_type,
        generation_parameters={
            "size": tuple(int(value) for value in size),
            "coords_offset": tuple(float(value) for value in coords_offset),
            "rotation": tuple(float(value) for value in rotation),
        },
    )


# ============================================================================
# Rectangular grids
# ============================================================================


def _normalize_grid_pair_parameter(
    value: float | Sequence[float],
    *,
    name: str,
    minimum: float,
    strictly_greater: bool,
) -> tuple[float, float]:
    """
    Normalize a scalar or length-two parameter for the two grid layers.

    A scalar is applied to both layers. A sequence is interpreted as
    ``(value1, value2)``.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        normalized = (float(value), float(value))
    else:
        normalized = tuple(float(item) for item in value)

        if len(normalized) != 2:
            raise ValueError(
                f"{name} must be one number or a two-element sequence."
            )

    if not all(np.isfinite(item) for item in normalized):
        raise ValueError(f"{name} must contain only finite values.")

    if strictly_greater:
        if not all(item > minimum for item in normalized):
            raise ValueError(
                f"Both {name} values must be greater than {minimum}."
            )
    else:
        if not all(item >= minimum for item in normalized):
            raise ValueError(
                f"Both {name} values must be at least {minimum}."
            )

    return normalized


def build_grid_graph(
    num_loops: tuple[int, int],
    *,
    xy_spacing: float = 1.0,
    height: float = 1.0,
    checkerboard_phase: int = 0,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> nx.Graph:
    """
    Build one rectangular grid with alternating out-of-plane node heights.

    The graph begins as an ordinary rectangular grid in the xy-plane. Existing
    grid nodes are displaced to ``z=+height`` or ``z=-height`` according to a
    checkerboard pattern. No additional nodes are introduced.

    Parameters
    ----------
    num_loops
        ``(n, m)``, where ``n`` and ``m`` are the numbers of elementary loops
        along x and y. The graph contains ``(n + 1) * (m + 1)`` nodes and
        ``n * m`` elementary loops.
    xy_spacing
        Separation between adjacent grid coordinates in the xy-plane.
    height
        Magnitude of the out-of-plane displacement.
    checkerboard_phase
        ``0`` places grid node ``(0, 0)`` at ``z=+height`` and ``1`` places it
        at ``z=-height``. Other integers are reduced modulo 2.
    offset
        Global translation applied after constructing the grid.

    Returns
    -------
    nx.Graph
        Embedded grid with deterministic integer node IDs, stable edge
        ordering, and three-dimensional ``pos`` attributes.
    """
    try:
        n_loops_x, n_loops_y = num_loops
    except (TypeError, ValueError) as error:
        raise ValueError(
            "num_loops must be a two-element sequence "
            "(n_loops_x, n_loops_y)."
        ) from error

    n_loops_x = int(n_loops_x)
    n_loops_y = int(n_loops_y)
    xy_spacing = float(xy_spacing)
    height = float(height)
    checkerboard_phase = int(checkerboard_phase) % 2
    offset = _as_vector3(offset, name="offset")

    if n_loops_x < 1 or n_loops_y < 1:
        raise ValueError("Both loop counts must be at least 1.")

    if not np.isfinite(xy_spacing) or xy_spacing <= 0.0:
        raise ValueError("xy_spacing must be a positive finite number.")

    if not np.isfinite(height) or height < 0.0:
        raise ValueError("height must be a nonnegative finite number.")

    base_grid = nx.grid_2d_graph(
        n_loops_x + 1,
        n_loops_y + 1,
        periodic=False,
    )

    # Explicit index order fixes node IDs and therefore downstream edge order.
    grid_indices = [
        (x_index, y_index)
        for x_index in range(n_loops_x + 1)
        for y_index in range(n_loops_y + 1)
    ]
    node_from_grid_index = {
        grid_index: node_index
        for node_index, grid_index in enumerate(grid_indices)
    }

    graph = nx.Graph()

    for x_index, y_index in grid_indices:
        node_index = node_from_grid_index[(x_index, y_index)]

        height_sign = (
            1
            if (x_index + y_index + checkerboard_phase) % 2 == 0
            else -1
        )

        position = offset + np.asarray(
            [
                x_index * xy_spacing,
                y_index * xy_spacing,
                height_sign * height,
            ],
            dtype=float,
        )

        graph.add_node(
            node_index,
            pos=position,
            label=node_index,
            grid_index=(x_index, y_index),
            height_sign=height_sign,
        )

    ordered_edges = sorted(
        (
            min(
                node_from_grid_index[grid_u],
                node_from_grid_index[grid_v],
            ),
            max(
                node_from_grid_index[grid_u],
                node_from_grid_index[grid_v],
            ),
        )
        for grid_u, grid_v in base_grid.edges()
    )

    nearest_neighbor_distance = float(
        np.sqrt(
            xy_spacing**2
            + (2.0 * height) ** 2
        )
    )

    for edge_index, (node_u, node_v) in enumerate(ordered_edges):
        position_u = graph.nodes[node_u]["pos"]
        position_v = graph.nodes[node_v]["pos"]
        edge_length = float(
            np.linalg.norm(position_v - position_u)
        )

        if not np.isclose(
            edge_length,
            nearest_neighbor_distance,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Unexpected edge length while constructing the grid."
            )

        graph.add_edge(
            node_u,
            node_v,
            label=edge_index,
            length=edge_length,
        )

    graph.graph["network_type"] = "grid"
    graph.graph["num_loops"] = (n_loops_x, n_loops_y)
    graph.graph["xy_spacing"] = xy_spacing
    graph.graph["height"] = height
    graph.graph["checkerboard_phase"] = checkerboard_phase
    graph.graph["offset"] = tuple(float(value) for value in offset)
    graph.graph["nearest_neighbor_distance"] = nearest_neighbor_distance

    return graph


def build_grid_graphs(
    num_loops: tuple[int, int],
    *,
    xy_spacing: float | Sequence[float] = (1.0, 1.0),
    height: float | Sequence[float] = (1.0, 1.0),
    layer_offset: Sequence[float] = (0.5, 0.5, 0.1),
    second_rotation: Sequence[float] = (90.0, 0.0, 0.0),
    invert_second: bool = True,
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build two isomorphic rectangular grids with independent embeddings.

    Parameters
    ----------
    num_loops
        ``(n, m)`` gives the numbers of elementary loops along x and y.
    xy_spacing
        One positive number for both grids or
        ``(xy_spacing1, xy_spacing2)``.
    height
        One nonnegative number for both grids or ``(height1, height2)``.
    layer_offset
        Translation applied to graph 2 after rotation.
    second_rotation
        Euler rotation angles ``(x, y, z)`` in degrees applied to graph 2
        around the centre of its bounding box.
    invert_second
        If True, graph 2 uses the opposite checkerboard phase.

    Returns
    -------
    graph1, graph2
        Two isomorphic NetworkX graphs with different 3D embeddings.
    """
    xy_spacing1, xy_spacing2 = _normalize_grid_pair_parameter(
        xy_spacing,
        name="xy_spacing",
        minimum=0.0,
        strictly_greater=True,
    )
    height1, height2 = _normalize_grid_pair_parameter(
        height,
        name="height",
        minimum=0.0,
        strictly_greater=False,
    )

    layer_offset = _as_vector3(
        layer_offset,
        name="layer_offset",
    )
    second_rotation = _as_vector3(
        second_rotation,
        name="second_rotation",
    )

    graph1 = build_grid_graph(
        num_loops=num_loops,
        xy_spacing=xy_spacing1,
        height=height1,
        checkerboard_phase=0,
    )

    graph2_base = build_grid_graph(
        num_loops=num_loops,
        xy_spacing=xy_spacing2,
        height=height2,
        checkerboard_phase=1 if invert_second else 0,
    )

    positions2 = np.asarray(
        [
            node_data["pos"]
            for _, node_data in graph2_base.nodes(data=True)
        ],
        dtype=float,
    )

    # Rotate graph 2 around its bounding-box centre, then translate it.
    rotation_center = (
        positions2.min(axis=0)
        + positions2.max(axis=0)
    ) / 2.0

    rotation_matrix = rotation_matrix_3d(
        second_rotation
    )

    # transform_graph_copy applies:
    # new_position = R @ old_position + affine_offset.
    affine_offset = (
        rotation_center
        + layer_offset
        - rotation_matrix @ rotation_center
    )

    graph2 = transform_graph_copy(
        graph2_base,
        matrix=rotation_matrix,
        offset=affine_offset,
    )

    return graph1, graph2


def create_grids(
    num_loops: tuple[int, int],
    *,
    xy_spacing: float | Sequence[float] = (1.0, 1.0),
    height: float | Sequence[float] = (1.0, 1.0),
    layer_offset: Sequence[float] = (0.5, 0.5, 0.1),
    second_rotation: Sequence[float] = (90.0, 0.0, 0.0),
    invert_second: bool = True,
    circuit_type: str = "default",
) -> Any:
    """
    Create a Kirchhoff dual circuit containing two isomorphic grids.

    ``xy_spacing`` and ``height`` may each be specified once for both layers
    or separately as a length-two sequence.
    """
    normalized_xy_spacing = _normalize_grid_pair_parameter(
        xy_spacing,
        name="xy_spacing",
        minimum=0.0,
        strictly_greater=True,
    )
    normalized_height = _normalize_grid_pair_parameter(
        height,
        name="height",
        minimum=0.0,
        strictly_greater=False,
    )
    layer_offset = _as_vector3(
        layer_offset,
        name="layer_offset",
    )
    second_rotation = _as_vector3(
        second_rotation,
        name="second_rotation",
    )

    graph1, graph2 = build_grid_graphs(
        num_loops=num_loops,
        xy_spacing=normalized_xy_spacing,
        height=normalized_height,
        layer_offset=layer_offset,
        second_rotation=second_rotation,
        invert_second=invert_second,
    )

    nearest_neighbor_distances = tuple(
        float(
            np.sqrt(
                spacing**2
                + (2.0 * layer_height) ** 2
            )
        )
        for spacing, layer_height in zip(
            normalized_xy_spacing,
            normalized_height,
        )
    )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type="grids",
        circuit_type=circuit_type,
        generation_parameters={
            "num_loops": tuple(
                int(value)
                for value in num_loops
            ),
            "xy_spacing": normalized_xy_spacing,
            "height": normalized_height,
            "nearest_neighbor_distances": nearest_neighbor_distances,
            "layer_offset": tuple(
                float(value)
                for value in layer_offset
            ),
            "second_rotation": tuple(
                float(value)
                for value in second_rotation
            ),
            "invert_second": bool(invert_second),
        },
        require_isomorphic=True,
    )



# ============================================================================
# Cubic lattices
# ============================================================================

def build_cubic_lattice(period: int = 1) -> nx.Graph:
    """
    Build the skeleton of a finite simple-cubic lattice.

    ``period`` is the side length measured in unit cubes:

    - ``period=1`` produces one unit cube.
    - ``period=2`` produces a 2 x 2 x 2 array of unit cubes.
    - ``period=n`` produces an n x n x n array of unit cubes.

    The lattice contains ``(period + 1)^3`` vertices. Nearest-neighbour
    vertices one unit apart along the x, y, or z direction are connected.

    Parameters
    ----------
    period
        Positive integer giving the number of unit cubes along each axis.
    """
    if not isinstance(period, (int, np.integer)):
        raise TypeError("period must be an integer.")

    period = int(period)

    if period < 1:
        raise ValueError("period must be at least 1.")

    graph = nx.Graph()
    node_from_coordinate: dict[tuple[int, int, int], int] = {}

    # Create vertices in deterministic x-y-z order.
    node_index = 0

    for x_coordinate in range(period + 1):
        for y_coordinate in range(period + 1):
            for z_coordinate in range(period + 1):
                coordinate = (
                    x_coordinate,
                    y_coordinate,
                    z_coordinate,
                )
                node_from_coordinate[coordinate] = node_index

                graph.add_node(
                    node_index,
                    pos=np.asarray(coordinate, dtype=float),
                    label=node_index,
                )
                node_index += 1

    # Add every nearest-neighbour edge exactly once.
    ordered_edges: list[tuple[int, int]] = []

    for x_coordinate in range(period + 1):
        for y_coordinate in range(period + 1):
            for z_coordinate in range(period + 1):
                current_node = node_from_coordinate[
                    (x_coordinate, y_coordinate, z_coordinate)
                ]

                if x_coordinate < period:
                    ordered_edges.append(
                        (
                            current_node,
                            node_from_coordinate[
                                (
                                    x_coordinate + 1,
                                    y_coordinate,
                                    z_coordinate,
                                )
                            ],
                        )
                    )

                if y_coordinate < period:
                    ordered_edges.append(
                        (
                            current_node,
                            node_from_coordinate[
                                (
                                    x_coordinate,
                                    y_coordinate + 1,
                                    z_coordinate,
                                )
                            ],
                        )
                    )

                if z_coordinate < period:
                    ordered_edges.append(
                        (
                            current_node,
                            node_from_coordinate[
                                (
                                    x_coordinate,
                                    y_coordinate,
                                    z_coordinate + 1,
                                )
                            ],
                        )
                    )

    for edge_index, (node_u, node_v) in enumerate(ordered_edges):
        graph.add_edge(
            node_u,
            node_v,
            label=edge_index,
        )

    return graph


def build_dual_cubic_graphs(
    periods: Sequence[int],
    coords_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build two finite simple-cubic lattices.

    ``periods=(n, m)`` gives graph 1 side length ``n`` and graph 2 side
    length ``m``. ``coords_offset=(x, y, z)`` translates graph 1, while
    graph 2 remains at the origin. This matches the transformation convention
    used by ``build_dual_prism_graphs``.

    Parameters
    ----------
    periods
        Two positive integers ``(n, m)`` specifying the side lengths of the
        first and second cubic lattices in unit cubes.
    coords_offset
        Translation ``(x, y, z)`` applied to graph 1.
    """
    if len(periods) != 2:
        raise ValueError(
            "periods must contain exactly two values: (period1, period2)."
        )

    period1, period2 = periods

    graph1_base = build_cubic_lattice(period1)
    graph2 = build_cubic_lattice(period2)

    graph1 = translated_graph_copy(
        graph1_base,
        coords_offset=coords_offset,
    )

    assign_graph_labels(graph1)
    assign_graph_labels(graph2)

    return graph1, graph2


def create_cubic(
    periods: Sequence[int],
    coords_offset: Sequence[float] = (0.0, 0.0, 0.0),
    circuit_type: str = "default",
) -> Any:
    """
    Create a labeled Kirchhoff dual circuit containing two cubic lattices.

    Examples
    --------
    Two unit cubes, with graph 1 translated by ``(0.5, 0.5, 0.5)``:

    ``D = create_cubic(periods=(1, 1),
                      coords_offset=(0.5, 0.5, 0.5))``

    A side-length-2 lattice and a unit cube:

    ``D = create_cubic(periods=(2, 1),
                      coords_offset=(1.0, 1.0, 1.0))``
    """
    graph1, graph2 = build_dual_cubic_graphs(
        periods=periods,
        coords_offset=coords_offset,
    )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type="cubic",
        circuit_type=circuit_type,
        generation_parameters={
            "periods": tuple(int(value) for value in periods),
            "coords_offset": tuple(
                float(value) for value in coords_offset
            ),
        },
    )


# ============================================================================
# Diamond lattices
# ============================================================================


def build_diamond_unit_cell() -> nx.Graph:
    """
    Construct one conventional cubic diamond-cell generating graph.

    The coordinate scale uses a conventional cubic side length of ``2`` and
    therefore a nearest-neighbour distance of ``sqrt(3) / 2``. The returned
    open-boundary graph contains selected boundary representatives; coincident
    representatives are merged when cells or motifs are repeated.
    """
    base_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )

    subcell_translations = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    points = [
        point + translation
        for translation in subcell_translations
        for point in base_points
    ]

    return _graph_from_points_at_distance(
        points,
        target_distance=DIAMOND_NEAREST_NEIGHBOUR_DISTANCE,
        tolerance=1e-10,
    )


def diamond_fcc_translation_vectors(size: int) -> np.ndarray:
    """
    Return FCC translation vectors in a finite Cartesian index window.

    Parameters
    ----------
    size
        Positive integer defining ``0 <= i, j, k < size``. Only index triples
        satisfying ``i + j + k`` even are retained.

    Returns
    -------
    ndarray, shape (n, 3)
        Accepted translation vectors in deterministic lexicographic order.

    Notes
    -----
    ``size`` is the side length of the translation-index window. It is not the
    number of conventional cubic diamond cells along each axis.
    """
    if isinstance(size, (bool, np.bool_)) or not isinstance(
        size,
        (int, np.integer),
    ):
        raise TypeError("size must be an integer.")

    size = int(size)

    if size < 1:
        raise ValueError("size must be at least 1.")

    return np.asarray(
        [
            [i, j, k]
            for i in range(size)
            for j in range(size)
            for k in range(size)
            if (i + j + k) % 2 == 0
        ],
        dtype=float,
    )


def build_diamond_conventional_supercell(
    periods: PeriodSpecification = 1,
    distance_tolerance: float = 1e-10,
) -> nx.Graph:
    """
    Build an open-boundary conventional cubic diamond supercell.

    ``periods=(px, py, pz)`` repeats the side-length-2 conventional-cell
    generating graph along the Cartesian axes using origins

    ``2 * (i, j, k)``,

    with ``0 <= i < px``, ``0 <= j < py``, and ``0 <= k < pz``. Coincident
    boundary vertices are merged before nearest-neighbour edges are generated
    globally.

    Parameters
    ----------
    periods
        One positive integer, used along all three axes, or a length-three
        sequence giving the number of conventional cubic cells along x, y,
        and z.
    distance_tolerance
        Absolute tolerance for nearest-neighbour distance tests.
    """
    periods = normalize_periods(periods)
    unit_cell = build_diamond_unit_cell()
    unit_points = [
        np.asarray(data["pos"], dtype=float)
        for _, data in unit_cell.nodes(data=True)
    ]

    repeated_points: list[np.ndarray] = []

    for period_x in range(periods[0]):
        for period_y in range(periods[1]):
            for period_z in range(periods[2]):
                translation = DIAMOND_TRANSLATION_LENGTH * np.asarray(
                    [period_x, period_y, period_z],
                    dtype=float,
                )

                repeated_points.extend(
                    point + translation
                    for point in unit_points
                )

    return _graph_from_points_at_distance(
        repeated_points,
        target_distance=DIAMOND_NEAREST_NEIGHBOUR_DISTANCE,
        tolerance=distance_tolerance,
    )


def build_diamond_fcc_cluster(
    size: int = 2,
    distance_tolerance: float = 1e-10,
) -> nx.Graph:
    """
    Build an open-boundary diamond cluster using FCC translations.

    The conventional-cell generating graph is placed at all integer
    translations ``(i, j, k)`` satisfying

    ``0 <= i, j, k < size`` and ``i + j + k`` even.

    Coincident points are merged and nearest-neighbour edges are then
    reconstructed globally at distance ``sqrt(3) / 2``.

    Parameters
    ----------
    size
        Side length of the Cartesian FCC translation-index window. This is
        not a conventional-cell repetition count.
    distance_tolerance
        Absolute tolerance for nearest-neighbour distance tests.

    Notes
    -----
    This reproduces the raw finite-cluster geometry of the SnarlPy diamond
    placement rule. It does not apply SnarlPy's later cross-layer affiliation
    pruning.
    """
    translations = diamond_fcc_translation_vectors(size)
    unit_cell = build_diamond_unit_cell()
    unit_points = [
        np.asarray(data["pos"], dtype=float)
        for _, data in unit_cell.nodes(data=True)
    ]

    repeated_points = [
        point + translation
        for translation in translations
        for point in unit_points
    ]

    return _graph_from_points_at_distance(
        repeated_points,
        target_distance=DIAMOND_NEAREST_NEIGHBOUR_DISTANCE,
        tolerance=distance_tolerance,
    )


def build_diamond_lattice(
    periods: PeriodSpecification = 1,
    distance_tolerance: float = 1e-10,
) -> nx.Graph:
    """
    Build a conventional cubic diamond supercell.

    This backward-compatible name is an alias for
    :func:`build_diamond_conventional_supercell`. New code should prefer the
    explicit constructor name so that it cannot be confused with the
    FCC-indexed finite-cluster construction.
    """
    return build_diamond_conventional_supercell(
        periods=periods,
        distance_tolerance=distance_tolerance,
    )


def build_isomorphic_diamond_graphs(
    periods: PeriodSpecification = 1,
    coords_offset: Sequence[float] = (1.0, 0.0, 0.0),
    distance_tolerance: float = 1e-10,
    construction: DiamondConstruction = "conventional",
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build a diamond graph and an exact translated isomorphic copy.

    Parameters
    ----------
    periods
        For ``construction="conventional"``, one positive integer or a
        length-three sequence specifying conventional cubic cells along each
        Cartesian axis.

        For ``construction="fcc_cluster"``, one positive integer specifying
        the side length of the FCC translation-index window.
    coords_offset
        Translation applied to the second graph.
    distance_tolerance
        Absolute tolerance for nearest-neighbour distance tests.
    construction
        ``"conventional"`` for a Cartesian conventional-cell supercell or
        ``"fcc_cluster"`` for an FCC-indexed finite cluster.
    """
    if construction == "conventional":
        graph1 = build_diamond_conventional_supercell(
            periods=periods,
            distance_tolerance=distance_tolerance,
        )
    elif construction == "fcc_cluster":
        if isinstance(periods, (bool, np.bool_)) or not isinstance(
            periods,
            (int, np.integer),
        ):
            raise TypeError(
                "For construction='fcc_cluster', periods must be one integer."
            )

        graph1 = build_diamond_fcc_cluster(
            size=int(periods),
            distance_tolerance=distance_tolerance,
        )
    else:
        raise ValueError(
            "construction must be 'conventional' or 'fcc_cluster'."
        )

    graph2 = translated_graph_copy(
        graph1,
        coords_offset=coords_offset,
    )

    return graph1, graph2


def create_isomorphic_diamond(
    periods: PeriodSpecification = 1,
    coords_offset: Sequence[float] = (1.0, 0.0, 0.0),
    distance_tolerance: float = 1e-10,
    circuit_type: str = "default",
    construction: DiamondConstruction = "conventional",
) -> Any:
    """
    Create a labeled Kirchhoff circuit of two isomorphic diamond networks.

    The default remains the backward-compatible conventional cubic supercell
    construction. Set ``construction="fcc_cluster"`` to use the FCC
    translation-index convention.
    """
    graph1, graph2 = build_isomorphic_diamond_graphs(
        periods=periods,
        coords_offset=coords_offset,
        distance_tolerance=distance_tolerance,
        construction=construction,
    )

    if construction == "conventional":
        normalized_periods: int | tuple[int, int, int] = normalize_periods(
            periods
        )
        network_type = "isomorphic_diamond"
    elif construction == "fcc_cluster":
        # Input validation has already been performed by the graph builder.
        normalized_periods = int(periods)
        network_type = "isomorphic_diamond_fcc_cluster"
    else:
        # Retained defensively in case the graph-building implementation is
        # changed independently in the future.
        raise ValueError(
            "construction must be 'conventional' or 'fcc_cluster'."
        )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type=network_type,
        circuit_type=circuit_type,
        generation_parameters={
            "construction": construction,
            "periods": normalized_periods,
            "coords_offset": tuple(float(value) for value in coords_offset),
            "distance_tolerance": float(distance_tolerance),
        },
        require_isomorphic=True,
    )


# ============================================================================
# Laves / gyroid-skeleton lattices
# ============================================================================


def _normalize_chirality(chirality: Chirality) -> Literal["right", "left"]:
    """Normalize chirality labels accepted by the public Laves builder."""
    normalized = str(chirality).strip().lower()

    if normalized in {"r", "right"}:
        return "right"

    if normalized in {"l", "left"}:
        return "left"

    raise ValueError("chirality must be 'right', 'left', 'R', or 'L'.")


def build_laves_skeleton(
    periods: PeriodSpecification = 2,
    chirality: Chirality = "right",
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    cycle_core: bool = True,
    distance_tolerance: float = 1e-10,
) -> nx.Graph:
    """
    Build a finite triply repeated Laves-network skeleton.

    Parameters
    ----------
    periods
        Number of repeated unit cells in x, y, and z.
    chirality
        ``"right"``/``"R"`` uses the fundamental motif directly.
        ``"left"``/``"L"`` reflects the motif across the yz-plane before
        translation.
    offset
        Translation applied after chirality and periodic repetition.
    cycle_core
        If ``True``, remove degree-0 and degree-1 boundary material after the
        finite repeated block has been constructed.
    distance_tolerance
        Absolute tolerance for nearest-neighbour distance tests.

    Notes
    -----
    This is a finite block made by repetition in all three coordinate
    directions. It does not add toroidal wraparound edges between opposite
    boundaries.

    The unit-cell implementation is intentionally self-contained and can be
    revised later without changing the public pair-construction functions.
    """
    periods = normalize_periods(periods)
    chirality = _normalize_chirality(chirality)
    offset = _as_vector3(offset, name="offset")

    if chirality == "right":
        motif = LAVES_FUNDAMENTAL_POINTS.copy()
    else:
        motif = LAVES_FUNDAMENTAL_POINTS * np.asarray([-1.0, 1.0, 1.0])

    points: list[np.ndarray] = []

    for period_x in range(periods[0]):
        for period_y in range(periods[1]):
            for period_z in range(periods[2]):
                translation = LAVES_TRANSLATION_LENGTH * np.asarray(
                    [period_x, period_y, period_z],
                    dtype=float,
                )

                points.extend(
                    point + translation + offset
                    for point in motif
                )

    graph = _graph_from_points_at_distance(
        points,
        target_distance=LAVES_NEAREST_NEIGHBOUR_DISTANCE,
        tolerance=distance_tolerance,
    )

    if cycle_core:
        graph = prune_to_cycle_core(graph)

    return assign_graph_labels(graph)


def build_isomorphic_laves_graphs(
    periods: PeriodSpecification = 2,
    coords_offset: Sequence[float] = (3.0, 2.0, 0.0),
    chirality: Chirality = "right",
    *,
    cycle_core: bool = True,
    distance_tolerance: float = 1e-10,
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build a Laves skeleton and an exact same-handed translated copy.

    Graph 2 is copied from graph 1 rather than independently regenerated. This
    guarantees identical topology, node IDs, and edge insertion order.
    """
    graph1 = build_laves_skeleton(
        periods=periods,
        chirality=chirality,
        cycle_core=cycle_core,
        distance_tolerance=distance_tolerance,
    )

    graph2 = translated_graph_copy(
        graph1,
        coords_offset=coords_offset,
    )

    return graph1, graph2


def create_isomorphic_laves(
    periods: PeriodSpecification = 2,
    coords_offset: Sequence[float] = (3.0, 2.0, 0.0),
    chirality: Chirality = "right",
    *,
    cycle_core: bool = True,
    distance_tolerance: float = 1e-10,
    circuit_type: str = "default",
) -> Any:
    """Create two same-handed, exactly isomorphic Laves skeletons."""
    graph1, graph2 = build_isomorphic_laves_graphs(
        periods=periods,
        coords_offset=coords_offset,
        chirality=chirality,
        cycle_core=cycle_core,
        distance_tolerance=distance_tolerance,
    )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type="isomorphic_laves",
        circuit_type=circuit_type,
        generation_parameters={
            "periods": normalize_periods(periods),
            "coords_offset": tuple(float(value) for value in coords_offset),
            "chirality": _normalize_chirality(chirality),
            "cycle_core": bool(cycle_core),
            "distance_tolerance": float(distance_tolerance),
        },
        require_isomorphic=True,
    )


def _default_double_gyroid_reflection_offset(
    periods: tuple[int, int, int],
) -> np.ndarray:
    """
    Return the provisional reflection/translation used for the second labyrinth.

    For one x-period this gives ``(3, 2, 0)``, matching the conventional
    reflected Laves motif placement. The x-shift is extended to keep a larger
    reflected block in the same approximate x-range as graph 1.
    """
    return np.asarray(
        [
            LAVES_TRANSLATION_LENGTH * periods[0] - 1.0,
            2.0,
            0.0,
        ],
        dtype=float,
    )


def build_double_gyroid_laves_graphs(
    periods: PeriodSpecification = 2,
    reflection_axis: Literal["x", "y", "z"] = "x",
    reflection_offset: Sequence[float] | None = None,
    *,
    cycle_core: bool = True,
    distance_tolerance: float = 1e-10,
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build opposite-handed Laves skeletons for the two gyroid labyrinths.

    Graph 1 is generated once. Graph 2 is then produced by reflecting the exact
    graph-1 topology and coordinates. The layers therefore remain isomorphic
    with one-to-one node and edge correspondence, while the reflection reverses
    handedness.

    ``reflection_offset`` is provisional and intentionally configurable. When
    omitted, an x-reflection uses the standard one-cell shift ``(3, 2, 0)`` and
    extends its x-component for larger repeated blocks.
    """
    periods = normalize_periods(periods)

    graph1 = build_laves_skeleton(
        periods=periods,
        chirality="right",
        cycle_core=cycle_core,
        distance_tolerance=distance_tolerance,
    )

    if reflection_offset is None:
        if reflection_axis == "x":
            reflection_offset_array = _default_double_gyroid_reflection_offset(
                periods
            )
        else:
            reflection_offset_array = np.zeros(3, dtype=float)
    else:
        reflection_offset_array = _as_vector3(
            reflection_offset,
            name="reflection_offset",
        )

    graph2 = reflected_graph_copy(
        graph1,
        axis=reflection_axis,
        offset=reflection_offset_array,
    )

    return graph1, graph2


def create_double_gyroid_laves(
    periods: PeriodSpecification = 2,
    reflection_axis: Literal["x", "y", "z"] = "x",
    reflection_offset: Sequence[float] | None = None,
    *,
    cycle_core: bool = True,
    distance_tolerance: float = 1e-10,
    circuit_type: str = "default",
) -> Any:
    """Create a Kirchhoff circuit representing opposite gyroid labyrinths."""
    normalized_periods = normalize_periods(periods)

    graph1, graph2 = build_double_gyroid_laves_graphs(
        periods=normalized_periods,
        reflection_axis=reflection_axis,
        reflection_offset=reflection_offset,
        cycle_core=cycle_core,
        distance_tolerance=distance_tolerance,
    )

    effective_offset = (
        _default_double_gyroid_reflection_offset(normalized_periods)
        if reflection_offset is None and reflection_axis == "x"
        else (
            np.zeros(3, dtype=float)
            if reflection_offset is None
            else _as_vector3(
                reflection_offset,
                name="reflection_offset",
            )
        )
    )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type="double_gyroid_laves",
        circuit_type=circuit_type,
        generation_parameters={
            "periods": normalized_periods,
            "reflection_axis": reflection_axis,
            "reflection_offset": tuple(float(value) for value in effective_offset),
            "cycle_core": bool(cycle_core),
            "distance_tolerance": float(distance_tolerance),
            "laves_geometry_status": "provisional",
        },
        require_isomorphic=True,
    )

# ============================================================================
# Random Voronoi networks
# ============================================================================


def _copy_graph_with_array_positions(
    graph: nx.Graph,
) -> nx.Graph:
    """
    Return a deep graph copy with NumPy-array node positions.

    A deep copy is used because some downstream geometric operations modify
    the position arrays in place. This prevents the copied embedding from
    sharing mutable coordinate arrays with the source graph.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a NetworkX graph.")

    copied_graph = copy.deepcopy(graph)

    for node, node_data in copied_graph.nodes(data=True):
        if "pos" not in node_data:
            raise KeyError(f"Node {node!r} has no 'pos' attribute.")

        position = np.asarray(
            node_data["pos"],
            dtype=float,
        )

        if position.ndim != 1:
            raise ValueError(
                f"Node {node!r} position must be one-dimensional; "
                f"received shape {position.shape}."
            )

        if not np.all(np.isfinite(position)):
            raise ValueError(
                f"Node {node!r} position contains non-finite values."
            )

        copied_graph.nodes[node]["pos"] = position.copy()

    return copied_graph


def default_isomorphic_voronoi_embedding(
    graph: nx.Graph,
    rng: int | np.random.Generator | None = None,
    jitter: float = 0.0,
) -> nx.Graph:
    """
    Re-embed a graph without changing its topology or edge ordering.

    The coordinates are cyclically permuted around the geometric centre and
    optional Gaussian jitter is added independently along each coordinate
    direction. Node IDs and edge insertion order are retained, so the result
    is exactly isomorphic to the input graph with direct node correspondence.

    Parameters
    ----------
    graph
        Embedded NetworkX graph to copy and re-embed.
    rng
        Seed or NumPy random generator controlling the jitter.
    jitter
        Gaussian standard deviation as a fraction of the coordinate range along each
        axis. Set to zero to apply only the deterministic coordinate
        permutation.

    Returns
    -------
    networkx.Graph
        Topology-preserving graph copy with a different embedding.
    """
    jitter = float(jitter)

    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be a nonnegative finite number.")

    random_generator = np.random.default_rng(rng)
    embedded_copy = _copy_graph_with_array_positions(graph)

    nodes = list(embedded_copy.nodes())

    if not nodes:
        return embedded_copy

    positions = np.asarray(
        [embedded_copy.nodes[node]["pos"] for node in nodes],
        dtype=float,
    )

    if positions.ndim != 2:
        raise ValueError(
            "Voronoi node positions must form a two-dimensional coordinate "
            "array."
        )

    coordinate_dimension = positions.shape[1]

    if coordinate_dimension < 2:
        raise ValueError(
            "Voronoi positions must have at least two coordinates per node."
        )

    center = positions.mean(axis=0)

    # In 3D this maps (x, y, z) to (z, x, y), reproducing the notebook
    # transformation. For other dimensions it performs the analogous cyclic
    # permutation without changing coordinate dimensionality.
    coordinate_order = np.roll(
        np.arange(coordinate_dimension),
        1,
    )

    transformed_positions = (
        (positions - center)[:, coordinate_order]
        + center
    )

    if jitter > 0.0:
        coordinate_scale = np.ptp(
            positions,
            axis=0,
        )
        coordinate_scale[coordinate_scale == 0.0] = 1.0

        transformed_positions += random_generator.normal(
            scale=jitter * coordinate_scale,
            size=transformed_positions.shape,
        )

    for node, position in zip(
        nodes,
        transformed_positions,
    ):
        embedded_copy.nodes[node]["pos"] = np.asarray(
            position,
            dtype=float,
        )

    return embedded_copy


def _normalize_voronoi_periods(
    periods: Sequence[int],
) -> tuple[int, int]:
    """Validate the graph-specific period counts for a Voronoi pair."""
    normalized_periods = tuple(
        int(value)
        for value in periods
    )

    if len(normalized_periods) != 2:
        raise ValueError(
            "periods must contain exactly two positive integers: "
            "one for each graph."
        )

    if any(value < 1 for value in normalized_periods):
        raise ValueError("Both Voronoi period counts must be at least 1.")

    return normalized_periods


def build_voronoi_graphs(
    random_type: str,
    periods: Sequence[int],
    sidelength: float = 1.0,
    *,
    make_isomorphic: bool = False,
    jitter: float = 0.0,  # CHANGED: caller-controlled embedding jitter
    seed: int | None = None,
    second_embedding: Callable[..., nx.Graph] | None = None,
    embedding_kwargs: Mapping[str, Any] | None = None,
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build a pair of random Voronoi graphs.

    Parameters
    ----------
    random_type
        Random-network type accepted by Kirchhoff's
        ``init_graph_from_random`` function. The commonly used three-
        dimensional option is ``"voronoi_volume"``.
    periods
        Two positive integers ``(periods1, periods2)``. Each value is passed
        to the corresponding Kirchhoff graph generation call. When
        ``make_isomorphic=True``, only ``periods[0]`` is used for generation;
        graph 2 is obtained by re-embedding graph 1.
    sidelength
        Domain side length passed to Kirchhoff's random graph generator.
    make_isomorphic
        If ``False``, generate graph 1 and graph 2 independently. If ``True``,
        generate graph 1 once and construct graph 2 by applying
        ``second_embedding`` to an exact topology-preserving copy.
    seed
        Optional NumPy seed used around Kirchhoff's legacy global-RNG graph
        generation. The original NumPy random state is restored afterward.
    jitter
        Gaussian jitter standard deviation as a fraction of each coordinate
        range. Used only for the default isomorphic embedding. Zero applies
        only the coordinate permutation. This parameter takes precedence
        over a "jitter" entry in embedding_kwargs for the default embedding.
    second_embedding
        Optional function used to re-embed graph 1 when
        ``make_isomorphic=True``. It must return a NetworkX graph. When
        omitted, :func:`default_isomorphic_voronoi_embedding` is used.
    embedding_kwargs
        Optional keyword arguments passed to ``second_embedding``. For the
        default embedding, ``rng=seed + 1`` is inserted unless explicitly
        supplied.

    Returns
    -------
    graph1, graph2
        Embedded Voronoi graph pair.
    """
    normalized_periods = _normalize_voronoi_periods(periods)
    sidelength = float(sidelength)
    jitter = float(jitter)

    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be a nonnegative finite number.")

    if not isinstance(random_type, str) or not random_type:
        raise ValueError("random_type must be a nonempty string.")

    if not np.isfinite(sidelength) or sidelength <= 0.0:
        raise ValueError("sidelength must be a positive finite number.")

    if seed is not None:
        seed = int(seed)

    # Kirchhoff 0.2.x uses NumPy's global random state internally. Preserve the
    # caller's state so this constructor remains locally reproducible without
    # changing randomness elsewhere in the notebook.
    previous_random_state = None

    if seed is not None:
        previous_random_state = np.random.get_state()
        np.random.seed(seed)

    try:
        graph1 = kir.init_graph_from_random(
            random_type,
            normalized_periods[0],
            sidelength,
        )
        graph1 = _copy_graph_with_array_positions(graph1)

        if make_isomorphic:
            embedding_function = (
                default_isomorphic_voronoi_embedding
                if second_embedding is None
                else second_embedding
            )

            kwargs = dict(embedding_kwargs or {})

            if second_embedding is None:
                kwargs["jitter"] = jitter  # CHANGED: forward the input value
                kwargs.setdefault(
                    "rng",
                    None if seed is None else seed + 1,
                )

            graph2 = embedding_function(
                graph1,
                **kwargs,
            )

            if not isinstance(graph2, nx.Graph):
                raise TypeError(
                    "second_embedding must return a NetworkX graph."
                )

            graph2 = _copy_graph_with_array_positions(graph2)

            if not nx.is_isomorphic(graph1, graph2):
                raise ValueError(
                    "second_embedding changed the graph topology; the "
                    "returned graph is not isomorphic to graph 1."
                )
        else:
            graph2 = kir.init_graph_from_random(
                random_type,
                normalized_periods[1],
                sidelength,
            )
            graph2 = _copy_graph_with_array_positions(graph2)

    finally:
        if previous_random_state is not None:
            np.random.set_state(previous_random_state)

    return graph1, graph2


class NetworkxDualVoronoi:
    """
    Backward-compatible graph-pair container for Voronoi notebooks.

    New code can call :func:`build_voronoi_graphs` or :func:`create_voronoi`
    directly. This class is retained so older cells that instantiate
    ``NetworkxDualVoronoi`` continue to work.
    """

    def __init__(
        self,
        random_type: str,
        periods: Sequence[int],
        sidelength: float = 1.0,
        *,
        make_isomorphic: bool = False,
        jitter: float = 0.0,  # CHANGED
        seed: int | None = None,
        second_embedding: Callable[..., nx.Graph] | None = None,
        embedding_kwargs: Mapping[str, Any] | None = None,
    ):
        graph1, graph2 = build_voronoi_graphs(
            random_type=random_type,
            periods=periods,
            sidelength=sidelength,
            make_isomorphic=make_isomorphic,
            jitter=jitter,  # CHANGED
            seed=seed,
            second_embedding=second_embedding,
            embedding_kwargs=embedding_kwargs,
        )

        self.layer = [graph1, graph2]
        self.make_isomorphic = bool(make_isomorphic)
        self.jitter = float(jitter)  # CHANGED
        self.random_type = random_type
        self.periods = _normalize_voronoi_periods(periods)
        self.sidelength = float(sidelength)
        self.seed = seed


def create_voronoi(
    random_type: str,
    periods: Sequence[int],
    sidelength: float = 1.0,
    *,
    make_isomorphic: bool = False,
    jitter: float = 0.0,  # CHANGED: set jitter when calling create_voronoi
    seed: int | None = None,
    second_embedding: Callable[..., nx.Graph] | None = None,
    embedding_kwargs: Mapping[str, Any] | None = None,
    circuit_type: str = "default",
) -> Any:
    """Create a labeled Kirchhoff dual circuit from random Voronoi graphs.

    jitter controls the default isomorphic embedding; zero disables jitter.
    It takes precedence over embedding_kwargs["jitter"] for that embedding.
    For custom embeddings, supply their options through embedding_kwargs.
    """
    normalized_periods = _normalize_voronoi_periods(periods)

    graph1, graph2 = build_voronoi_graphs(
        random_type=random_type,
        periods=normalized_periods,
        sidelength=sidelength,
        make_isomorphic=make_isomorphic,
        jitter=jitter,  # CHANGED
        seed=seed,
        second_embedding=second_embedding,
        embedding_kwargs=embedding_kwargs,
    )

    embedding_name = None

    if make_isomorphic:
        embedding_function = (
            default_isomorphic_voronoi_embedding
            if second_embedding is None
            else second_embedding
        )
        embedding_name = getattr(
            embedding_function,
            "__name__",
            type(embedding_function).__name__,
        )

    return construct_dual_circuit(
        graph1,
        graph2,
        network_type="voronoi",
        circuit_type=circuit_type,
        generation_parameters={
            "random_type": random_type,
            "periods": normalized_periods,
            "sidelength": float(sidelength),
            "make_isomorphic": bool(make_isomorphic),
            "jitter": float(jitter),  # CHANGED: record the requested value
            "seed": seed,
            "second_embedding": embedding_name,
            "embedding_kwargs": dict(embedding_kwargs or {}),
        },
        require_isomorphic=bool(make_isomorphic),
    )


def init_dual_voronoi(
    random_type: str,
    periods: Sequence[int],
    sidelength: float = 1.0,
    *,
    make_isomorphic: bool = False,
    jitter: float = 0.0,  # CHANGED
    seed: int | None = None,
    second_embedding: Callable[..., nx.Graph] | None = None,
    embedding_kwargs: Mapping[str, Any] | None = None,
    circuit_type: str = "default",
) -> Any:
    """Backward-compatible alias for :func:`create_voronoi`."""
    return create_voronoi(
        random_type=random_type,
        periods=periods,
        sidelength=sidelength,
        make_isomorphic=make_isomorphic,
        jitter=jitter,  # CHANGED
        seed=seed,
        second_embedding=second_embedding,
        embedding_kwargs=embedding_kwargs,
        circuit_type=circuit_type,
    )


def create_label_voronoi(
    random_type: str,
    periods: Sequence[int],
    sidelength: float = 1.0,
    *,
    make_isomorphic: bool = False,
    jitter: float = 0.0,  # CHANGED
    seed: int | None = None,
    second_embedding: Callable[..., nx.Graph] | None = None,
    embedding_kwargs: Mapping[str, Any] | None = None,
    circuit_type: str = "default",
) -> Any:
    """Backward-compatible labeled Voronoi constructor."""
    return create_voronoi(
        random_type=random_type,
        periods=periods,
        sidelength=sidelength,
        make_isomorphic=make_isomorphic,
        jitter=jitter,  # CHANGED
        seed=seed,
        second_embedding=second_embedding,
        embedding_kwargs=embedding_kwargs,
        circuit_type=circuit_type,
    )

# ============================================================================
# Isomorphism and corresponding-cycle helpers
# ============================================================================


def find_graph_isomorphism(
    graph1: nx.Graph,
    graph2: nx.Graph,
) -> dict[Any, Any]:
    """Return one node mapping from ``graph1`` to ``graph2``."""
    matcher = iso.GraphMatcher(graph1, graph2)

    if not matcher.is_isomorphic():
        raise ValueError("graph1 and graph2 are not isomorphic.")

    return next(matcher.isomorphisms_iter())


def map_cycle_to_graph(
    cycle_graph1: nx.Graph,
    graph2: nx.Graph,
    node_mapping: Mapping[Any, Any],
) -> nx.Graph:
    """
    Map a cycle from graph 1 to graph 2 and copy graph-2 attributes.
    """
    mapped_cycle = nx.Graph()

    for u, v in cycle_graph1.edges():
        mapped_u = node_mapping[u]
        mapped_v = node_mapping[v]

        if not graph2.has_edge(mapped_u, mapped_v):
            raise ValueError(
                f"Mapped edge {(mapped_u, mapped_v)} is absent from graph2."
            )

        mapped_cycle.add_edge(mapped_u, mapped_v)

    for node in mapped_cycle.nodes():
        mapped_cycle.nodes[node].update(graph2.nodes[node])

    for u, v in mapped_cycle.edges():
        mapped_cycle.edges[u, v].update(graph2.edges[u, v])

    return mapped_cycle


def mapped_edge_in_graph2(
    edge_graph1: tuple[Any, Any],
    graph2: nx.Graph,
    node_mapping: Mapping[Any, Any],
) -> tuple[Any, Any]:
    """Map one graph-1 edge to the corresponding graph-2 edge."""
    u, v = edge_graph1
    mapped_u = node_mapping[u]
    mapped_v = node_mapping[v]

    if graph2.has_edge(mapped_u, mapped_v):
        return mapped_u, mapped_v

    if graph2.has_edge(mapped_v, mapped_u):
        return mapped_v, mapped_u

    raise ValueError(
        f"Mapped edge {(mapped_u, mapped_v)} is absent from graph2."
    )


def construct_corresponding_cycle_bases(
    graph1: nx.Graph,
    graph2: nx.Graph,
    basis_function: Callable[[nx.Graph], Sequence[nx.Graph]],
) -> tuple[list[nx.Graph], list[nx.Graph], dict[Any, Any]]:
    """
    Compute a cycle basis on graph 1 and map it to an isomorphic graph 2.

    Parameters
    ----------
    graph1, graph2
        Isomorphic embedded graphs.
    basis_function
        Function returning graph-form cycles for ``graph1``. For example,
        pass ``sampling.calc_cycle_basis`` or
        ``sampling.calc_cycle_minimum_basis``.
    """
    node_mapping = find_graph_isomorphism(graph1, graph2)
    basis1 = list(basis_function(graph1))
    basis2 = [
        map_cycle_to_graph(cycle, graph2, node_mapping)
        for cycle in basis1
    ]

    return basis1, basis2, node_mapping


# ============================================================================
# Notebook examples
# ============================================================================

# Import during development:
#
#     %load_ext autoreload
#     %autoreload 2
#     import ensnarlment_utils.network_generation as ng


# ----------------------------------------------------------------------------
# Network type 1: Hopf-linked polygon loops
# ----------------------------------------------------------------------------
#
#     D = ng.create_hopf_link(
#         n_edges=(8, 8),
#         radius1=1.0,
#         radius2=1.0,
#         center1=(0.0, 0.0, 0.0),
#         center2=(1.0, 0.0, 0.0),
#         rotation1=(0.0, 0.0, 0.0),
#         rotation2=(90.0, 0.0, 0.0),
#         circuit_type="default",
#     )


# ----------------------------------------------------------------------------
# Network type 2: Symmetric ladders
# ----------------------------------------------------------------------------
#
#     D = ng.create_symmetric_ladders(
#         num_loops=5,
#         edge_length=1.0,
#         tolerance=1e-4,
#         circuit_type="default",
#     )


# ----------------------------------------------------------------------------
# Network type 3: Dual prisms
# ----------------------------------------------------------------------------
#
#     D = ng.create_dual_prism(
#         size=(8, 8),
#         coords_offset=(1.0, 0.0, 0.0),
#         rotation=(90.0, 0.0, 0.0),
#         circuit_type="default",
#     )

# ----------------------------------------------------------------------------
# Network type 4: Finite simple-cubic lattices
# ----------------------------------------------------------------------------
#
# ``periods=(n, m)`` creates:
#
#     graph 1: n x n x n unit cubes
#     graph 2: m x m x m unit cubes
#
# ``coords_offset=(x, y, z)`` translates graph 1. Graph 2 remains at
# the origin. The lattices have open boundaries, without wraparound edges.
#
#     D = ng.create_cubic(
#         periods=(2, 2),
#         coords_offset=(0.5, 0.5, 0.5),
#         circuit_type="default",
#     )

# ----------------------------------------------------------------------------
# Network type 5: Rectangular grids
# ----------------------------------------------------------------------------
#
#     D = ng.create_grids(
#         num_loops=(4, 4),
#         xy_spacing=(1.0, 1.0),
#         height=(1.0, 1.0),
#         layer_offset=(0.5, 0.5, 0.1),
#         second_rotation=(90.0, 0.0, 0.0),
#         invert_second=True,
#         circuit_type="default",
#     )


# ----------------------------------------------------------------------------
# Network type 6: Same-handed isomorphic diamond lattices
# ----------------------------------------------------------------------------
#
#     D = ng.create_isomorphic_diamond(
#         periods=2,
#         coords_offset=(0.5, -0.75, 0.25),
#         distance_tolerance=1e-10,
#         circuit_type="default",
#     )


# ----------------------------------------------------------------------------
# Network type 7: Same-handed isomorphic Laves skeletons
# ----------------------------------------------------------------------------
#
#     D = ng.create_isomorphic_laves(
#         periods=2,
#         coords_offset=(3.0, 2.0, 0.0),
#         chirality="right",
#         cycle_core=True,
#         distance_tolerance=1e-10,
#         circuit_type="default",
#     )


# ----------------------------------------------------------------------------
# Network type 8: Opposite-handed double-gyroid Laves skeletons
# ----------------------------------------------------------------------------
#
#     D = ng.create_double_gyroid_laves(
#         periods=2,
#         reflection_axis="x",
#         reflection_offset=None,
#         cycle_core=True,
#         distance_tolerance=1e-10,
#         circuit_type="default",
#     )


# ----------------------------------------------------------------------------
# Network type 9: Random Voronoi pair
# ----------------------------------------------------------------------------
#
# Independent graph realizations:
#
#     D = ng.create_voronoi(
#         random_type="voronoi_volume",
#         periods=(3, 3),
#         sidelength=1.0,
#         make_isomorphic=False,
#         seed=None,
#         second_embedding=None,
#         embedding_kwargs=None,
#         circuit_type="default",
#     )
#
# Isomorphic topology with a different graph-2 embedding:
#
#     D = ng.create_voronoi(
#         random_type="voronoi_volume",
#         periods=(3, 3),
#         sidelength=1.0,
#         make_isomorphic=True,
#         seed=42,
#         second_embedding=None,
#         embedding_kwargs={"jitter": 0.02},
#         circuit_type="default",
#     )
#
# Existing notebooks may continue to use the compatibility name:
#
#     D = ng.create_label_voronoi(
#         random_type="voronoi_volume",
#         periods=(3, 3),
#         sidelength=1.0,
#         make_isomorphic=True,
#         seed=42,
#         second_embedding=None,
#         embedding_kwargs={"jitter": 0.02},
#         circuit_type="default",
#     )


# ----------------------------------------------------------------------------
# Stable graph and edge access for downstream calculations
# ----------------------------------------------------------------------------
#
#     G1, G2 = ng.extract_layer_graphs(D)
#     edge_set1, edge_set2 = ng.get_edge_sets(D)
#
#     # Equivalent direct access:
#     edge_set1 = D.edge_set1
#     edge_set2 = D.edge_set2
#
# Use these exact edge lists when constructing M1_dir and M2_dir and when
# mapping edge-cut indices back to graph edges.


# ----------------------------------------------------------------------------
# Plot the generated dual circuit
# ----------------------------------------------------------------------------
#
#     fig = D.plot_circuit("")
#     fig.show()

