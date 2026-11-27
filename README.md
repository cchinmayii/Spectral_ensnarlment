# Ensnarlment utilities

`ensnarlment_utils` is the Python module accompanying *STEM: A Spectral Theory of Ensnarlment Measures*. It constructs the ensnarlment operator for pairs of spatially embedded networks, analyzes its singular spectrum, visualizes edge participation in singular subspaces, and identifies edge cuts that reduce the rank of the remaining linkage.

## Setup

Use Python 3.11 and run these commands from the directory containing the `ensnarlment_utils/` folder. If you downloaded a ZIP, extract it first.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install \
  "snarlpy==0.0.1" "kirchhoff==0.2.7" "networkx==2.8.4" \
  "numpy==1.23.5" "scipy==1.10.1" "pandas==1.5.3" \
  "matplotlib==3.7.1" "seaborn==0.13.2" "plotly==5.9.0"
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it in PowerShell with `.\.venv\Scripts\Activate.ps1`. Run the `pip install` command on a single line in PowerShell. The dependency versions were tested with the examples below and are based on the [SnarlPy requirements](https://pypi.org/project/snarlpy/0.0.1/).

## Quick start

Construct a Hopf link between polygonal loops with five and seven edges. The operator has shape `(5, 7)` and rank one. Its only nonzero singular value is $1/\sqrt{35} \approx 0.169031$.

```python
from ensnarlment_utils.compute_ensnarlment import compute_ensnarlment
from ensnarlment_utils.network_generation import create_hopf_link
from ensnarlment_utils.spectral_analysis import compute_svd

circuit = create_hopf_link(n_edges=(5, 7))
result = compute_ensnarlment(circuit, cycle_basis_type="minimum", verbose=False)
operator = result["ensnarlment_operator"]
spectrum = compute_svd(operator)

print(operator.shape)                       # (5, 7)
print(spectrum["rank"])                     # 1
print(spectrum["nonzero_singular_values"])  # [0.16903085]
```

The construction result includes the two cycle bases, the directed edge-by-cycle matrices `M1_dir` and `M2_dir`, the rounded cycle-by-cycle `linkage_matrix`, and the ordered lists `edge_set1` and `edge_set2`. Those lists identify the rows and columns of the operator, respectively. For custom inputs, pass a pair of undirected NetworkX graphs with three-dimensional node positions stored in the `pos` attribute. Use disjoint embeddings and choose either `cycle_basis_type="minimum"` or `"fundamental"`.

### Visualize the principal singular subspace

```python
from ensnarlment_utils.network_visualization import plot_degenerate_singular_subspace

fig = plot_degenerate_singular_subspace(circuit, operator, verbose=False)
fig.write_html("hopf_link.html")
```

The plot colors each edge by the sum of squared singular-vector components in the principal subspace. When plotting, use the circuit’s native edge order. If you supplied custom edge orders or used `use_isomorphism=True`, align the operator or mesh matrices to the circuit’s edge order first.

To export PDF and HTML together with `export_network_plot`, install `kaleido==0.2.1` with the Plotly version above. [Kaleido 0.2.1](https://pypi.org/project/kaleido/0.2.1/)

### Cut edges to reduce linkage rank

```python
from ensnarlment_utils.edge_cutting import cut_with_recomputed_svd

cuts = cut_with_recomputed_svd(
    M1_dir=result["M1_dir"],
    M2_dir=result["M2_dir"],
    lk_mat=result["linkage_matrix"],
    edge_set1=result["edge_set1"],
    verbose=False,
)

cut_edges = [result["edge_set1"][i] for i in cuts["cut_edges1"]]
print(cut_edges)
print(cuts["final_rank"])  # 0 for this example
```

This routine selects edges in the first network and recomputes the residual SVD after a cut. The returned indices are zero-based and refer to the matrix edge order. The module also provides routines for selecting cuts from the original singular subspaces, cutting either layer, and validating cut sets by recomputing linkage after edge removal.

## Modules

| Module | Purpose |
| --- | --- |
| `network_generation.py` | Build embedded network pairs, including Hopf-linked loops, ladders, prisms, grids, cubic and diamond lattices, Laves networks, and Voronoi networks. |
| `compute_ensnarlment.py` | Construct cycle bases, directed edge-by-cycle matrices, and the Gauss-linking matrix, then assemble the ensnarlment operator. |
| `spectral_analysis.py` | Compute SVDs, extract singular subspaces, and analyze spectra, singular-value spacings, and rank across parameter sweeps. |
| `edge_cutting.py` | Score edges, search for rank-reducing cuts, and validate selected cut sets. |
| `network_visualization.py` | Plot embedded networks, singular-vector weights, singular subspaces, and selected cycles. |
| `matrix_tools.py` | Reorder, inspect, and plot mesh and linkage matrices. |

## Citation

If you use this code, cite *STEM: A Spectral Theory of Ensnarlment Measures* and the version of the software release used in your analysis. The module uses [SnarlPy](https://pypi.org/project/snarlpy/) for cycle and linkage calculations and [Kirchhoff](https://pypi.org/project/kirchhoff/) for circuit construction and network visualization. For the original framework, see Kramer and Modes, [*Ensnarled: Topological linkage of spatially embedded network pairs*](https://doi.org/10.1103/PhysRevResearch.5.033203).
