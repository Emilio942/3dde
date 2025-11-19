# Molecular Logic Activation

## Overview
We have successfully activated the logic for handling real molecular structures (or arbitrary point clouds) instead of synthetic grids. This addresses the need for "Surface Association" by constructing a graph based on spatial proximity.

**Date:** Today
**Status:** Active & Verified

## The "Compromise" Solution
To balance **Accuracy** (Physical Plausibility) and **Efficiency** (Speed/Generalization), we implemented a **Radius Graph** approach:

1.  **Representation:** Instead of a fixed grid, we treat the object as a cloud of points (atoms or surface vertices).
2.  **Graph Construction:** We connect points that are within a `cutoff` distance (e.g., 2.5 Å).
    *   *Why?* This creates a mesh-like structure that represents the surface topology without needing a full mesh.
    *   *Efficiency:* Sparse matrices (Laplacian) keep computation fast.
    *   *Accuracy:* The graph Laplacian captures the geometry better than a simple point cloud.

## How to Use
We created a demo script `demo_molecular.py` to verify this workflow.

```bash
python demo_molecular.py --atoms 100 --cutoff 3.0
```

### Integration in Config
To use this in training, update your `experiments/configs/default.yaml`:

```yaml
graph:
  type: "molecular"  # Changed from "grid"
  molecular_config:
    cutoff_radius: 2.5
    k_neighbors: 10  # Optional: Force k-NN connectivity
```

## Next Steps
- Load real `.xyz` or `.pdb` files using `src/data/dataset.py`.
- Train the model on a dataset of molecular surfaces to learn valid chemical geometries.
