# Advanced Sampling Benchmark Results

## Overview
We implemented **DDIM (Denoising Diffusion Implicit Models)** to accelerate the inference process. This allows us to skip time steps during sampling, drastically reducing the time required to generate 3D structures.

**Date:** Today
**Device:** CUDA (NVIDIA GeForce RTX 3060)
**Batch Size:** 16

## Performance Comparison

| Method | Steps | Time (s) | Speedup |
|--------|-------|----------|---------|
| **Standard (DDPM)** | 1000 | 21.41s | 1.0x (Baseline) |
| **DDIM** | 100 | 2.23s | **9.6x** |
| **DDIM** | 50 | 1.09s | **19.6x** |
| **DDIM** | 20 | 0.44s | **48.5x** |
| **DDIM** | 10 | 0.21s | **100.1x** |

## Conclusion
- **DDIM with 50 steps** provides a **~20x speedup** (1 second vs 21 seconds) and is likely the sweet spot for quality/speed.
- **DDIM with 10 steps** is extremely fast (0.2s) and can be used for real-time previews.
- The implementation correctly handles the spectral diffusion matrices ($\Phi_t, \Sigma_t$) and the static graph Laplacian $L$.

## Usage
To use the new sampler:

```python
from core.advanced_sampling import ddim_sample

# ... load model and precompute Phi, Sigma ...

final_sample, trajectory = ddim_sample(
    model=model,
    L=L,
    shape=(batch_size, num_nodes, 3),
    Phi_all=Phi_all,
    Sigma_all=Sigma_all,
    num_inference_steps=50,  # Adjust for speed/quality
    eta=0.0  # Deterministic sampling
)
```
