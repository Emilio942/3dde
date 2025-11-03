"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from src.data.graph_builder import build_grid_laplacian, build_laplacian_from_edges


@pytest.fixture
def device():
    """Get available device (CPU/CUDA)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_laplacian():
    """Create a small 3x3 grid Laplacian for testing."""
    return build_grid_laplacian((3, 3))  # 9 nodes


@pytest.fixture
def medium_laplacian():
    """Create a medium 5x5 grid Laplacian for testing."""
    return build_grid_laplacian((5, 5))  # 25 nodes


@pytest.fixture
def chain_laplacian():
    """Create a simple chain graph: 0-1-2-3-4."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    return build_laplacian_from_edges(edges, num_nodes=5)


@pytest.fixture
def star_laplacian():
    """Create a star graph: center connected to 4 nodes."""
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    return build_laplacian_from_edges(edges, num_nodes=5)


@pytest.fixture
def sample_states(device):
    """Create sample states tensor (batch=4, nodes=9, dim=3)."""
    return torch.randn(4, 9, 3, device=device)


@pytest.fixture
def sample_batch():
    """Create a sample data batch."""
    batch_size = 4
    num_nodes = 9
    node_dim = 3
    
    return {
        'states': torch.randn(batch_size, num_nodes, node_dim),
        'original': torch.randn(batch_size, num_nodes, node_dim),
        'metadata': {'idx': torch.arange(batch_size)}
    }


@pytest.fixture(params=[10, 50, 100])
def num_steps(request):
    """Parametrize number of diffusion steps."""
    return request.param


@pytest.fixture(params=['linear', 'cosine'])
def beta_schedule(request):
    """Parametrize beta schedules."""
    return request.param


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def epsilon_tolerance():
    """Numerical tolerance for comparisons."""
    return 1e-5


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir  # Return Path object, not string
