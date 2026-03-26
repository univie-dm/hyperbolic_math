# Hyperbolic Math

A PyTorch-based library for hyperbolic geometry and deep learning in hyperbolic spaces. This library provides implementations of various hyperbolic manifolds, neural network layers, visualization techniques and Riemannian optimizers for working with hyperbolic embeddings.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)


## Features

### Manifolds
- **Poincaré Ball**: Poincaré model of hyperbolic space with constraint $x_1^2 + x_1^2 + \ldots + x_d^2 < 1/c$
- **Hyperboloid**: Lorentz model of hyperbolic space with constraint $-x_0^2 + x_1^2 + \ldots + x_d^2 = -1/c,  x_0 > 0$
- **Euclidean**: Standard Euclidean space for comparison and baseline experiments
- **Trainable curvature**: Support for learnable curvature parameters via `trainable_c` flag
- **Precision control**: Fully configurable manifold precision (`float32` or `float64`) for optimal performance-accuracy trade-offs
- **Comprehensive operations**: Exponential/logarithmic/retraction maps, parallel transport, geodesic distances, scalar multiplication, manifold conversion maps, etc.

### Neural Network Layers
- **Standard hyperbolic layers**: Expmap, Logmap, Retraction, Activation modules for flexible network architectures
- **Model-specific layers**: Specialized linear and regression layers for both Poincaré and Hyperboloid models
- **PyTorch integration**: Seamless integration with PyTorch's `nn.Module` for easy model building

### Riemannian Optimizers
- **Riemannian SGD**: Stochastic gradient descent with momentum and Nesterov acceleration on manifolds
- **Riemannian Adam**: Adaptive moment estimation with AMSGrad variant for Riemannian optimization
- **Flexible updates**: Support for both exponential map and retraction-based parameter updates
- **PyTorch-compatible API**: Drop-in replacement for standard PyTorch optimizers

### Numerical Stability & Testing
- **Fine-tuned precision handling**: Carefully calibrated numerical thresholds for stable computations
- **Comprehensive test suite**: Unit tests covering geometric properties, numerical stability, and edge cases
- **Debugging support**: Extensive assertions and validation to catch errors early

### Utilities
- **Math utilities**: Stable implementations of hyperbolic functions (acosh, asinh, atanh, etc.)
- **Horospherical PCA++**: Stable dimensionality reduction for hyperbolic data
- **Visualization tools**: Functions for plotting and analyzing hyperbolic embeddings


## Installation

### From PyPi

```bash
pip install hyperbolic-math
```

### From Source

```bash
git clone https://github.com/univie-dm/hyperbolic-math.git
cd hyperbolic-math
pip install -e .
```

Alternatively for development and testing:
```bash
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.8, < 3.11
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- Pytest (for testing)


## Quick Start

### Basic Manifold Operations

```python
import torch
from src.manifolds import PoincareBall, Hyperboloid

# Initialize a Poincaré ball manifold (with curvature c=1 with operations performed in float64)
poincare = PoincareBall(c=torch.tensor([1.0]), dtype="float64")

# Create random points in the tangent space at the manifold's origin
u = torch.randn(10, 5, dtype=torch.float64) * 0.1
v = torch.randn(10, 5, dtype=torch.float64) * 0.1

# Map the 5-dimensional points to the manifold using the exponential map
x = poincare.expmap_0(u, axis=-1)
y = poincare.expmap_0(v, axis=-1)

# Compute the distances between the points
distances = poincare.dist(x, y, axis=-1)
print(f"Distances: {distances}")


# Hyperboloid manifold
hyperboloid = Hyperboloid(c=torch.tensor([1.0]), dtype="float64")

# Create random points in the tangent space at the manifold's origin
u_hyp = torch.randn(10, 5, dtype=torch.float64) * 0.1
u_hyp = torch.cat([torch.zeros_like(u_hyp[..., :1]), u_hyp], dim=-1)

# Map to the manifold using the exponential map
x_hyp = hyperboloid.expmap_0(u_hyp, axis=-1)
print(f"Points are in the manifold: {hyperboloid.is_in_manifold(x_hyp)}")

# Compute the distances of the points from the manifold's origin
distances_hyp = hyperboloid.dist_0(x_hyp, axis=-1)
print(f"Distances from the origin: {distances_hyp}")
```

### Building Hyperbolic Neural Networks

```python
import torch
import torch.nn as nn
from src.manifolds import PoincareBall
from src.nn_layers import Expmap_0, Logmap_0, HyperbolicLinearPoincarePP

# Define a simple hyperbolic neural network
class HyperbolicNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, manifold):
        super().__init__()
        self.manifold = manifold

        # Map to manifold
        self.expmap_0 = Expmap_0(manifold)

        # Hyperbolic linear layer
        self.hyp_linear = HyperbolicLinearPoincarePP(
            manifold, input_dim, hidden_dim, hyperbolic_axis=-1
        )

        # Map back to tangent space for output
        self.logmap_0 = Logmap_0(manifold)

        # Euclidean output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is in tangent space
        x = self.expmap_0(x)  # Map to manifold
        x = self.hyp_linear(x)  # Hyperbolic linear transformation
        x = self.logmap_0(x)  # Map back to tangent space
        x = self.output(x)  # Euclidean output
        return x

# Initialize the network
poincare = PoincareBall(c=torch.tensor([1.0]))
model = HyperbolicNet(input_dim=10, hidden_dim=20, output_dim=5, manifold=poincare)

# Example tangent vector input
u = torch.randn(32, 10)
output = model(u)
print(f"Output shape: {output.shape}")
```

### Using Riemannian Optimizers

```python
import torch
from src.manifolds import PoincareBall, ManifoldParameter
from src.optim import RiemannianAdam

# Create hyperbolic parameters
poincare = PoincareBall(c=torch.tensor([1.0]))
hyperbolic_weights = ManifoldParameter(
    data=torch.randn(100, 50),
    requires_grad=True,
    manifold=poincare
)

# Initialize the Riemannian Adam optimizer with exponential map updates
optimizer = RiemannianAdam(
    [hyperbolic_weights],
    lr=1e-3,
    expmap_update=True,
    hyperbolic_axis=-1
)

# Training loop example
for epoch in range(10):
    optimizer.zero_grad()

    # Your loss computation here
    # loss = compute_loss(hyperbolic_weights)

    # loss.backward()
    # optimizer.step()
```


## Testing

The library includes comprehensive unit tests for all manifold operations and optimizers.

### Running Tests

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_manifolds.py

# Run a specific test
pytest tests/test_manifolds.py::test_scalar_mul
```

### Test Coverage

Tests include:
- Geometric properties verification (triangle inequality, identity elements, etc.)
- Numerical stability tests
- Manifold-specific operations
- Optimizer convergence tests


## Citation
If you use this library in your research, please cite:
```bibtex
@software{hyperbolic_math_2026,
  title = {Hyperbolic Math: A PyTorch Library for Hyperbolic Deep Learning},
  author = {Lang, Thomas and Sidak, Kevin},
  version = {0.1.7},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19236209},
  url = {https://doi.org/10.5281/zenodo.19236209}
}
```

### Related Publications

This library builds upon research in hyperbolic geometry and deep learning, including:

- Aaron Lou, et al. "Differentiating through the fréchet mean."
    *International conference on machine learning (2020)*.
- Abraham Ungar. "A gyrovector space approach to hyperbolic geometry."
    *Springer Nature (2022)*.
- Ahmad Bdeir, Kristian Schwethelm, and Niels Landwehr. "Fully hyperbolic convolutional neural networks for computer vision."
    *arXiv preprint arXiv:2303.15919 (2023)*.
- Edoardo Cetin, et al. "Hyperbolic deep reinforcement learning."
    *arXiv (2022)*.
- Ganea Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic neural networks."
    *Advances in neural information processing systems 31 (2018)*.
- Gary Bécigneul and Octavian Ganea. "Riemannian adaptive optimization methods."
    *International Conference on Learning Representations (2019)*.
- Ines Chami, et al. "Horopca: Hyperbolic dimensionality reduction via horospherical projections."
    *International Conference on Machine Learning (2021)*.
- Ines Chami, et al. "Hyperbolic graph convolutional neural networks."
    *Advances in neural information processing systems 32 (2019)*.
- Max Kochurov, Rasul Karimov and Serge Kozlukov. "Geoopt: Riemannian Optimization in PyTorch."
    *arXiv (2020)*.
- Maximillian Nickel and Douwe Kiela. "Learning continuous hierarchies in the lorentz model of hyperbolic geometry."
    *International conference on machine learning. PMLR (2018)*.
- Maximillian Nickel and Douwe Kiela. "Poincaré embeddings for learning hierarchical representations."
    *Advances in neural information processing systems 30 (2017)*.
- Marc T. Law, et al. "Lorentzian distance learning for hyperbolic representations."
    *International Conference on Machine Learning (2019)*.
- Sashank J. Reddi, Satyen Kale, and Sanjiv Kumar. "On the convergence of adam and beyond."
    *arXiv preprint arXiv:1904.09237 (2019)*.
- Shimizu Ryohei, Yusuke Mukuta, and Tatsuya Harada. "Hyperbolic neural networks++."
    *arXiv preprint arXiv:2006.08210 (2020)*.
- Silvere Bonnabel. "Stochastic gradient descent on Riemannian manifolds."
    *IEEE Transactions on Automatic Control 58.9 (2013): 2217-2229*.
- Weize Chen, et al. "Fully hyperbolic neural networks."
    *arXiv preprint arXiv:2105.14686 (2021)*.


## License
MIT License

Copyright (c) 2025 Data Mining and Machine Learning - University of Vienna

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
