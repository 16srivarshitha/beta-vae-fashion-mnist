# β-VAE for Fashion-MNIST: Exploring the Reconstruction-Regularization Trade-off

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation and analysis of β-Variational Autoencoders (β-VAE) on the Fashion-MNIST dataset. This project systematically explores how the β parameter controls the trade-off between reconstruction quality and latent space organization.

## Key Features

- **Complete β-VAE Implementation**: Clean, modular PyTorch implementation with full training pipeline
- **Systematic β Analysis**: Comparison across β = {0.1, 0.5, 1.0} with quantitative metrics
- **Latent Space Evolution**: Animated GIFs showing how latent space structure improves during training
- **FID Score Evaluation**: Quantitative assessment of generation quality (54% improvement from β=0.1 to β=1.0)
- **Frozen Parameter Analysis**: Investigation of variance impact on generation diversity
- **Production-Ready Code**: Modular structure with config files, comprehensive logging, and checkpointing

## Key Results

| β Value | Reconstruction Loss | KL Divergence | FID Score | Best For |
|---------|-------------------|--------------|-----------|----------|
| 0.1 | **214.55** | 38.28 | 18.08 | Reconstruction tasks |
| 0.5 | 221.34 | 18.32 | 9.34 | Balanced applications |
| 1.0 | 227.67 | **12.33** | **8.31** | Generation & sampling |

**Key Finding**: Higher β values (1.0) achieve 54% better FID scores despite slightly higher reconstruction loss, demonstrating the importance of latent space regularization for generation quality.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/16srivarshitha/beta-vae-fashion-mnist.git
cd beta-vae-fashion-mnist

# Install dependencies
pip install -r requirements.txt

# Create output directories
mkdir -p outputs/{checkpoints,figures,logs}
```

### Training

```bash
# Train models with all beta values (0.1, 0.5, 1.0)
python scripts/train.py --config configs/config.yaml

# Training will save:
# - Model checkpoints to outputs/checkpoints/
# - Training curves to outputs/figures/
# - Latent space evolution GIFs to outputs/figures/
```

### Evaluation

```bash
# Evaluate trained models and generate visualizations
python scripts/evaluate.py --config configs/config.yaml

# Generates:
# - Reconstruction comparisons
# - Generated samples from N(0, I)
# - FID scores
# - Frozen parameter analysis
# - Comparison tables
```

## Project Structure

```
beta-vae-fashion-mnist/
├── configs/
│   └── config.yaml           # All hyperparameters
├── src/
│   ├── data/
│   │   └── dataset.py        # Data loading
│   ├── models/
│   │   ├── vae.py           # VAE architecture
│   │   └── loss.py          # Loss functions
│   ├── training/
│   │   └── trainer.py       # Training logic
│   ├── evaluation/
│   │   └── metrics.py       # FID, reconstruction error
│   └── visualization/
│       └── plots.py         # All visualization functions
├── scripts/
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
└── outputs/
    ├── checkpoints/         # Saved models
    ├── figures/             # Plots and GIFs
    └── logs/                # Training logs
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  latent_dim: 20              # Latent space dimension
  hidden_dim: 400             # Hidden layer size

training:
  epochs: 30
  learning_rate: 0.001
  beta_values: [0.1, 0.5, 1.0]  # Beta values to test

data:
  batch_size: 128
  selected_classes: [0, 1, 2]   # Classes for latent viz
```

## Visualizations

### Latent Space Evolution
The project generates animated GIFs showing how the latent space organization improves during training:

![Latent Space Evolution](outputs/figures/latent_evolution_beta_1.0.gif)

*Latent space evolution for β=1.0 showing progressive class separation*

### Training Curves
Comprehensive tracking of loss components:

![Training Curves](outputs/figures/training_curves_beta_1.0.png)

### Generated Samples
Samples from N(0, I) showing generation quality:

![Generated Samples](outputs/figures/generated_samples.png)

## Experiments

### 1. Beta-VAE Analysis
Systematic comparison of β = {0.1, 0.5, 1.0}:
- **β = 0.1**: Best reconstruction, poor generation (FID: 18.08)
- **β = 0.5**: Balanced trade-off (FID: 9.34)
- **β = 1.0**: Best generation quality (FID: 8.31)

### 2. Frozen Parameter Study
Analysis of generation with μ=0 and varying σ:
- **σ = 0.1**: Low diversity (std: 0.015)
- **σ = 0.5**: Moderate diversity (std: 0.065)
- **σ = 1.0**: High diversity matching prior (std: 0.121)

## Key Insights

1. **Reconstruction-Regularization Trade-off**: Higher β improves latent space structure at the cost of reconstruction fidelity
2. **FID as Quality Metric**: 54% improvement (18.08 → 8.31) demonstrates β=1.0's superior generation capability
3. **Latent Space Organization**: Higher β values create well-separated clusters enabling smooth interpolation
4. **Variance Control**: σ parameter directly controls generation diversity, with σ=1.0 optimal for varied outputs

## Technical Details

### Model Architecture
- **Encoder**: 784 → 400 → 400 → (μ, log σ²) ∈ ℝ²⁰
- **Decoder**: 20 → 400 → 400 → 784
- **Activation**: ReLU (hidden), Sigmoid (output)

### Loss Function
```
L = E[log p(x|z)] - β · D_KL(q(z|x) || p(z))
```
where β controls the regularization strength.

### Training Details
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 128
- **Epochs**: 30
- **Device**: CUDA (if available)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{vae-fashion-mnist-2025,
  author = {Srivarshitha M},
  title = {β-VAE for Fashion-MNIST: Exploring the Reconstruction-Regularization Trade-off},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/16srivarshitha/beta-vae-fashion-mnist}
}
```

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE.md) file for details.


## Acknowledgments

- Original VAE paper: [Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
- β-VAE paper: [Higgins et al., 2017](https://openreview.net/forum?id=Sy2fzU9gl)
- Fashion-MNIST dataset: [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)

