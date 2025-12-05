"""
Latent space interpolation and traversal visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import imageio


def interpolate_between_images(
    model,
    img1: torch.Tensor,
    img2: torch.Tensor,
    n_steps: int = 10,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Interpolate between two real images in latent space.
    
    Args:
        model: Trained VAE model
        img1: First image [1, 1, 28, 28]
        img2: Second image [1, 1, 28, 28]
        n_steps: Number of interpolation steps
        device: Device to use
        
    Returns:
        Interpolated images [n_steps, 28, 28]
    """
    model.eval()
    
    with torch.no_grad():
        # Encode both images
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        mu1, _ = model.encode(img1.view(-1, 784))
        mu2, _ = model.encode(img2.view(-1, 784))
        
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, n_steps).to(device)
        z_interp = torch.stack([
            (1 - alpha) * mu1 + alpha * mu2 
            for alpha in alphas
        ]).squeeze(1)
        
        # Decode interpolated latent vectors
        samples = model.decode(z_interp).cpu().view(-1, 28, 28)
    
    return samples


def latent_space_traversal(
    model,
    base_latent: Optional[torch.Tensor] = None,
    dim_idx: int = 0,
    n_steps: int = 11,
    range_scale: float = 3.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Traverse a single dimension in latent space while keeping others fixed.
    Useful for understanding what each latent dimension encodes.
    
    Args:
        model: Trained VAE model
        base_latent: Base latent vector [1, latent_dim]. If None, uses zero vector
        dim_idx: Which latent dimension to traverse
        n_steps: Number of steps in traversal
        range_scale: How far to traverse (in standard deviations)
        device: Device to use
        
    Returns:
        Generated images [n_steps, 28, 28]
    """
    model.eval()
    
    with torch.no_grad():
        # Initialize base latent vector
        if base_latent is None:
            base_latent = torch.zeros(1, model.latent_dim).to(device)
        else:
            base_latent = base_latent.to(device)
        
        # Create traversal values
        traversal_values = torch.linspace(
            -range_scale, range_scale, n_steps
        ).to(device)
        
        # Generate images by varying single dimension
        samples = []
        for value in traversal_values:
            z = base_latent.clone()
            z[0, dim_idx] = value
            sample = model.decode(z).cpu().view(28, 28)
            samples.append(sample)
        
        samples = torch.stack(samples)
    
    return samples


def create_interpolation_grid(
    model,
    test_loader,
    n_pairs: int = 4,
    n_steps: int = 10,
    device: str = 'cuda',
    username: str = 'user',
    output_dir: str = './outputs/figures'
) -> None:
    """
    Create a grid showing multiple interpolations.
    
    Args:
        model: Trained VAE model
        test_loader: Test data loader
        n_pairs: Number of image pairs to interpolate
        n_steps: Steps per interpolation
        device: Device to use
        username: Username for watermark
        output_dir: Output directory
    """
    model.eval()
    
    # Get random image pairs
    data_iter = iter(test_loader)
    images, _ = next(data_iter)
    
    fig, axes = plt.subplots(n_pairs, n_steps, figsize=(20, 4 * n_pairs))
    
    for i in range(n_pairs):
        img1 = images[2*i:2*i+1]
        img2 = images[2*i+1:2*i+2]
        
        # Interpolate
        interpolated = interpolate_between_images(
            model, img1, img2, n_steps, device
        )
        
        # Plot
        for j in range(n_steps):
            axes[i, j].imshow(interpolated[j].numpy(), cmap='gray')
            axes[i, j].axis('off')
            
            # Label endpoints
            if j == 0:
                axes[i, j].set_title('Start', fontsize=10)
            elif j == n_steps - 1:
                axes[i, j].set_title('End', fontsize=10)
    
    plt.suptitle('Latent Space Interpolations', fontsize=16, fontweight='bold')
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'interpolation_grid_{username}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved interpolation grid: {output_path}")


def create_traversal_visualization(
    model,
    n_dims: int = 10,
    n_steps: int = 11,
    device: str = 'cuda',
    username: str = 'user',
    output_dir: str = './outputs/figures'
) -> None:
    """
    Visualize traversals across multiple latent dimensions.
    
    Args:
        model: Trained VAE model
        n_dims: Number of dimensions to visualize
        n_steps: Steps per dimension
        device: Device
        username: Username for watermark
        output_dir: Output directory
    """
    model.eval()
    
    fig, axes = plt.subplots(n_dims, n_steps, figsize=(22, 2 * n_dims))
    
    for dim in range(n_dims):
        # Traverse this dimension
        samples = latent_space_traversal(
            model, dim_idx=dim, n_steps=n_steps, device=device
        )
        
        for step in range(n_steps):
            axes[dim, step].imshow(samples[step].numpy(), cmap='gray')
            axes[dim, step].axis('off')
            
            if step == 0:
                axes[dim, step].set_ylabel(
                    f'Dim {dim}', 
                    rotation=0, 
                    labelpad=30,
                    fontsize=10,
                    va='center'
                )
    
    plt.suptitle('Latent Dimension Traversals', fontsize=16, fontweight='bold')
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'latent_traversals_{username}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved latent traversals: {output_path}")


def create_interpolation_animation(
    model,
    img1: torch.Tensor,
    img2: torch.Tensor,
    n_steps: int = 30,
    device: str = 'cuda',
    output_path: str = './outputs/figures/interpolation_animation.gif',
    fps: int = 10
) -> None:
    """
    Create an animated GIF of latent space interpolation.
    
    Args:
        model: Trained VAE model
        img1: First image
        img2: Second image
        n_steps: Number of frames
        device: Device
        output_path: Output file path
        fps: Frames per second
    """
    # Generate interpolation
    samples = interpolate_between_images(model, img1, img2, n_steps, device)
    
    # Create frames
    frames = []
    for i in range(n_steps):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(samples[i].numpy(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Step {i+1}/{n_steps}', fontsize=14)
        
        # Convert to image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
    
    # Save GIF (forward and backward for smooth loop)
    frames_loop = frames + frames[::-1]
    imageio.mimsave(output_path, frames_loop, fps=fps)
    
    print(f"Saved interpolation animation: {output_path}")


def analyze_latent_dimensions(
    model,
    test_loader,
    n_samples: int = 1000,
    device: str = 'cuda'
) -> dict:
    """
    Analyze variance and activity of latent dimensions.
    
    Args:
        model: Trained VAE model
        test_loader: Test data loader
        n_samples: Number of samples to analyze
        device: Device
        
    Returns:
        Dictionary with analysis results
    """
    model.eval()
    
    latents = []
    count = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            data_flat = data.view(-1, 784)
            mu, log_var = model.encode(data_flat)
            latents.append(mu.cpu().numpy())
            
            count += len(data)
            if count >= n_samples:
                break
    
    latents = np.vstack(latents)[:n_samples]
    
    # Compute statistics per dimension
    dim_means = np.mean(latents, axis=0)
    dim_stds = np.std(latents, axis=0)
    dim_activity = np.mean(np.abs(latents), axis=0)
    
    # Compute correlations
    correlation_matrix = np.corrcoef(latents.T)
    
    results = {
        'dim_means': dim_means,
        'dim_stds': dim_stds,
        'dim_activity': dim_activity,
        'correlation_matrix': correlation_matrix,
        'latent_samples': latents
    }
    
    return results


def plot_latent_dimension_analysis(
    analysis: dict,
    username: str = 'user',
    output_dir: str = './outputs/figures'
) -> None:
    """
    Plot analysis of latent dimensions.
    
    Args:
        analysis: Results from analyze_latent_dimensions
        username: Username for watermark
        output_dir: Output directory
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Dimension variances
    ax1 = fig.add_subplot(gs[0, 0])
    dims = range(len(analysis['dim_stds']))
    ax1.bar(dims, analysis['dim_stds'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Latent Dimension', fontsize=12)
    ax1.set_ylabel('Standard Deviation', fontsize=12)
    ax1.set_title('Variance per Latent Dimension', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Dimension activity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(dims, analysis['dim_activity'], color='coral', alpha=0.7)
    ax2.set_xlabel('Latent Dimension', fontsize=12)
    ax2.set_ylabel('Mean Absolute Value', fontsize=12)
    ax2.set_title('Activity per Latent Dimension', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Correlation matrix
    ax3 = fig.add_subplot(gs[1, :])
    im = ax3.imshow(analysis['correlation_matrix'], cmap='coolwarm', 
                    vmin=-1, vmax=1, aspect='auto')
    ax3.set_xlabel('Latent Dimension', fontsize=12)
    ax3.set_ylabel('Latent Dimension', fontsize=12)
    ax3.set_title('Latent Dimension Correlations', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Correlation')
    
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    output_path = Path(output_dir) / f'latent_analysis_{username}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved latent dimension analysis: {output_path}")