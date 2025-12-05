"""
Visualization functions for VAE analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def visualize_latent_space_2d(latents, labels, beta_value, epoch, username):
    """
    Visualize 2D latent space with color-coded classes.
    
    Args:
        latents: Latent vectors [N, latent_dim]
        labels: Class labels
        beta_value: Beta value used
        epoch: Current epoch
        username: Email username
        
    Returns:
        Figure object
    """
    # Use first 2 dimensions
    if latents.shape[1] >= 2:
        z1, z2 = latents[:, 0], latents[:, 1]
    else:
        z1, z2 = latents[:, 0], np.zeros_like(latents[:, 0])
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(z1[mask], z2[mask], c=colors[i % len(colors)], 
                   label=f'Class {label}: {CLASS_NAMES[label]}', 
                   alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    
    plt.xlabel('Latent Dimension 1', fontsize=14, fontweight='bold')
    plt.ylabel('Latent Dimension 2', fontsize=14, fontweight='bold')
    plt.title(f'Latent Space (β={beta_value}, Epoch {epoch})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.text(0.02, 0.98, f'Epoch: {epoch}', 
             transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gca().transAxes, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    return plt.gcf()


def plot_training_curves(history, beta_value, username, output_dir):
    """
    Plot training curves for loss components.
    
    Args:
        history: Dictionary containing training history
        beta_value: Beta value used
        username: Email username
        output_dir: Output directory
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title(f'Total Loss (β={beta_value})', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, history['train_recon'], label='Train', linewidth=2)
    axes[1].plot(epochs, history['test_recon'], label='Test', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Reconstruction Loss', fontsize=12)
    axes[1].set_title(f'Reconstruction Loss (β={beta_value})', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[2].plot(epochs, history['train_kl'], label='Train', linewidth=2)
    axes[2].plot(epochs, history['test_kl'], label='Test', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('KL Divergence', fontsize=12)
    axes[2].set_title(f'KL Divergence (β={beta_value})', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.text(0.95, 0.95, username, ha='right', va='top',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    filename = Path(output_dir) / f'training_curves_beta_{beta_value}_{username}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def visualize_reconstructions(model, test_loader, device, n_samples, username, output_dir):
    """
    Display original and reconstructed images side-by-side.
    
    Args:
        model: Trained VAE model
        test_loader: Test data loader
        device: Device
        n_samples: Number of samples to display
        username: Email username
        output_dir: Output directory
    """
    model.eval()
    
    data, _ = next(iter(test_loader))
    data = data[:n_samples].to(device)
    
    with torch.no_grad():
        x_recon, _, _ = model(data)
    
    data = data.cpu().view(-1, 28, 28)
    x_recon = x_recon.cpu().view(-1, 28, 28)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 4))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(data[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12, fontweight='bold')
        
        # Reconstructed
        axes[1, i].imshow(x_recon[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12, fontweight='bold')
    
    plt.text(0.95, 0.95, username, ha='right', va='top',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    filename = Path(output_dir) / f'reconstructions_{username}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def generate_and_plot_samples(model, device, n_samples, username, output_dir):
    """
    Generate and plot new samples from N(0, I).
    
    Args:
        model: Trained VAE model
        device: Device
        n_samples: Number of samples
        username: Email username
        output_dir: Output directory
        
    Returns:
        Generated samples as numpy array
    """
    samples = model.generate(n_samples, device)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    
    plt.suptitle('Generated Samples from N(0, I)', fontsize=16, fontweight='bold', y=0.98)
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    filename = Path(output_dir) / f'generated_samples_{username}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    return samples.numpy()


def plot_frozen_params(model, device, sigma_values, n_samples, username, output_dir):
    """
    Plot samples with frozen mu=0 and varying sigma.
    
    Args:
        model: VAE model
        device: Device
        sigma_values: List of sigma values
        n_samples: Number of samples per sigma
        username: Email username
        output_dir: Output directory
        
    Returns:
        Dictionary of results
    """
    model.eval()
    
    fig, axes = plt.subplots(len(sigma_values), n_samples, 
                            figsize=(16, 3*len(sigma_values)))
    
    results = {}
    
    for row, sigma in enumerate(sigma_values):
        with torch.no_grad():
            z = torch.randn(n_samples, model.latent_dim).to(device) * sigma
            samples = model.decode(z).cpu().view(-1, 28, 28)
            results[sigma] = samples.numpy()
        
        for col in range(n_samples):
            if len(sigma_values) > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            
            ax.imshow(samples[col], cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(f'σ={sigma}', fontsize=14, fontweight='bold', 
                             rotation=0, labelpad=30, va='center')
    
    plt.suptitle('Generated Samples with Frozen μ=0 and Varying σ', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    filename = Path(output_dir) / f'frozen_params_{username}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    return results


def create_comparison_table(results, username, output_dir):
    """
    Create comparison table for different beta values.
    
    Args:
        results: Dictionary of results for each beta
        username: Email username
        output_dir: Output directory
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['β Value', 'Final\nRecon Loss', 'Final\nKL Div', 'Total\nLoss', 
               'Image\nQuality', 'Diversity', 'Cluster\nSeparation']
    
    quality_desc = {0.1: 'Sharp/Detailed', 0.5: 'Balanced', 1.0: 'Slightly Blurred'}
    diversity_desc = {0.1: 'Low', 0.5: 'Medium', 1.0: 'High'}
    cluster_desc = {0.1: 'Poor', 0.5: 'Moderate', 1.0: 'Good'}
    
    for beta in sorted(results.keys()):
        history = results[beta]['history']
        table_data.append([
            f'{beta:.1f}',
            f"{history['test_recon'][-1]:.2f}",
            f"{history['test_kl'][-1]:.2f}",
            f"{history['test_loss'][-1]:.2f}",
            quality_desc.get(beta, 'N/A'),
            diversity_desc.get(beta, 'N/A'),
            cluster_desc.get(beta, 'N/A')
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.12, 0.14, 0.14, 0.12, 0.16, 0.12, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('β-VAE Comparison: Effect of KL Weighting', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    filename = Path(output_dir) / f'beta_comparison_table_{username}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")