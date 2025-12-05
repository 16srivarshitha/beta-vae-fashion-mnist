"""
latent space visualization.

visualization for analyzing VAE latent spaces
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

COLOR_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F38181', '#AA96DA', '#FCBAD3', '#FFFFD2', '#A8E6CF'
]


def extract_latent_representations(
    model,
    data_loader,
    device: str = 'cuda',
    max_samples: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract latent representations and labels from dataset.
    
    Args:
        model: Trained VAE model
        data_loader: Data loader
        device: Device to use
        max_samples: Maximum samples to extract
        
    Returns:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Class labels [n_samples]
    """
    model.eval()
    latents = []
    labels_list = []
    count = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            if count >= max_samples:
                break
            
            data = data.to(device)
            data_flat = data.view(-1, 784)
            mu, _ = model.encode(data_flat)
            
            latents.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
            
            count += len(data)
    
    latents = np.vstack(latents)[:max_samples]
    labels = np.hstack(labels_list)[:max_samples]
    
    return latents, labels


def visualize_latent_space_2d(
    latents: np.ndarray,
    labels: np.ndarray,
    title: str = "Latent Space Visualization",
    method: str = 'pca',
    username: str = 'user',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Visualize latent space in 2D using dimensionality reduction.
    
    Args:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Class labels [n_samples]
        title: Plot title
        method: Dimensionality reduction method ('pca', 'tsne', or 'first2')
        username: Username for watermark
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(latents)
        method_name = f"PCA (var: {reducer.explained_variance_ratio_.sum():.2%})"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = reducer.fit_transform(latents)
        method_name = "t-SNE"
    else:  # 'first2'
        coords = latents[:, :2]
        method_name = "First 2 Dimensions"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each class
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            label=f'{label}: {CLASS_NAMES[label]}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax.set_xlabel('Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Component 2', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\n({method_name})', fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add watermark
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def visualize_latent_space_3d_interactive(
    latents: np.ndarray,
    labels: np.ndarray,
    title: str = "Interactive 3D Latent Space",
    output_path: Optional[str] = None
) -> go.Figure:
    """
    Create interactive 3D visualization of latent space using Plotly.
    
    Args:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Class labels [n_samples]
        title: Plot title
        output_path: Path to save HTML file
        
    Returns:
        Plotly Figure object
    """
    # Use PCA to reduce to 3D
    if latents.shape[1] > 3:
        pca = PCA(n_components=3, random_state=42)
        coords = pca.fit_transform(latents)
        var_explained = pca.explained_variance_ratio_
    else:
        coords = latents[:, :3] if latents.shape[1] == 3 else np.c_[latents, np.zeros(len(latents))]
        var_explained = [1.0, 0.0, 0.0]
    
    # Create DataFrame for plotly
    import pandas as pd
    df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'z': coords[:, 2],
        'class': labels,
        'class_name': [CLASS_NAMES[l] for l in labels]
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='class_name',
        title=title,
        labels={
            'x': f'PC1 ({var_explained[0]:.1%})',
            'y': f'PC2 ({var_explained[1]:.1%})',
            'z': f'PC3 ({var_explained[2]:.1%})'
        },
        color_discrete_sequence=COLOR_PALETTE
    )
    
    fig.update_traces(marker=dict(size=4, opacity=0.7, line=dict(width=0.5, color='black')))
    fig.update_layout(
        font=dict(size=12),
        scene=dict(
            xaxis=dict(backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)"),
        ),
        height=800
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"Saved interactive plot: {output_path}")
    
    return fig


def plot_latent_space_comparison(
    latents_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    username: str = 'user',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare latent spaces from different models side-by-side.
    
    Args:
        latents_dict: Dictionary mapping model names to (latents, labels) tuples
        username: Username for watermark
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    n_models = len(latents_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, (latents, labels)) in enumerate(latents_dict.items()):
        ax = axes[idx]
        
        # Use first 2 dimensions
        coords = latents[:, :2]
        
        # Plot each class
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                label=CLASS_NAMES[label] if idx == 0 else "",
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel('Latent Dim 1', fontsize=12)
        ax.set_ylabel('Latent Dim 2', fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.suptitle('Latent Space Comparison Across Models', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_latent_density(
    latents: np.ndarray,
    labels: np.ndarray,
    username: str = 'user',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot density of latent space per class.
    
    Args:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Class labels [n_samples]
        username: Username for watermark
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    # Use PCA for 2D projection
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(latents)
    
    # Select 3 classes for clarity
    selected_classes = [0, 1, 2]  # T-shirt, Trouser, Pullover
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, class_label in enumerate(selected_classes):
        ax = axes[idx]
        mask = labels == class_label
        class_coords = coords[mask]
        
        # Create density plot
        ax.hexbin(
            class_coords[:, 0], class_coords[:, 1],
            gridsize=30, cmap='viridis', alpha=0.7
        )
        
        # Overlay scatter
        ax.scatter(
            class_coords[:, 0], class_coords[:, 1],
            c=COLOR_PALETTE[class_label],
            alpha=0.3,
            s=10,
            edgecolors='none'
        )
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title(f'{CLASS_NAMES[class_label]} Density', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latent Space Density Analysis', 
                 fontsize=16, fontweight='bold')
    
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_latent_statistics(
    latents: np.ndarray,
    labels: np.ndarray,
    username: str = 'user',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot statistical analysis of latent dimensions.
    
    Args:
        latents: Latent vectors [n_samples, latent_dim]
        labels: Class labels [n_samples]
        username: Username for watermark
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Mean and std per dimension
    ax1 = fig.add_subplot(gs[0, :])
    dims = range(latents.shape[1])
    means = np.mean(latents, axis=0)
    stds = np.std(latents, axis=0)
    
    ax1.errorbar(dims, means, yerr=stds, fmt='o-', capsize=5, 
                 linewidth=2, markersize=6, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero mean')
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Unit std')
    ax1.axhline(y=-1, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean ± Std', fontsize=12, fontweight='bold')
    ax1.set_title('Latent Dimension Statistics', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Variance per dimension
    ax2 = fig.add_subplot(gs[1, 0])
    variances = np.var(latents, axis=0)
    ax2.bar(dims, variances, color='coral', alpha=0.7)
    ax2.set_xlabel('Latent Dimension', fontsize=12)
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Dimension Variance', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Correlation heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    corr_matrix = np.corrcoef(latents.T)
    im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xlabel('Latent Dimension', fontsize=12)
    ax3.set_ylabel('Latent Dimension', fontsize=12)
    ax3.set_title('Dimension Correlations', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    # 4. Distribution per class
    ax4 = fig.add_subplot(gs[2, :])
    selected_dims = [0, 1]  # Show first 2 dimensions
    for dim in selected_dims:
        for class_label in [0, 1, 2]:
            mask = labels == class_label
            class_latents = latents[mask, dim]
            ax4.hist(class_latents, bins=50, alpha=0.4, 
                    label=f'{CLASS_NAMES[class_label]} (dim {dim})',
                    color=COLOR_PALETTE[class_label])
    
    ax4.set_xlabel('Latent Value', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Latent Distribution per Class (First 2 Dims)', 
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.text(0.95, 0.02, username, ha='right', va='bottom',
             transform=plt.gcf().transFigure, fontsize=10, 
             color='gray', alpha=0.7)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def create_latent_space_dashboard(
    model,
    data_loader,
    device: str = 'cuda',
    username: str = 'user',
    output_dir: str = './outputs/figures'
) -> None:
    """
    Create comprehensive latent space visualization dashboard.
    
    Args:
        model: Trained VAE model
        data_loader: Data loader
        device: Device to use
        username: Username for watermark
        output_dir: Output directory
    """
    print("Creating latent space dashboard...")
    
    # Extract latent representations
    print("Extracting latent representations...")
    latents, labels = extract_latent_representations(
        model, data_loader, device, max_samples=5000
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualizations
    print("\n1. Creating 2D visualization (PCA)...")
    visualize_latent_space_2d(
        latents, labels, 
        title="Latent Space (PCA)",
        method='pca',
        username=username,
        output_path=output_dir / f'latent_space_pca_{username}.png'
    )
    plt.close()
    
    print("2. Creating 2D visualization (t-SNE)...")
    visualize_latent_space_2d(
        latents, labels,
        title="Latent Space (t-SNE)",
        method='tsne',
        username=username,
        output_path=output_dir / f'latent_space_tsne_{username}.png'
    )
    plt.close()
    
    print("3. Creating interactive 3D visualization...")
    visualize_latent_space_3d_interactive(
        latents, labels,
        title="Interactive 3D Latent Space",
        output_path=output_dir / f'latent_space_3d_{username}.html'
    )
    
    print("4. Creating density analysis...")
    plot_latent_density(
        latents, labels,
        username=username,
        output_path=output_dir / f'latent_density_{username}.png'
    )
    plt.close()
    
    print("5. Creating statistical analysis...")
    plot_latent_statistics(
        latents, labels,
        username=username,
        output_path=output_dir / f'latent_statistics_{username}.png'
    )
    plt.close()
    
    print(f"\n Dashboard complete! All visualizations saved to: {output_dir}/")