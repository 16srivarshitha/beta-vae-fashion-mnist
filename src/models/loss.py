"""
Loss functions for VAE training.
"""

import torch
import torch.nn.functional as F


def vae_loss(x_recon, x, mu, log_var, beta=1.0):
    """
    Compute VAE loss = Reconstruction Loss + beta * KL Divergence.
    
    The reconstruction loss measures how well the decoder reconstructs the input,
    using Binary Cross-Entropy for pixel-wise comparison.
    
    The KL divergence regularizes the latent space to follow a standard normal
    distribution, enabling smooth interpolation and generation.
    
    Args:
        x_recon: Reconstructed images [batch_size, 784]
        x: Original images [batch_size, 1, 28, 28]
        mu: Mean of latent distribution [batch_size, latent_dim]
        log_var: Log variance of latent distribution [batch_size, latent_dim]
        beta: Weight for KL divergence (beta-VAE)
        
    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (Binary Cross-Entropy)
    x = x.view(-1, 784)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss