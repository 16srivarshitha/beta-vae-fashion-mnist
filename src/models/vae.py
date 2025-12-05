"""
Variational Autoencoder (VAE) model architecture.
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Variational Autoencoder for Fashion-MNIST.
    
    Architecture:
        - Encoder: Maps 28x28 images to latent space (mean and log_var)
        - Decoder: Reconstructs images from latent vectors
    """
    
    def __init__(self, latent_dim=20, hidden_dim=400):
        """
        Initialize VAE.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
        """
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def encode(self, x):
        """
        Encode input to latent space parameters.
        
        Args:
            x: Input images [batch_size, 784]
            
        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + eps * sigma
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed image
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x: Input images [batch_size, 1, 28, 28]
            
        Returns:
            x_recon: Reconstructed images
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        x = x.view(-1, 784)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def generate(self, n_samples=16, device='cuda'):
        """
        Generate new samples from standard normal distribution.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to use ('cuda' or 'cpu')
            
        Returns:
            Generated samples
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            samples = self.decode(z).cpu().view(-1, 28, 28)
        return samples