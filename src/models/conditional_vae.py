"""
Conditional Variational Autoencoder (CVAE) for class-specific generation.
Allows generating images conditioned on class labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for Fashion-MNIST.
    
    Conditions both encoder and decoder on class labels to enable
    controlled generation of specific fashion item classes.
    
    Architecture:
        - Encoder: [image + class] → latent space (μ, log σ²)
        - Decoder: [latent + class] → reconstructed image
    """
    
    def __init__(
        self,
        latent_dim: int = 20,
        hidden_dim: int = 400,
        num_classes: int = 10,
        class_embed_dim: int = 10
    ):
        """
        Initialize Conditional VAE.
        
        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Dimension of hidden layers
            num_classes: Number of classes in dataset (10 for Fashion-MNIST)
            class_embed_dim: Dimension of class embedding
        """
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        
        # Class embedding layer
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # Encoder: image (784) + class embedding (class_embed_dim)
        encoder_input_dim = 784 + class_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: latent (latent_dim) + class embedding (class_embed_dim)
        decoder_input_dim = latent_dim + class_embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )
    
    def encode(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input conditioned on class labels.
        
        Args:
            x: Input images [batch_size, 784]
            labels: Class labels [batch_size]
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            log_var: Log variance [batch_size, latent_dim]
        """
        # Get class embeddings
        class_embed = self.class_embedding(labels)  # [batch_size, class_embed_dim]
        
        # Concatenate image and class embedding
        x_conditional = torch.cat([x, class_embed], dim=1)
        
        # Encode
        h = self.encoder(x_conditional)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        
        return mu, log_var
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + ε * σ
        
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
    
    def decode(
        self,
        z: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent vector conditioned on class labels.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            labels: Class labels [batch_size]
            
        Returns:
            Reconstructed image [batch_size, 784]
        """
        # Get class embeddings
        class_embed = self.class_embedding(labels)
        
        # Concatenate latent and class embedding
        z_conditional = torch.cat([z, class_embed], dim=1)
        
        # Decode
        return self.decoder(z_conditional)
    
    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Conditional VAE.
        
        Args:
            x: Input images [batch_size, 1, 28, 28]
            labels: Class labels [batch_size]
            
        Returns:
            x_recon: Reconstructed images [batch_size, 784]
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        x = x.view(-1, 784)
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, labels)
        return x_recon, mu, log_var
    
    def generate(
        self,
        labels: torch.Tensor,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate samples conditioned on specific class labels.
        
        Args:
            labels: Class labels to generate [n_samples]
            device: Device to use ('cuda' or 'cpu')
            
        Returns:
            Generated samples [n_samples, 28, 28]
        """
        self.eval()
        with torch.no_grad():
            n_samples = len(labels)
            z = torch.randn(n_samples, self.latent_dim).to(device)
            labels = labels.to(device)
            samples = self.decode(z, labels).cpu().view(-1, 28, 28)
        return samples
    
    def generate_class_specific(
        self,
        class_label: int,
        n_samples: int = 16,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate multiple samples of a specific class.
        
        Args:
            class_label: Class to generate (0-9 for Fashion-MNIST)
            n_samples: Number of samples to generate
            device: Device to use
            
        Returns:
            Generated samples [n_samples, 28, 28]
        """
        labels = torch.full((n_samples,), class_label, dtype=torch.long)
        return self.generate(labels, device)
    
    def interpolate_within_class(
        self,
        class_label: int,
        n_steps: int = 10,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate interpolation between random samples of same class.
        
        Args:
            class_label: Class to interpolate within
            n_steps: Number of interpolation steps
            device: Device to use
            
        Returns:
            Interpolated samples [n_steps, 28, 28]
        """
        self.eval()
        with torch.no_grad():
            # Sample two random points in latent space
            z1 = torch.randn(1, self.latent_dim).to(device)
            z2 = torch.randn(1, self.latent_dim).to(device)
            
            # Linear interpolation
            alphas = torch.linspace(0, 1, n_steps).to(device)
            z_interp = torch.stack([
                (1 - alpha) * z1 + alpha * z2 
                for alpha in alphas
            ]).squeeze(1)
            
            # Generate with same class
            labels = torch.full(
                (n_steps,), class_label, 
                dtype=torch.long, device=device
            )
            samples = self.decode(z_interp, labels).cpu().view(-1, 28, 28)
        
        return samples


def cvae_loss(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Conditional VAE loss (same as standard VAE loss).
    
    Args:
        x_recon: Reconstructed images [batch_size, 784]
        x: Original images [batch_size, 1, 28, 28]
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        beta: Weight for KL divergence
        
    Returns:
        total_loss: Total CVAE loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss
    x = x.view(-1, 784)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class CVAETrainer:
    """
    Trainer for Conditional VAE with class-conditional generation support.
    """
    
    def __init__(
        self,
        model: ConditionalVAE,
        optimizer: torch.optim.Optimizer,
        device: str
    ):
        """
        Initialize CVAE trainer.
        
        Args:
            model: ConditionalVAE model
            optimizer: Optimizer
            device: Device to use
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train_epoch(
        self,
        train_loader,
        beta: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            beta: Beta value for beta-VAE
            
        Returns:
            avg_loss, avg_recon, avg_kl
        """
        self.model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for data, labels in train_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            x_recon, mu, log_var = self.model(data, labels)
            loss, recon, kl = cvae_loss(x_recon, data, mu, log_var, beta)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
        
        n = len(train_loader.dataset)
        return train_loss / n, train_recon / n, train_kl / n
    
    def evaluate(
        self,
        test_loader,
        beta: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Evaluate on test set.
        
        Args:
            test_loader: Test data loader
            beta: Beta value
            
        Returns:
            avg_loss, avg_recon, avg_kl
        """
        self.model.eval()
        test_loss = 0
        test_recon = 0
        test_kl = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                x_recon, mu, log_var = self.model(data, labels)
                loss, recon, kl = cvae_loss(x_recon, data, mu, log_var, beta)
                
                test_loss += loss.item()
                test_recon += recon.item()
                test_kl += kl.item()
        
        n = len(test_loader.dataset)
        return test_loss / n, test_recon / n, test_kl / n