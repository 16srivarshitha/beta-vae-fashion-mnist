"""
Training logic for VAE.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imageio

from src.models.loss import vae_loss
from src.visualization.plots import visualize_latent_space_2d


class VAETrainer:
    """Trainer class for VAE with support for beta-VAE and GIF generation."""
    
    def __init__(self, model, optimizer, device, output_dir):
        """
        Initialize trainer.
        
        Args:
            model: VAE model
            optimizer: Optimizer
            device: Device to use
            output_dir: Directory for outputs
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, train_loader, beta=1.0):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            beta: Beta value for beta-VAE
            
        Returns:
            avg_loss: Average total loss
            avg_recon: Average reconstruction loss
            avg_kl: Average KL divergence
        """
        self.model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for data, _ in tqdm(train_loader, desc="Training", leave=False):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            x_recon, mu, log_var = self.model(data)
            loss, recon, kl = vae_loss(x_recon, data, mu, log_var, beta)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
        
        n = len(train_loader.dataset)
        return train_loss / n, train_recon / n, train_kl / n
    
    def evaluate(self, test_loader, beta=1.0):
        """
        Evaluate on test set.
        
        Args:
            test_loader: Test data loader
            beta: Beta value for beta-VAE
            
        Returns:
            avg_loss: Average total loss
            avg_recon: Average reconstruction loss
            avg_kl: Average KL divergence
        """
        self.model.eval()
        test_loss = 0
        test_recon = 0
        test_kl = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                x_recon, mu, log_var = self.model(data)
                loss, recon, kl = vae_loss(x_recon, data, mu, log_var, beta)
                
                test_loss += loss.item()
                test_recon += recon.item()
                test_kl += kl.item()
        
        n = len(test_loader.dataset)
        return test_loss / n, test_recon / n, test_kl / n
    
    def extract_latent_space(self, data_loader, selected_classes, max_samples=5000):
        """
        Extract latent representations for selected classes.
        
        Args:
            data_loader: Data loader
            selected_classes: List of classes to extract
            max_samples: Maximum samples to extract
            
        Returns:
            latents: Latent vectors [N, latent_dim]
            labels: Class labels [N]
        """
        self.model.eval()
        latents = []
        labels_list = []
        count = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                # Filter for selected classes
                mask = torch.zeros(len(labels), dtype=torch.bool)
                for cls in selected_classes:
                    mask |= (labels == cls)
                
                if mask.sum() == 0:
                    continue
                
                data = data[mask].to(self.device)
                labels_filtered = labels[mask]
                
                data_flat = data.view(-1, 784)
                mu, _ = self.model.encode(data_flat)
                
                latents.append(mu.cpu().numpy())
                labels_list.append(labels_filtered.numpy())
                
                count += len(labels_filtered)
                if count >= max_samples:
                    break
        
        latents = np.vstack(latents)
        labels_arr = np.hstack(labels_list)
        
        return latents, labels_arr
    
    def train_with_gif(self, train_loader, test_loader, epochs, beta, 
                       username, selected_classes=[0, 1, 2], save_interval=1):
        """
        Train VAE and create GIF of latent space evolution.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            epochs: Number of epochs
            beta: Beta value
            username: Username for plots
            selected_classes: Classes to visualize
            save_interval: Save frame every N epochs
            
        Returns:
            history: Dictionary of training history
            gif_path: Path to saved GIF
        """
        history = {
            'train_loss': [], 'test_loss': [],
            'train_recon': [], 'test_recon': [],
            'train_kl': [], 'test_kl': []
        }
        
        frames = []
        
        print(f"\nTraining with β={beta} for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss, train_recon, train_kl = self.train_epoch(
                train_loader, beta
            )
            
            # Testing
            test_loss, test_recon, test_kl = self.evaluate(
                test_loader, beta
            )
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_recon'].append(train_recon)
            history['test_recon'].append(test_recon)
            history['train_kl'].append(train_kl)
            history['test_kl'].append(test_kl)
            
            # Save latent space visualization
            if epoch % save_interval == 0:
                latents, labels = self.extract_latent_space(
                    test_loader, selected_classes, max_samples=5000
                )
                
                fig = visualize_latent_space_2d(
                    latents, labels, beta, epoch, username
                )
                
                # Convert figure to image array for GIF
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)
                
                import matplotlib.pyplot as plt
                plt.close(fig)
                
                print(f"Epoch {epoch}/{epochs} - "
                      f"Loss: {test_loss:.4f}, "
                      f"Recon: {test_recon:.4f}, "
                      f"KL: {test_kl:.4f} - Frame saved")
            elif epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Loss: {test_loss:.4f}, "
                      f"Recon: {test_recon:.4f}, "
                      f"KL: {test_kl:.4f}")
        
        # Save GIF
        gif_path = self.output_dir / f'latent_evolution_beta_{beta}_{username}.gif'
        imageio.mimsave(gif_path, frames, fps=2, loop=0)
        print(f"\nSaved GIF with {len(frames)} frames: {gif_path}")
        
        return history, str(gif_path)
    
    def save_checkpoint(self, epoch, beta, history, filepath):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            beta: Beta value used
            history: Training history
            filepath: Path to save checkpoint
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'beta': beta,
            'history': history,
        }, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            checkpoint: Loaded checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint