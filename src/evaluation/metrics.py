"""
Evaluation metrics for VAE.
"""

import torch
import numpy as np
from scipy.linalg import sqrtm


def calculate_fid_score(real_images, generated_images):
    """
    Calculate simplified Fréchet Inception Distance (FID).
    
    Note: This is a simplified version using pixel statistics.
    For full FID, use Inception network features.
    
    Args:
        real_images: Real images [N, 28, 28]
        generated_images: Generated images [N, 28, 28]
        
    Returns:
        fid: FID score
    """
    real_flat = real_images.reshape(len(real_images), -1)
    gen_flat = generated_images.reshape(len(generated_images), -1)
    
    mu_real = np.mean(real_flat, axis=0)
    mu_gen = np.mean(gen_flat, axis=0)
    
    sigma_real = np.cov(real_flat, rowvar=False)
    sigma_gen = np.cov(gen_flat, rowvar=False)
    
    diff = mu_real - mu_gen
    
    # Add small epsilon to avoid numerical issues
    epsilon = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * epsilon
    sigma_gen += np.eye(sigma_gen.shape[0]) * epsilon
    
    covmean = sqrtm(sigma_real @ sigma_gen)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return fid


def calculate_reconstruction_error(model, data_loader, device, n_samples=1000):
    """
    Calculate average reconstruction error on test set.
    
    Args:
        model: VAE model
        data_loader: Data loader
        device: Device to use
        n_samples: Number of samples to evaluate
        
    Returns:
        mse: Mean squared error
        mae: Mean absolute error
    """
    model.eval()
    mse_total = 0
    mae_total = 0
    count = 0
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            x_recon, _, _ = model(data)
            
            data_flat = data.view(-1, 784)
            
            mse = torch.mean((x_recon - data_flat) ** 2, dim=1).sum().item()
            mae = torch.mean(torch.abs(x_recon - data_flat), dim=1).sum().item()
            
            mse_total += mse
            mae_total += mae
            count += len(data)
            
            if count >= n_samples:
                break
    
    return mse_total / count, mae_total / count


def analyze_frozen_params(model, sigma_values, n_samples, device):
    """
    Analyze generation with frozen mean (mu=0) and varying sigma.
    
    Args:
        model: VAE model
        sigma_values: List of sigma values to test
        n_samples: Number of samples per sigma
        device: Device to use
        
    Returns:
        results: Dictionary with statistics for each sigma
    """
    model.eval()
    results = {}
    
    for sigma in sigma_values:
        with torch.no_grad():
            # Frozen mu = 0, varying sigma
            z = torch.randn(n_samples, model.latent_dim).to(device) * sigma
            samples = model.decode(z).cpu().numpy().reshape(-1, 28, 28)
            
            results[sigma] = {
                'mean_intensity': np.mean(samples),
                'std_intensity': np.std(samples),
                'diversity': np.std([np.mean(samples[i]) for i in range(len(samples))])
            }
    
    return results