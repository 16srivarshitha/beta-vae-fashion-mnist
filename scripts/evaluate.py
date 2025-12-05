#!/usr/bin/env python3
"""
Evaluation script for trained VAE models.
Usage: python scripts/evaluate.py --config configs/config.yaml
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from src.data.dataset import prepare_data
from src.models.vae import VAE
from src.evaluation.metrics import calculate_fid_score, analyze_frozen_params
from src.visualization.plots import (
    visualize_reconstructions, 
    generate_and_plot_samples,
    plot_frozen_params,
    create_comparison_table
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trained_models(config, device):
    """Load all trained models from checkpoints."""
    models = {}
    
    for beta in config['training']['beta_values']:
        checkpoint_path = Path(config['output']['checkpoint_dir']) / f'vae_beta_{beta}.pth'
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found for β={beta} at {checkpoint_path}")
            continue
        
        model = VAE(
            latent_dim=config['model']['latent_dim'],
            hidden_dim=config['model']['hidden_dim']
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models[beta] = {
            'model': model,
            'history': checkpoint['history']
        }
        
        print(f"✓ Loaded model for β={beta}")
    
    return models


def main(config_path):
    """Main evaluation pipeline."""
    
    # Load config
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n" + "="*60)
    print("Loading Fashion-MNIST Dataset")
    print("="*60)
    train_loader, test_loader = prepare_data(
        batch_size=config['data']['batch_size'],
        data_root=config['data']['data_root'],
        num_workers=config['data']['num_workers']
    )
    
    # Load trained models
    print("\n" + "="*60)
    print("Loading Trained Models")
    print("="*60)
    results = load_trained_models(config, device)
    
    if not results:
        print("Error: No trained models found. Run 'python scripts/train.py' first.")
        return
    
    # Evaluation on best model (β=1.0)
    print("\n" + "="*60)
    print("Detailed Evaluation (β=1.0 Model)")
    print("="*60)
    
    best_beta = 1.0
    if best_beta not in results:
        best_beta = list(results.keys())[0]
        print(f"Note: Using β={best_beta} as best model")
    
    best_model = results[best_beta]['model']
    
    # Visualize reconstructions
    print("\nGenerating reconstructions...")
    visualize_reconstructions(
        model=best_model,
        test_loader=test_loader,
        device=device,
        n_samples=config['evaluation']['n_reconstructions'],
        username=config['user']['username'],
        output_dir=config['output']['figures_dir']
    )
    
    # Generate new samples
    print("Generating new samples...")
    generated_samples = generate_and_plot_samples(
        model=best_model,
        device=device,
        n_samples=config['evaluation']['n_samples'],
        username=config['user']['username'],
        output_dir=config['output']['figures_dir']
    )
    
    # Calculate FID scores
    print("\n" + "="*60)
    print("Calculating FID Scores")
    print("="*60)
    
    fid_scores = {}
    
    # Extract real images
    print("Extracting real images...")
    real_images = []
    for data, _ in test_loader:
        real_images.append(data.numpy())
        if len(real_images) >= 10:
            break
    real_images = np.concatenate(real_images, axis=0)[:config['evaluation']['fid_samples']]
    real_images = real_images.reshape(-1, 28, 28)
    
    for beta, result in results.items():
        print(f"Calculating FID for β={beta}...")
        model = result['model']
        model.eval()
        
        with torch.no_grad():
            z = torch.randn(config['evaluation']['fid_samples'], model.latent_dim).to(device)
            fake_images = model.decode(z).cpu().numpy().reshape(-1, 28, 28)
        
        fid_score = calculate_fid_score(real_images, fake_images)
        fid_scores[beta] = fid_score
        print(f"  FID Score (β={beta}): {fid_score:.4f}")
    
    # Frozen parameters analysis
    print("\n" + "="*60)
    print("Frozen Parameters Analysis")
    print("="*60)
    
    print("Testing with frozen μ=0 and varying σ...")
    frozen_results = plot_frozen_params(
        model=best_model,
        device=device,
        sigma_values=config['evaluation']['sigma_values'],
        n_samples=8,
        username=config['user']['username'],
        output_dir=config['output']['figures_dir']
    )
    
    frozen_analysis = analyze_frozen_params(
        model=best_model,
        sigma_values=config['evaluation']['sigma_values'],
        n_samples=100,
        device=device
    )
    
    for sigma, metrics in frozen_analysis.items():
        print(f"\nσ={sigma}:")
        print(f"  Mean Intensity: {metrics['mean_intensity']:.4f}")
        print(f"  Std Intensity: {metrics['std_intensity']:.4f}")
        print(f"  Diversity: {metrics['diversity']:.4f}")
    
    # Create comparison visualizations
    print("\n" + "="*60)
    print("Creating Comparison Visualizations")
    print("="*60)
    
    print("Generating comparison table...")
    create_comparison_table(
        results=results,
        username=config['user']['username'],
        output_dir=config['output']['figures_dir']
    )
    
    # Save numerical results
    print("\n" + "="*60)
    print("Saving Numerical Results")
    print("="*60)
    
    results_file = Path(config['output']['output_dir']) / f"numerical_results_{config['user']['username']}.txt"
    
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("VAE Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Latent Dimension: {config['model']['latent_dim']}\n")
        f.write(f"Hidden Dimension: {config['model']['hidden_dim']}\n")
        f.write(f"Epochs: {config['training']['epochs']}\n\n")
        
        f.write("Beta-VAE Comparison:\n")
        f.write("-" * 40 + "\n")
        for beta in sorted(results.keys()):
            history = results[beta]['history']
            f.write(f"\nβ = {beta}\n")
            f.write(f"  Final Total Loss: {history['test_loss'][-1]:.4f}\n")
            f.write(f"  Final Recon Loss: {history['test_recon'][-1]:.4f}\n")
            f.write(f"  Final KL Div: {history['test_kl'][-1]:.4f}\n")
            if beta in fid_scores:
                f.write(f"  FID Score: {fid_scores[beta]:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Frozen Parameters Analysis:\n")
        f.write("-" * 40 + "\n")
        for sigma, metrics in frozen_analysis.items():
            f.write(f"\nσ = {sigma}:\n")
            f.write(f"  Mean Intensity: {metrics['mean_intensity']:.4f}\n")
            f.write(f"  Std Intensity: {metrics['std_intensity']:.4f}\n")
            f.write(f"  Diversity: {metrics['diversity']:.4f}\n")
    
    print(f"Saved numerical results: {results_file}")
    
    # Final Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE - SUMMARY")
    print("="*60)
    
    print(f"\n{'Beta':<10} {'Total Loss':<15} {'Recon Loss':<15} {'KL Div':<15} {'FID Score':<15}")
    print("-" * 70)
    for beta in sorted(results.keys()):
        history = results[beta]['history']
        fid = fid_scores.get(beta, float('nan'))
        print(f"{beta:<10.1f} {history['test_loss'][-1]:<15.4f} "
              f"{history['test_recon'][-1]:<15.4f} {history['test_kl'][-1]:<15.4f} "
              f"{fid:<15.4f}")
    
    print(f"\n\nAll evaluation outputs saved to: {config['output']['figures_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained VAE models')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    main(args.config)