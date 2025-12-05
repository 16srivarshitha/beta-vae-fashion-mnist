#!/usr/bin/env python3
"""
Main training script for VAE experiments.
Usage: python scripts/train.py --config configs/config.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.dataset import prepare_data
from src.models.vae import VAE
from src.training.trainer import VAETrainer
from src.visualization.plots import plot_training_curves


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create necessary directories."""
    dirs = [
        config['output']['output_dir'],
        config['output']['checkpoint_dir'],
        config['output']['figures_dir'],
        config['output']['logs_dir']
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main(config_path):
    """Main training pipeline."""
    
    # Load config
    config = load_config(config_path)
    setup_directories(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Configuration loaded from: {config_path}")
    
    # Prepare data
    print("\n" + "="*60)
    print("Loading Fashion-MNIST Dataset")
    print("="*60)
    train_loader, test_loader = prepare_data(
        batch_size=config['data']['batch_size'],
        data_root=config['data']['data_root'],
        num_workers=config['data']['num_workers']
    )
    
    # Train models with different beta values
    print("\n" + "="*60)
    print("Training VAE Models with β-VAE Variants")
    print("="*60)
    
    results = {}
    
    for beta in config['training']['beta_values']:
        print(f"\n{'='*60}")
        print(f"Training Model with β = {beta}")
        print(f"{'='*60}")
        
        # Initialize model
        model = VAE(
            latent_dim=config['model']['latent_dim'],
            hidden_dim=config['model']['hidden_dim']
        ).to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate']
        )
        
        # Initialize trainer
        trainer = VAETrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            output_dir=config['output']['figures_dir']
        )
        
        # Train with GIF generation
        history, gif_path = trainer.train_with_gif(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['training']['epochs'],
            beta=beta,
            username=config['user']['username'],
            selected_classes=config['data']['selected_classes'],
            save_interval=config['training']['save_interval']
        )
        
        results[beta] = {
            'model': model,
            'history': history,
            'gif_path': gif_path
        }
        
        # Plot training curves
        plot_training_curves(
            history=history,
            beta_value=beta,
            username=config['user']['username'],
            output_dir=config['output']['figures_dir']
        )
        
        # Save checkpoint
        checkpoint_path = Path(config['output']['checkpoint_dir']) / f'vae_beta_{beta}.pth'
        trainer.save_checkpoint(
            epoch=config['training']['epochs'],
            beta=beta,
            history=history,
            filepath=checkpoint_path
        )
        
        print(f"✓ Completed training for β={beta}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"\n{'Beta':<10} {'Total Loss':<15} {'Recon Loss':<15} {'KL Div':<15}")
    print("-" * 55)
    for beta in config['training']['beta_values']:
        history = results[beta]['history']
        print(f"{beta:<10.1f} {history['test_loss'][-1]:<15.4f} "
              f"{history['test_recon'][-1]:<15.4f} {history['test_kl'][-1]:<15.4f}")
    
    print(f"\n\nAll outputs saved to: {config['output']['output_dir']}/")
    print("Run 'python scripts/evaluate.py' to generate evaluation metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE models')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    main(args.config)