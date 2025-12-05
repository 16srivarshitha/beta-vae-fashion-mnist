#!/usr/bin/env python3
"""
Generate latent space interpolations and traversals.
Usage: python scripts/interpolate.py --checkpoint outputs/checkpoints/vae_beta_1.0.pth
"""

import argparse
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vae import VAE
from src.data.dataset import prepare_data
from src.visualization.interpolation import (
    create_interpolation_grid,
    create_traversal_visualization,
    create_interpolation_animation,
    analyze_latent_dimensions,
    plot_latent_dimension_analysis
)


def main(args):
    """Main interpolation generation pipeline."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = VAE(
        latent_dim=checkpoint.get('latent_dim', 20),
        hidden_dim=checkpoint.get('hidden_dim', 400)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded successfully")
    
    # Load data
    print("\nLoading Fashion-MNIST dataset...")
    _, test_loader = prepare_data(
        batch_size=128,
        data_root='./data',
        num_workers=2
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Interpolation Visualizations")
    print("="*60)
    
    if args.interpolation_grid:
        print("\n1. Creating interpolation grid...")
        create_interpolation_grid(
            model=model,
            test_loader=test_loader,
            n_pairs=args.n_pairs,
            n_steps=args.n_steps,
            device=device,
            username=args.username,
            output_dir=output_dir
        )
    
    if args.traversals:
        print("\n2. Creating latent traversals...")
        create_traversal_visualization(
            model=model,
            n_dims=min(args.n_dims, model.latent_dim),
            n_steps=args.n_steps,
            device=device,
            username=args.username,
            output_dir=output_dir
        )
    
    if args.animation:
        print("\n3. Creating interpolation animation...")
        # Get two random images
        data_iter = iter(test_loader)
        images, _ = next(data_iter)
        img1 = images[0:1]
        img2 = images[1:2]
        
        create_interpolation_animation(
            model=model,
            img1=img1,
            img2=img2,
            n_steps=args.animation_steps,
            device=device,
            output_path=output_dir / f'interpolation_animation_{args.username}.gif',
            fps=args.fps
        )
    
    if args.analyze:
        print("\n4. Analyzing latent dimensions...")
        analysis = analyze_latent_dimensions(
            model=model,
            test_loader=test_loader,
            n_samples=args.analysis_samples,
            device=device
        )
        
        plot_latent_dimension_analysis(
            analysis=analysis,
            username=args.username,
            output_dir=output_dir
        )
        
        # Print summary statistics
        print("\nLatent Dimension Analysis Summary:")
        print("-" * 40)
        print(f"Average dimension std: {analysis['dim_stds'].mean():.4f}")
        print(f"Max dimension std: {analysis['dim_stds'].max():.4f}")
        print(f"Min dimension std: {analysis['dim_stds'].min():.4f}")
        print(f"\nMost active dimensions (top 5):")
        top_dims = analysis['dim_activity'].argsort()[::-1][:5]
        for i, dim in enumerate(top_dims, 1):
            print(f"  {i}. Dimension {dim}: {analysis['dim_activity'][dim]:.4f}")
    
    print("\n" + "="*60)
    print("✓ All visualizations generated successfully!")
    print("="*60)
    print(f"\nOutputs saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate latent space visualizations'
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/interpolations',
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--username',
        type=str,
        default='user',
        help='Username for watermark'
    )
    
    # Visualization flags
    parser.add_argument(
        '--interpolation-grid',
        action='store_true',
        help='Generate interpolation grid'
    )
    
    parser.add_argument(
        '--traversals',
        action='store_true',
        help='Generate latent traversals'
    )
    
    parser.add_argument(
        '--animation',
        action='store_true',
        help='Generate interpolation animation'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze latent dimensions'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all visualizations'
    )
    
    # Hyperparameters
    parser.add_argument(
        '--n-pairs',
        type=int,
        default=4,
        help='Number of image pairs for interpolation grid'
    )
    
    parser.add_argument(
        '--n-steps',
        type=int,
        default=10,
        help='Number of interpolation steps'
    )
    
    parser.add_argument(
        '--n-dims',
        type=int,
        default=10,
        help='Number of dimensions to visualize in traversals'
    )
    
    parser.add_argument(
        '--animation-steps',
        type=int,
        default=30,
        help='Number of frames in animation'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='Frames per second for animation'
    )
    
    parser.add_argument(
        '--analysis-samples',
        type=int,
        default=1000,
        help='Number of samples for latent analysis'
    )
    
    args = parser.parse_args()
    
    # If --all is specified, enable all visualizations
    if args.all:
        args.interpolation_grid = True
        args.traversals = True
        args.animation = True
        args.analyze = True
    
    # If no visualization specified, enable interpolation grid by default
    if not any([args.interpolation_grid, args.traversals, 
                args.animation, args.analyze]):
        print("No visualization specified. Enabling interpolation grid by default.")
        print("Use --all to generate all visualizations.")
        args.interpolation_grid = True
    
    main(args)