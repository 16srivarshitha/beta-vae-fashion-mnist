"""
Setup script for β-VAE Fashion-MNIST package.
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="beta-vae-fashion-mnist",
    version="1.0.0",
    author="Srivarshitha M",
    author_email="varshitham2022@gmail.com",
    description="β-VAE implementation on Fashion-MNIST with comprehensive analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/16srivarshitha/beta-vae-fashion-mnist",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "imageio>=2.31.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
        "gradio>=3.40.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.5.0",
        ],
        "tracking": [
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vae-train=scripts.train:main",
            "vae-evaluate=scripts.evaluate:main",
            "vae-interpolate=scripts.interpolate:main",
            "vae-demo=demo.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
)