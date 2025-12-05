"""
Disentanglement metrics for evaluating latent representations.
Implements MIG (Mutual Information Gap), SAP, and DCI metrics.
"""

import torch
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from typing import Tuple, Dict


def compute_mig(
    latents: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 20
) -> float:
    """
    Compute Mutual Information Gap (MIG) metric.
    
    MIG measures how well each latent dimension captures a single
    factor of variation in the data.
    
    Args:
        latents: Latent representations [n_samples, latent_dim]
        labels: Ground truth factors [n_samples]
        n_bins: Number of bins for discretization
        
    Returns:
        MIG score (higher is better, range [0, 1])
    """
    n_samples, latent_dim = latents.shape
    
    # Discretize latent dimensions
    latents_discrete = np.zeros_like(latents, dtype=int)
    for i in range(latent_dim):
        latents_discrete[:, i] = np.digitize(
            latents[:, i],
            bins=np.linspace(latents[:, i].min(), latents[:, i].max(), n_bins)
        )
    
    # Compute mutual information between each latent dim and labels
    mis = []
    for i in range(latent_dim):
        mi = mutual_info_score(labels, latents_discrete[:, i])
        mis.append(mi)
    
    mis = np.array(mis)
    
    # Sort MI values
    sorted_mis = np.sort(mis)[::-1]
    
    # MIG is the gap between top two MI values, normalized by entropy
    if len(sorted_mis) < 2:
        return 0.0
    
    # Normalize by entropy of labels
    h_labels = entropy(np.bincount(labels) / len(labels))
    
    if h_labels == 0:
        return 0.0
    
    mig = (sorted_mis[0] - sorted_mis[1]) / h_labels
    
    return float(mig)


def compute_sap(
    latents: np.ndarray,
    labels: np.ndarray,
    continuous: bool = False
) -> float:
    """
    Compute Separated Attribute Predictability (SAP) score.
    
    SAP measures how well individual latent dimensions predict
    specific factors of variation.
    
    Args:
        latents: Latent representations [n_samples, latent_dim]
        labels: Ground truth factors [n_samples]
        continuous: Whether factors are continuous
        
    Returns:
        SAP score (higher is better, range [0, 1])
    """
    n_samples, latent_dim = latents.shape
    n_classes = len(np.unique(labels))
    
    # Train classifier for each latent dimension
    scores = np.zeros((latent_dim, n_classes))
    
    for i in range(latent_dim):
        # Use single latent dimension to predict labels
        X = latents[:, i:i+1]
        
        if continuous:
            # Use regression for continuous factors
            model = LinearRegression()
            model.fit(X, labels)
            pred = model.predict(X)
            score = 1 - np.mean((pred - labels) ** 2) / np.var(labels)
        else:
            # Use classifier for discrete factors
            model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                random_state=42
            )
            model.fit(X, labels)
            score = model.score(X, labels)
        
        scores[i, 0] = score
    
    # Compute SAP as difference between top two scores
    sap_scores = []
    for j in range(n_classes):
        col_scores = scores[:, j]
        sorted_scores = np.sort(col_scores)[::-1]
        if len(sorted_scores) >= 2:
            sap = sorted_scores[0] - sorted_scores[1]
            sap_scores.append(sap)
    
    return float(np.mean(sap_scores)) if sap_scores else 0.0


def compute_dci(
    latents: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute Disentanglement, Completeness, and Informativeness (DCI) metrics.
    
    Args:
        latents: Latent representations [n_samples, latent_dim]
        labels: Ground truth factors [n_samples]
        
    Returns:
        Dictionary with 'disentanglement', 'completeness', 'informativeness'
    """
    n_samples, latent_dim = latents.shape
    
    # Train classifier to predict labels from latents
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = latents[:split], latents[split:]
    y_train, y_test = labels[:split], labels[split:]
    
    model.fit(X_train, y_train)
    
    # Informativeness: overall prediction accuracy
    informativeness = model.score(X_test, y_test)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Normalize importances per factor
    importance_matrix = importances.reshape(1, -1)
    
    # Disentanglement: each factor should be captured by one dimension
    normalized_importances = importance_matrix / (
        np.sum(importance_matrix, axis=1, keepdims=True) + 1e-12
    )
    disentanglement = 1 - entropy(
        normalized_importances.T,
        base=latent_dim
    ).mean()
    
    # Completeness: each dimension should capture one factor
    normalized_importances_t = importance_matrix / (
        np.sum(importance_matrix, axis=0, keepdims=True) + 1e-12
    )
    completeness = 1 - entropy(
        normalized_importances_t,
        base=importance_matrix.shape[0]
    ).mean()
    
    return {
        'disentanglement': float(disentanglement),
        'completeness': float(completeness),
        'informativeness': float(informativeness)
    }


def compute_factor_vae_metric(
    model,
    data_loader,
    device: str = 'cuda',
    n_samples: int = 10000,
    n_votes: int = 100
) -> float:
    """
    Compute FactorVAE disentanglement metric.
    
    Measures variance of latent dimensions when varying factors.
    
    Args:
        model: Trained VAE model
        data_loader: Data loader
        device: Device to use
        n_samples: Number of samples
        n_votes: Number of voting iterations
        
    Returns:
        FactorVAE score (higher is better)
    """
    model.eval()
    
    # Collect latent representations
    latents_list = []
    labels_list = []
    count = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            if count >= n_samples:
                break
            
            data = data.to(device)
            data_flat = data.view(-1, 784)
            mu, _ = model.encode(data_flat)
            
            latents_list.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
            
            count += len(data)
    
    latents = np.vstack(latents_list)[:n_samples]
    labels = np.hstack(labels_list)[:n_samples]
    
    # Compute variance for each dimension when factor changes
    n_factors = len(np.unique(labels))
    latent_dim = latents.shape[1]
    
    votes = 0
    for _ in range(n_votes):
        # Sample two data points with different factors
        idx1 = np.random.randint(n_samples)
        # Find point with different label
        different_labels = np.where(labels != labels[idx1])[0]
        if len(different_labels) == 0:
            continue
        idx2 = np.random.choice(different_labels)
        
        # Compute variance of latent dimensions
        diff = np.abs(latents[idx1] - latents[idx2])
        
        # The dimension with highest variance should capture the factor
        max_dim = np.argmax(diff)
        
        # Vote if this makes sense (simplified)
        votes += 1 if diff[max_dim] > diff.mean() else 0
    
    return votes / n_votes if n_votes > 0 else 0.0


def evaluate_disentanglement(
    model,
    data_loader,
    device: str = 'cuda',
    n_samples: int = 5000
) -> Dict[str, float]:
    """
    Comprehensive disentanglement evaluation.
    
    Args:
        model: Trained VAE model
        data_loader: Data loader
        device: Device to use
        n_samples: Number of samples to evaluate
        
    Returns:
        Dictionary with all disentanglement metrics
    """
    model.eval()
    
    # Extract latent representations
    print("Extracting latent representations...")
    latents_list = []
    labels_list = []
    count = 0
    
    with torch.no_grad():
        for data, labels in data_loader:
            if count >= n_samples:
                break
            
            data = data.to(device)
            data_flat = data.view(-1, 784)
            mu, _ = model.encode(data_flat)
            
            latents_list.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
            
            count += len(data)
    
    latents = np.vstack(latents_list)[:n_samples]
    labels = np.hstack(labels_list)[:n_samples]
    
    # Compute metrics
    print("Computing MIG...")
    mig = compute_mig(latents, labels)
    
    print("Computing SAP...")
    sap = compute_sap(latents, labels)
    
    print("Computing DCI...")
    dci = compute_dci(latents, labels)
    
    print("Computing FactorVAE metric...")
    factor_vae = compute_factor_vae_metric(
        model, data_loader, device, n_samples=min(n_samples, 1000)
    )
    
    results = {
        'MIG': mig,
        'SAP': sap,
        'DCI_disentanglement': dci['disentanglement'],
        'DCI_completeness': dci['completeness'],
        'DCI_informativeness': dci['informativeness'],
        'FactorVAE': factor_vae
    }
    
    return results


def print_disentanglement_report(metrics: Dict[str, float]) -> None:
    """
    Print formatted disentanglement metrics report.
    
    Args:
        metrics: Dictionary of disentanglement metrics
    """
    print("\n" + "-"*10)
    print("DISENTANGLEMENT METRICS REPORT")
    print("-"*10)
    
    print(f"\nMutual Information Gap (MIG):  {metrics['MIG']:.4f}")
    print(f"   Measures how well each latent captures one factor")
    print(f"   Higher is better (range: [0, 1])")
    
    print(f"\nSeparated Attribute Predictability (SAP):  {metrics['SAP']:.4f}")
    print(f"   Measures predictability of factors from single dimensions")
    print(f"   Higher is better (range: [0, 1])")
    
    print(f"\nDCI Metrics:")
    print(f"  Disentanglement:    {metrics['DCI_disentanglement']:.4f}")
    print(f"  Completeness:       {metrics['DCI_completeness']:.4f}")
    print(f"  Informativeness:    {metrics['DCI_informativeness']:.4f}")
    print(f"   All in range [0, 1], higher is better")
    
    print(f"\nFactorVAE Score:  {metrics['FactorVAE']:.4f}")
    print(f"   Measures factor-wise code variance")
    print(f"   Higher is better")
    
    print("\n" + "-"*10)