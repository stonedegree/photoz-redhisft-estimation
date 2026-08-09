"""
Shared evaluation metrics for photometric redshift models.
Every model in this project should be scored with these same functions
so comparisons across Weeks 1-3 are apples-to-apples.
"""
import numpy as np


def delta_z(z_pred, z_true):
    """Normalized residual: (z_pred - z_true) / (1 + z_true)."""
    return (z_pred - z_true) / (1 + z_true)


def bias(z_pred, z_true):
    """Mean normalized residual."""
    return np.mean(delta_z(z_pred, z_true))


def nmad(z_pred, z_true):
    """Normalized median absolute deviation (robust scatter statistic)."""
    dz = delta_z(z_pred, z_true)
    return 1.4826 * np.median(np.abs(dz - np.median(dz)))


def outlier_fraction(z_pred, z_true, threshold=0.05):
    """Fraction of objects with |delta_z| > threshold."""
    dz = delta_z(z_pred, z_true)
    return np.mean(np.abs(dz) > threshold)


def summarize(z_pred, z_true, threshold=0.05):
    """Convenience: return all three metrics as a dict."""
    return {
        "bias": bias(z_pred, z_true),
        "nmad": nmad(z_pred, z_true),
        "outlier_fraction": outlier_fraction(z_pred, z_true, threshold),
    }
