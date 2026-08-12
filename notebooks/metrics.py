import numpy as np

def delta_z(z_pred, z_true):
    return (z_pred - z_true) / (1 + z_true)
    
def bias(z_pred, z_true):
    return np.mean(delta_z(z_pred, z_true))
    
def nmad(z_pred, z_true):    
    dz = delta_z(z_pred, z_true)    
    return 1.4826 * np.median(np.abs(dz - np.median(dz)))
    
def outlier_fraction(z_pred, z_true, threshold=0.05):
    dz = delta_z(z_pred, z_true)
    return np.mean(np.abs(dz) > threshold)