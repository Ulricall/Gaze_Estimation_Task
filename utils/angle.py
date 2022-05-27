import numpy as np

def angular_error(a, b):
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    ab = np.sum(np.multiply(a, b), axis = 1)
    a_norm = np.linalg.norm(a, axis = 1)
    b_norm = np.linalg.norm(b, axis = 1)
    a_norm = np.clip(a_norm, a_min = 1e-7, a_max = None)
    b_norm = np.clip(b_norm, a_min = 1e-7, a_max = None)
    
    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    return np.arccos(similarity) * 180.0 / np.pi
