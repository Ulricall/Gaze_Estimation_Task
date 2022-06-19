import numpy as np

def change_dim(x):
    x = list(x)
    
    x_new = []
    for i in x:
        i_new = [i[0], i[2]]
        x_new.append(i_new)
    
    x_new = np.array(x_new)
    
    return x_new
    
def angular_error_3d(a, b):
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    
    ab = np.sum(np.multiply(a, b), axis = 1)
    a_norm = np.linalg.norm(a, axis = 1)
    b_norm = np.linalg.norm(b, axis = 1)
    a_norm = np.clip(a_norm, a_min = 1e-7, a_max = None)
    b_norm = np.clip(b_norm, a_min = 1e-7, a_max = None)
    
    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    return np.arccos(similarity) * 180.0 / np.pi

def angular_error_2d(a, b):
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    
    b_2d = change_dim(b)
    
    ab = np.sum(np.multiply(a, b_2d), axis = 1)
    a_norm = np.linalg.norm(a, axis = 1)
    b_norm = np.linalg.norm(b_2d, axis = 1)
    a_norm = np.clip(a_norm, a_min = 1e-7, a_max = None)
    b_norm = np.clip(b_norm, a_min = 1e-7, a_max = None)
    
    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    return np.arccos(similarity) * 180.0 / np.pi

def angular_error_2d_2(a, b):
    a = a.cpu().detach().numpy()
    b = b.cpu().detach().numpy()
    
    ab = np.sum(np.multiply(a, b), axis = 1)
    a_norm = np.linalg.norm(a, axis = 1)
    b_norm = np.linalg.norm(b, axis = 1)
    a_norm = np.clip(a_norm, a_min = 1e-7, a_max = None)
    b_norm = np.clip(b_norm, a_min = 1e-7, a_max = None)
    
    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    return np.arccos(similarity) * 180.0 / np.pi