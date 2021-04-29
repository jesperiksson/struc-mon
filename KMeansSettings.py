from dataclasses import dataclass, field
@dataclass 
class KMeansSettings():
    n_clusters : int = 3
    init : str = 'k-means++'
    n_init : int = 5
    max_iter : int = 300
    tol : float = 1e-4    
    verbose : int = 1

