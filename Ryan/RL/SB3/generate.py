import numpy as np
from typing import Tuple, Optional

def generate(n_customers: int = 100, instance_type: str = 'RC', capacity: int = 200, grid_size: float = 100.0, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    if seed is not None:
        np.random.seed(seed)
    
    coords = np.zeros((n_customers + 1, 2))
    
    if instance_type == 'R':
        coords = np.random.rand(n_customers + 1, 2) * grid_size
    
    elif instance_type == 'C':
        coords[0] = [grid_size/2, grid_size/2]
        
        n_clusters = max(3, n_customers // 10)
        cluster_centers = np.random.rand(n_clusters, 2) * grid_size
        
        for i in range(1, n_customers + 1):
            cluster_idx = np.random.randint(0, n_clusters)
            noise = np.random.normal(0, grid_size/20, 2)
            coords[i] = cluster_centers[cluster_idx] + noise
    
    elif instance_type == 'RC':
        coords[0] = [grid_size/2, grid_size/2]
        
        n_clustered = n_customers // 2
        n_random = n_customers - n_clustered
        
        if n_clustered > 0:
            n_clusters = max(2, n_clustered // 15)
            cluster_centers = np.random.rand(n_clusters, 2) * grid_size
            
            for i in range(1, n_clustered + 1):
                cluster_idx = np.random.randint(0, n_clusters)
                noise = np.random.normal(0, grid_size/20, 2)
                coords[i] = cluster_centers[cluster_idx] + noise
        
        if n_random > 0:
            coords[n_clustered + 1:] = np.random.rand(n_random, 2) * grid_size
    
    demands = np.zeros(n_customers + 1, dtype=int)
    demands[1:] = np.random.randint(1, 10, n_customers)
    
    return coords.astype(np.float32), demands, capacity