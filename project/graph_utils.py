import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import haversine_distances

def calculate_haversine_matrix(coords):
    """
    Computes the Haversine distance matrix between all stations.
    coords: DataFrame with 'latitude' and 'longitude' in degrees.
    Returns distance matrix in kilometers.
    """
    # Convert degrees to radians
    coords_rad = np.radians(coords[['latitude', 'longitude']].values)
    
    # Calculate Haversine distance (result is in radians)
    dist_matrix_rad = haversine_distances(coords_rad, coords_rad)
    
    # Multiply by Earth's radius in km to get kilometers
    dist_matrix_km = dist_matrix_rad * 6371.0
    return dist_matrix_km

def calculate_weights(dist_matrix_km):
    """
    Calculates the edge weights A_{i,j} = max(0, 60 - dist(i,j)).
    """
    A = np.maximum(0, 60.0 - dist_matrix_km)
    np.fill_diagonal(A, 0) # No self-loops in the graph weights for neighbors
    return A

def create_delaunay_graph(coords):
    """
    Constructs the Delaunay Triangulation graph.
    Returns the weighted adjacency matrix based on Haversine distance.
    """
    points = coords[['longitude', 'latitude']].values
    tri = Delaunay(points)
    
    n_nodes = len(points)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    # Extract edges from the simplices (triangles)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                adj_matrix[u, v] = 1
                adj_matrix[v, u] = 1
                
    dist_matrix = calculate_haversine_matrix(coords)
    weights = calculate_weights(dist_matrix)
    
    # Apply mask
    A_delaunay = adj_matrix * weights
    return A_delaunay

def create_knn_graph(coords, k=3):
    """
    Constructs the k-Nearest Neighbors graph based on geographic distance.
    Returns the weighted adjacency matrix.
    """
    dist_matrix = calculate_haversine_matrix(coords)
    weights = calculate_weights(dist_matrix)
    
    n_nodes = len(coords)
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    # For each node, find the k nearest neighbors (excluding itself)
    # dist_matrix has 0s on diagonal
    for i in range(n_nodes):
        # argsort distances for node i
        # The closest is itself (idx 0, dist 0), so we take 1 to k+1
        nearest_indices = np.argsort(dist_matrix[i])[1:k+1]
        for j in nearest_indices:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1 # undirected graph
            
    A_knn = adj_matrix * weights
    return A_knn
