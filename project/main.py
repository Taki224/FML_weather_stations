import os
import sys

from data_loader import load_and_preprocess_data
from graph_utils import create_delaunay_graph, create_knn_graph
from models import (
    train_local_models, 
    evaluate_models, 
    train_global_model, 
    evaluate_global_model, 
    train_gtvmin_fl
)

def main():
    data_path = 'data.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found in the project directory.")
        print("Please upload your FMI weather station data as a CSV file named 'data.csv'.")
        print("Required columns: timestamp, station_id, latitude, longitude, Air temperature, and the 12 features.")
        sys.exit(1)
        
    print("=== 1. Data Loading and Preprocessing ===")
    train_df, val_df, test_df, coords, stations = load_and_preprocess_data(data_path)
    
    print("\n=== 2. Graph Construction ===")
    A_delaunay = create_delaunay_graph(coords)
    print("Created Delaunay Triangulation Graph.")
    
    A_knn = create_knn_graph(coords, k=3)
    print("Created k-Nearest Neighbors Graph (k=3).")
    
    print("\n=== 3. Baseline: Local-only Models ===")
    local_models = train_local_models(train_df, stations)
    local_val_mse = evaluate_models(local_models, val_df, stations)
    local_test_mse = evaluate_models(local_models, test_df, stations)
    print(f"Local-only Models - Val MSE: {local_val_mse:.4f} | Test MSE: {local_test_mse:.4f}")
    
    print("\n=== 4. Baseline: Global Shared Model ===")
    global_model = train_global_model(train_df)
    global_val_mse = evaluate_global_model(global_model, val_df)
    global_test_mse = evaluate_global_model(global_model, test_df)
    print(f"Global Shared Model - Val MSE: {global_val_mse:.4f} | Test MSE: {global_test_mse:.4f}")
    
    print("\n=== 5. GTVMin FL: Delaunay Triangulation Graph ===")
    delaunay_models = train_gtvmin_fl(
        train_df, val_df, stations, A_delaunay, alpha=10.0, max_iter=50
    )
    delaunay_test_mse = evaluate_models(delaunay_models, test_df, stations)
    print(f"FL Delaunay - Final Test MSE: {delaunay_test_mse:.4f}")
    
    print("\n=== 6. GTVMin FL: k-NN Graph (k=3) ===")
    knn_models = train_gtvmin_fl(
        train_df, val_df, stations, A_knn, alpha=10.0, max_iter=50
    )
    knn_test_mse = evaluate_models(knn_models, test_df, stations)
    print(f"FL k-NN - Final Test MSE: {knn_test_mse:.4f}")
    
    print("\n=== Summary of Test MSE ===")
    print(f"Local-only Baseline: {local_test_mse:.4f}")
    print(f"Global Shared Model: {global_test_mse:.4f}")
    print(f"FL with k-NN Graph : {knn_test_mse:.4f}")
    print(f"FL with Delaunay   : {delaunay_test_mse:.4f}")

if __name__ == "__main__":
    # Seed for reproducibility as requested in report
    import numpy as np
    np.random.seed(42)
    
    main()
