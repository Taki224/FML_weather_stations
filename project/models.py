import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_loader import extract_station_data, FEATURES

def train_local_models(train_df, stations):
    """
    Trains independent Linear Regression models for each station.
    Returns a dictionary mapping station_id to its trained model.
    """
    models = {}
    for station in stations:
        X_train, y_train = extract_station_data(train_df, station)
        if len(X_train) > 0:
            model = LinearRegression()
            model.fit(X_train, y_train)
            models[station] = model
        else:
            models[station] = None
    return models

def evaluate_models(models, df, stations):
    """
    Evaluates a dictionary of local models on the provided dataframe.
    Returns the average MSE across all stations.
    """
    mses = []
    for station in stations:
        if models[station] is not None:
            X_eval, y_eval = extract_station_data(df, station)
            if len(X_eval) > 0:
                preds = models[station].predict(X_eval)
                mse = mean_squared_error(y_eval, preds)
                mses.append(mse)
    
    if len(mses) == 0:
        return np.nan
    return np.mean(mses)

def train_global_model(train_df):
    """
    Trains a single global Linear Regression model on all pooled data.
    """
    X_train = train_df[FEATURES].values
    y_train = train_df['Air temperature'].values
    
    model = LinearRegression()
    if len(X_train) > 0:
        model.fit(X_train, y_train)
    return model

def evaluate_global_model(model, df):
    """
    Evaluates the global model on the pooled data.
    """
    X_eval = df[FEATURES].values
    y_eval = df['Air temperature'].values
    
    if len(X_eval) > 0:
        preds = model.predict(X_eval)
        return mean_squared_error(y_eval, preds)
    return np.nan

def train_gtvmin_fl(train_df, val_df, stations, A, alpha=10.0, max_iter=50, tol=1e-4):
    """
    Trains models using Graph Total Variation Minimization (GTVMin) via 
    Block Coordinate Descent.
    
    A: weighted adjacency matrix (numpy array) with nodes aligned to the 'stations' list.
    """
    n_nodes = len(stations)
    
    # Initialize models by training independently
    models = train_local_models(train_df, stations)
    
    # Pre-extract data and compute m_i (number of train samples)
    X_train_dict = {}
    y_train_dict = {}
    m_dict = {}
    
    for i, station in enumerate(stations):
        X_train, y_train = extract_station_data(train_df, station)
        X_train_dict[i] = X_train
        y_train_dict[i] = y_train
        m_dict[i] = len(X_train)
        
        # If a station has no model (no data), initialize with zero weights
        if models[station] is None:
            model = LinearRegression()
            model.coef_ = np.zeros(len(FEATURES))
            model.intercept_ = 0.0
            models[station] = model
            
    # Calculate degree for each node (sum of weights)
    d = np.sum(A, axis=1)
    
    prev_val_mse = evaluate_models(models, val_df, stations)
    print(f"Initial Local Models Val MSE: {prev_val_mse:.4f}")
    
    for t in range(max_iter):
        new_models = {}
        
        for i, station in enumerate(stations):
            # Compute w_bar (weighted average of neighbors' parameters)
            w_bar = np.zeros(len(FEATURES))
            if d[i] > 0:
                for j in range(n_nodes):
                    if A[i, j] > 0:
                        w_bar += A[i, j] * models[stations[j]].coef_
                w_bar /= d[i]
            else:
                w_bar = models[station].coef_ # no neighbors, pull to itself
                
            # Compute scaling S
            # S = sqrt(m_i * alpha * d_i)
            S = np.sqrt(m_dict[i] * alpha * d[i])
            
            # Augment data
            X_i = X_train_dict[i]
            y_i = y_train_dict[i]
            
            if m_dict[i] > 0:
                # Augment X
                S_I = S * np.eye(len(FEATURES))
                X_aug = np.vstack((X_i, S_I))
                
                # Augment y
                y_aug = np.concatenate((y_i, S * w_bar))
                
                # Update model
                model = LinearRegression()
                # To match the explicit formula which doesn't fit intercept on the penalty part,
                # we fit intercept=False on augmented if data is centered, 
                # but sklearn handles intercept properly if we let it, or we center everything.
                # Since we standardized features, X has zero mean. 
                # For exact matching of Ridge without intercept penalty:
                model.fit(X_aug, y_aug)
                new_models[station] = model
            else:
                new_models[station] = models[station]
                
        models = new_models
        
        val_mse = evaluate_models(models, val_df, stations)
        change = abs(prev_val_mse - val_mse)
        print(f"Iteration {t+1}/{max_iter} - Val MSE: {val_mse:.4f} (change: {change:.6f})")
        
        if change < tol:
            print("Convergence criteria met.")
            break
            
        prev_val_mse = val_mse
        
    return models
