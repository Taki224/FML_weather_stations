import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define features and target based on the report specifications
FEATURES = [
    'Cloud amount', 
    'Dew-point temperature', 
    'Gust speed', 
    'Horizontal visibility', 
    'Precipitation amount', 
    'Precipitation intensity', 
    'Present weather (auto)', 
    'Pressure (msl)', 
    'Relative humidity', 
    'Snow depth', 
    'Wind direction', 
    'Wind speed'
]

TARGET = 'Air temperature'

def load_and_preprocess_data(file_path):
    """
    Loads weather data from a CSV, splits chronologically into Train (60%), 
    Validation (20%), and Test (20%), and standardizes features per station.
    
    Assumes columns: 'timestamp', 'station_id', 'latitude', 'longitude', TARGET, and FEATURES.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Ensure timestamp is datetime and sort chronologically
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Optional: Filter by time range if necessary (Jan 1, 2024 to Feb 1, 2024)
    # df = df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] <= '2024-02-01')]
    
    stations = df['station_id'].unique()
    print(f"Found {len(stations)} unique stations.")
    
    # Split chronologically
    n_samples = len(df)
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Standardize features per station (using training data to fit the scaler)
    # We will iterate over stations to scale independently
    print("Standardizing features (zero mean, unit variance) per station...")
    
    scalers = {}
    for station in stations:
        station_train_mask = train_df['station_id'] == station
        station_val_mask = val_df['station_id'] == station
        station_test_mask = test_df['station_id'] == station
        
        # We need at least some training data to fit the scaler
        if station_train_mask.sum() > 0:
            scaler = StandardScaler()
            
            # Fit and transform on train
            train_df.loc[station_train_mask, FEATURES] = scaler.fit_transform(train_df.loc[station_train_mask, FEATURES])
            scalers[station] = scaler
            
            # Transform on val and test
            if station_val_mask.sum() > 0:
                val_df.loc[station_val_mask, FEATURES] = scaler.transform(val_df.loc[station_val_mask, FEATURES])
            if station_test_mask.sum() > 0:
                test_df.loc[station_test_mask, FEATURES] = scaler.transform(test_df.loc[station_test_mask, FEATURES])
        else:
            # If a station has no training data, we might fill with 0 (standardized mean)
            # or handle it otherwise. For simplicity, we fill 0s.
            if station_val_mask.sum() > 0:
                val_df.loc[station_val_mask, FEATURES] = 0.0
            if station_test_mask.sum() > 0:
                test_df.loc[station_test_mask, FEATURES] = 0.0

    # Extract coordinates for the graph
    # Taking the first occurrence of lat/lon for each station
    coords = df.groupby('station_id')[['latitude', 'longitude']].first()
    
    return train_df, val_df, test_df, coords, stations

def extract_station_data(df, station_id):
    """
    Helper function to extract X and y for a specific station.
    """
    station_df = df[df['station_id'] == station_id]
    X = station_df[FEATURES].values
    y = station_df[TARGET].values
    return X, y
