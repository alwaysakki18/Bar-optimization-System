import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import csv
import re

def preprocess_data(filepath):
    """Load and preprocess raw inventory data"""
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    
    # Load data with proper error handling
    try:
        df = pd.read_csv(filepath, quoting=csv.QUOTE_MINIMAL)
    except pd.errors.ParserError:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    
    # Clean column names
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
    
    # Handle scientific notation in closing balance
    if 'closing_balance_(ml)' in df.columns:
        df['closing_balance_(ml)'] = df['closing_balance_(ml)'].apply(
            lambda x: float(x) if isinstance(x, str) and 'E' in x else x
        )
        df['closing_balance_(ml)'] = pd.to_numeric(
            df['closing_balance_(ml)'], errors='coerce'
        )
    
    # Convert dates
    try:
        df['date'] = pd.to_datetime(df['date_time_served'], errors='coerce')
    except KeyError:
        date_col = [col for col in df.columns if 'date' in col.lower()]
        if date_col:
            df['date'] = pd.to_datetime(df[date_col[0]], errors='coerce')
        else:
            df['date'] = pd.NaT
    
    # Create time features
    df['day_of_week'] = df['date'].dt.day_name()
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    
    # Feature engineering - net consumption
    if all(col in df.columns for col in ['opening_balance_(ml)', 'closing_balance_(ml)', 'purchase_(ml)']):
        df['net_consumption'] = (
            df['opening_balance_(ml)'] - 
            df['closing_balance_(ml)'] + 
            df['purchase_(ml)']
        )
    
    # Sort data
    sort_cols = [col for col in ['bar_name', 'brand_name', 'date'] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    
    # Encode categorical features
    le = LabelEncoder()
    for col in ['brand_name', 'bar_name', 'alcohol_type']:
        if col in df.columns:
            encoded_col = col.replace(' ', '_') + '_encoded'
            df[encoded_col] = le.fit_transform(df[col].astype(str))
    
    # Aggregate daily consumption
    group_cols = [col for col in ['bar_name', 'alcohol_type', 'brand_name', 'date'] if col in df.columns]
    if group_cols and 'consumed_(ml)' in df.columns:
        daily_consumption = df.groupby(group_cols)['consumed_(ml)'].sum().reset_index()
    else:
        daily_consumption = pd.DataFrame()
    
    return df, daily_consumption

if __name__ == "__main__":
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the dataset file dynamically
    data_dir = os.path.join(script_dir, 'data')
    data_file = None
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Search for dataset file
    possible_names = [
        'Consumption Dataset - Dataset.csv',
        'consumption_dataset.csv',
        'dataset.csv',
        'consumption_data.csv'
    ]
    
    for name in possible_names:
        test_path = os.path.join(data_dir, name)
        if os.path.exists(test_path):
            data_file = name
            break
    
    # If not found, list available files
    if not data_file:
        print(f"Could not find dataset file in {data_dir}. Available files:")
        print("\n".join(os.listdir(data_dir)))
        raise FileNotFoundError("Required dataset file not found")
    
    data_path = os.path.join(data_dir, data_file)
    print(f"Using dataset file: {data_file}")
    
    # Process data
    df, daily = preprocess_data(data_path)
    print(f"Processed {len(df)} records")
    
    # Build output paths
    processed_path = os.path.join(data_dir, 'processed_data.csv')
    full_processed_path = os.path.join(data_dir, 'full_processed_data.csv')
    
    # Save processed data
    daily.to_csv(processed_path, index=False)
    df.to_csv(full_processed_path, index=False)
    print(f"Saved processed data to {processed_path}")
    print(f"Saved full processed data to {full_processed_path}")