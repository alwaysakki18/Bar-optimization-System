import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import traceback
import pickle

# SOLUTION: Use a Pipeline with ColumnTransformer for robust serialization
# This prevents the 'norm' error by properly encapsulating preprocessing

def train_forecasting_model():
    """Train demand forecasting model with robust serialization"""
    try:
        # Use absolute path for data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data', 'processed_data.csv')
        
        # Load processed data
        daily_data = pd.read_csv(data_path)
        
        # Prepare dataset
        df = daily_data.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Create time features
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        # Prepare features and target
        categorical_cols = ['bar_name', 'alcohol_type', 'brand_name', 'day_of_week', 'month', 'day']
        features = df[categorical_cols]
        target = df['consumed_(ml)']
        
        # Create model directory
        model_dir = os.path.join(script_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a robust pipeline with ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
        ])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Model training
        model.fit(X_train, y_train)
        
        # Evaluation
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"Model Performance: MAE={mae:.2f}, RMSE={rmse:.2f}")
        
        # Save model (absolute path)
        model_path = os.path.join(model_dir, 'consumption_forecaster.pkl')
        
        # Use protocol=4 for better compatibility
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)
            
        print(f"Model saved to {model_path}")
        
        return model
    
    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        return None

def forecast_demand(bar_name, alcohol_type, brand_name, date):
    """Forecast demand for a specific item at a bar on a given date"""
    try:
        # Use absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'models', 'consumption_forecaster.pkl')
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Prepare input features
        input_date = pd.to_datetime(date)
        features = pd.DataFrame({
            'bar_name': [bar_name],
            'alcohol_type': [alcohol_type],
            'brand_name': [brand_name],
            'day_of_week': [input_date.day_name()],
            'month': [input_date.month],
            'day': [input_date.day]
        })
        
        # Make prediction
        return model.predict(features)[0]
    
    except Exception as e:
        print(f"Forecasting failed: {str(e)}")
        traceback.print_exc()
        # Return a safe default value
        return 1000.0  # Default prediction of 1000 ml

if __name__ == "__main__":
    # Train and save model
    model = train_forecasting_model()
    
    # Test forecasting
    if model is not None:
        test_pred = forecast_demand("Main Bar", "Whiskey", "Premium Brand", "2023-07-15")
        print(f"Test prediction: {test_pred:.2f} ml")