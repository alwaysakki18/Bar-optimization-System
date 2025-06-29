import pandas as pd
import numpy as np
import joblib
from forecasting import forecast_demand
from inventory_logic import calculate_par_level, reorder_recommendation
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)

def run_simulation(start_date, end_date):
    """
    Simulate inventory system performance with detailed logging
    Returns a DataFrame with simulation results
    """
    try:
        # Load full processed data
        data_path = 'data/full_processed_data.csv'
        if not os.path.exists(data_path):
            logging.error("Full processed data file not found")
            return pd.DataFrame()
            
        data = pd.read_csv(data_path)
        logging.info(f"Loaded full processed data with {len(data)} records")
        
        # Convert dates
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date'])
        
        # Filter for simulation period
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        logging.info(f"Running simulation from {start_date} to {end_date}")
        
        # Check if data is within the simulation period
        if data['date'].min() > end_date or data['date'].max() < start_date:
            logging.warning("No data available for the simulation period")
            return pd.DataFrame()
        
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        logging.info(f"Found {len(data)} records in the simulation period")
        
        if data.empty:
            logging.warning("No data available after filtering for simulation period")
            return pd.DataFrame()
        
        results = []
        current_date = start_date
        
        # Get unique bars, brands for simulation
        unique_bars = data['bar_name'].unique()
        unique_brands = data['brand_name'].unique()
        
        while current_date <= end_date:
            logging.info(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
            
            for bar in unique_bars:
                for brand in unique_brands:
                    # Get specific record for this bar/brand/date
                    record = data[
                        (data['date'] == current_date) &
                        (data['bar_name'] == bar) &
                        (data['brand_name'] == brand)
                    ]
                    
                    if record.empty:
                        # Create a default record if missing
                        record = pd.DataFrame([{
                            'date': current_date,
                            'bar_name': bar,
                            'alcohol_type': 'Unknown',
                            'brand_name': brand,
                            'opening_balance_(ml)': 0,
                            'consumed_(ml)': 0
                        }])
                    else:
                        record = record.iloc[0]
                    
                    # Skip if opening balance is missing
                    if pd.isna(record.get('opening_balance_(ml)', np.nan)):
                        logging.warning(f"Skipping {bar}-{brand} due to missing opening balance")
                        continue
                    
                    current_stock = record['opening_balance_(ml)']
                    actual_consumption = record.get('consumed_(ml)', 0)
                    
                    try:
                        # Get alcohol type
                        alcohol = record['alcohol_type']
                        
                        # Forecast demand
                        forecast = forecast_demand(bar, alcohol, brand, str(current_date))
                        
                        # Get historical data (past 30 days)
                        hist_start = current_date - pd.Timedelta(days=30)
                        hist_data = data[
                            (data['date'] < current_date) & 
                            (data['date'] >= hist_start) &
                            (data['bar_name'] == bar) &
                            (data['alcohol_type'] == alcohol) &
                            (data['brand_name'] == brand)
                        ]
                        
                        if hist_data.empty:
                            logging.warning(f"No historical data found for {bar}-{alcohol}-{brand}. Using all available data.")
                            hist_data = data[
                                (data['bar_name'] == bar) &
                                (data['alcohol_type'] == alcohol) &
                                (data['brand_name'] == brand)
                            ]
                        
                        # Calculate par level
                        par_info = calculate_par_level(forecast, hist_data)
                        recommendation = reorder_recommendation(current_stock, par_info['par_level'])
                        
                        # Record results
                        result_entry = {
                            'date': current_date,
                            'bar': bar,
                            'alcohol': alcohol,
                            'brand': brand,
                            'current_stock': current_stock,
                            'actual_consumption': actual_consumption,
                            'forecast': par_info['forecast'],
                            'safety_stock': par_info['safety_stock'],
                            'par_level': par_info['par_level'],
                            'recommendation': recommendation
                        }
                        
                        results.append(result_entry)
                    except Exception as e:
                        logging.error(f"Error processing {bar}-{brand} on {current_date}: {str(e)}")
            
            current_date += pd.Timedelta(days=1)
        
        return pd.DataFrame(results)
    except Exception as e:
        logging.error(f"Simulation failed: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run simulation - use a date range that matches your data
    print("Running simulation...")
    start = datetime.now()
    sim_results = run_simulation('2023-01-01', '2023-01-07')  # Use a range that matches your dataset
    
    if not sim_results.empty:
        sim_results.to_csv('simulation_results.csv', index=False)
        duration = datetime.now() - start
        print(f"Simulation completed in {duration.total_seconds():.2f} seconds")
        print(f"Generated {len(sim_results)} records")
        print("Results saved to simulation_results.csv")
    else:
        print("No simulation results generated")
        print("Check simulation.log for details")