import numpy as np
from scipy.stats import norm

def calculate_par_level(forecast, historical_data, service_level=0.95):
    """
    Calculate par level with robust error handling
    Par Level = Forecast Demand + Safety Stock
    Safety Stock = Z * σ * √Lead Time
    """
    try:
        # Handle empty historical data
        if (historical_data is None or historical_data.empty or 
            'consumed_(ml)' not in historical_data.columns or
            historical_data['consumed_(ml)'].isnull().all()):
            return {
                'forecast': round(forecast, 2),
                'safety_stock': round(0.2 * forecast, 2),  # Default to 20% of forecast
                'par_level': round(1.2 * forecast, 2),
                'service_level': service_level
            }
        
        # Filter out null consumption values
        valid_data = historical_data[historical_data['consumed_(ml)'].notnull()]
        
        # Calculate demand variability (std dev)
        if len(valid_data) > 1:
            demand_std = np.std(valid_data['consumed_(ml)'])
        else:
            demand_std = 0.3 * forecast  # Default to 30% of forecast
        
        # Handle NaN/inf values
        if np.isnan(demand_std) or np.isinf(demand_std) or demand_std <= 0:
            demand_std = 0.25 * forecast
        
        # Z-score for service level
        z_score = norm.ppf(service_level)
        
        # Assume 1-day lead time
        lead_time = 1
        
        # Calculate safety stock
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        
        # Calculate par level
        par_level = forecast + safety_stock
        
        return {
            'forecast': round(forecast, 2),
            'safety_stock': round(safety_stock, 2),
            'par_level': round(par_level, 2),
            'service_level': service_level
        }
    except Exception as e:
        print(f"Error calculating par level: {str(e)}")
        return {
            'forecast': round(forecast, 2),
            'safety_stock': round(0.2 * forecast, 2),
            'par_level': round(1.2 * forecast, 2),
            'service_level': service_level
        }

def reorder_recommendation(current_stock, par_level):
    """Generate inventory reorder recommendation"""
    try:
        # Convert to float if needed
        current_stock = float(current_stock)
        par_level = float(par_level)
        
        if current_stock < par_level * 0.4:
            return "CRITICAL: Reorder immediately! Stock below 40% of par level"
        elif current_stock < par_level * 0.6:
            return "URGENT: Reorder today - Stock below 60% of par level"
        elif current_stock < par_level * 0.8:
            return "Warning: Reorder suggested - Stock below 80% of par level"
        return "Stock sufficient"
    except (TypeError, ValueError):
        return "Error: Invalid stock values"