import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, datetime
import joblib
import os
from forecasting import forecast_demand
from simulation import run_simulation
from inventory_logic import calculate_par_level, reorder_recommendation

# Page config
st.set_page_config(page_title="Bar Inventory Optimizer", layout="wide")

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load data
@st.cache_data
def load_data():
    try:
        raw_df = pd.read_csv('data/full_processed_data.csv')
        daily_df = pd.read_csv('data/processed_data.csv')
        return raw_df, daily_df
    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

raw_df, daily_df = load_data()

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    try:
        if os.path.exists('models/consumption_forecaster.pkl'):
            st.session_state.model_loaded = True
        else:
            st.warning("Forecast model not found. Please train model first.")
    except:
        st.session_state.model_loaded = False

# Sidebar controls
st.sidebar.header("Control Panel")
if not raw_df.empty:
    bar_options = raw_df['bar_name'].unique()
    bar = st.sidebar.selectbox("Select Bar", bar_options)
    
    alcohol_options = raw_df['alcohol_type'].unique()
    alcohol_type = st.sidebar.selectbox("Alcohol Type", alcohol_options)
    
    brand_options = raw_df[
        (raw_df['alcohol_type'] == alcohol_type) &
        (raw_df['bar_name'] == bar)
    ]['brand_name'].unique()
    brand = st.sidebar.selectbox("Brand", brand_options)
else:
    bar = st.sidebar.selectbox("Select Bar", [])
    alcohol_type = st.sidebar.selectbox("Alcohol Type", [])
    brand = st.sidebar.selectbox("Brand", [])

forecast_date = st.sidebar.date_input("Forecast Date", date.today())

# Main content
st.title("🏨 Bar Inventory Optimization System")
st.subheader("Demand Forecasting & Inventory Recommendations")

# Display historical consumption
if not daily_df.empty and not raw_df.empty:
    st.header(f"Historical Consumption: {bar} - {brand}")
    hist_data = daily_df[
        (daily_df['bar_name'] == bar) & 
        (daily_df['brand_name'] == brand)
    ].sort_values('date')
    
    if not hist_data.empty:
        fig = px.line(hist_data, x='date', y='consumed_(ml)', 
                      title=f"Daily Consumption Trend for {brand}")
        st.plotly_chart(fig)
        
        # Show recent consumption stats
        recent = hist_data.tail(7)
        st.write("Recent Consumption Stats:")
        st.dataframe(recent[['date', 'consumed_(ml)']].set_index('date').style.format("{:.0f} ml"))
    else:
        st.warning("No historical data available for this selection")
else:
    st.warning("No data available. Please process data first.")

# Forecasting section
st.header("Demand Forecast")
if st.sidebar.button("Generate Forecast"):
    if st.session_state.model_loaded:
        try:
            # Get forecast using the imported function
            with st.spinner("Calculating demand forecast..."):
                forecast = forecast_demand(bar, alcohol_type, brand, str(forecast_date))
            
            # Get historical data for this specific brand
            brand_hist_data = daily_df[
                (daily_df['bar_name'] == bar) &
                (daily_df['alcohol_type'] == alcohol_type) &
                (daily_df['brand_name'] == brand)
            ]
            
            # Calculate par level
            with st.spinner("Calculating inventory recommendations..."):
                par_info = calculate_par_level(forecast, brand_hist_data)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("Forecasted Demand", f"{par_info['forecast']:.0f} ml")
            col2.metric("Safety Stock", f"{par_info['safety_stock']:.0f} ml")
            col3.metric("Recommended Par Level", f"{par_info['par_level']:.0f} ml", 
                        help="Target inventory level to maintain")
            
            st.subheader("Inventory Recommendation")
            current_stock = st.number_input("Current Stock (ml)", value=0.0, min_value=0.0)
            recommendation = reorder_recommendation(current_stock, par_info['par_level'])
            
            if "CRITICAL" in recommendation:
                st.error(recommendation)
            elif "URGENT" in recommendation:
                st.error(recommendation)
            elif "Warning" in recommendation:
                st.warning(recommendation)
            else:
                st.success(recommendation)
                
            # Show calculation details
            with st.expander("Calculation Details"):
                st.write(f"**Service Level:** {par_info['service_level']*100:.0f}%")
                st.write(f"**Formula:** Par Level = Forecast + Safety Stock")
                st.write(f"**Safety Stock Formula:** Z * σ * √Lead Time")
                st.write(f"**Z-Score:** {norm.ppf(par_info['service_level']):.2f} (for {par_info['service_level']*100:.0f}% service level)")
                
                if not brand_hist_data.empty:
                    st.write(f"**Demand Std Dev:** {np.std(brand_hist_data['consumed_(ml)']):.2f} ml")
        except Exception as e:
            st.error(f"Forecasting failed: {str(e)}")
    else:
        st.warning("Forecast model not loaded. Please train model first.")

# Simulation results
st.header("System Simulation")
st.subheader("Test inventory performance over time")

# Add controls for simulation date range
col1, col2 = st.columns(2)
sim_start = col1.date_input("Simulation Start Date", date(2023, 1, 1))
sim_end = col2.date_input("Simulation End Date", date(2023, 1, 7))

if st.button("Run Simulation"):
    try:
        # Convert to string for the simulation function
        start_str = sim_start.strftime("%Y-%m-%d")
        end_str = sim_end.strftime("%Y-%m-%d")
        
        # Run simulation
        with st.spinner(f"Running simulation from {start_str} to {end_str}..."):
            start_time = datetime.now()
            sim_results = run_simulation(start_str, end_str)
            duration = datetime.now() - start_time
        
        if not sim_results.empty:
            st.success(f"Simulation completed in {duration.total_seconds():.1f} seconds")
            st.success(f"Generated {len(sim_results)} records")
            
            # Show a sample of results
            st.dataframe(sim_results.head(10))
            
            # Performance metrics
            sim_results['abs_error'] = abs(sim_results['actual_consumption'] - sim_results['forecast'])
            sim_results['abs_perc_error'] = sim_results['abs_error'] / sim_results['actual_consumption'].clip(lower=1) * 100
            accuracy = 100 - sim_results['abs_perc_error'].mean()
            
            stockouts = sim_results[sim_results['current_stock'] < sim_results['actual_consumption']]
            stockout_rate = len(stockouts) / len(sim_results) * 100 if len(sim_results) > 0 else 0
            
            col1, col2 = st.columns(2)
            col1.metric("Forecast Accuracy", f"{accuracy:.1f}%")
            col2.metric("Stockout Rate", f"{stockout_rate:.1f}%")
            
            # Visualization
            fig1 = px.scatter(
                sim_results,
                x='actual_consumption',
                y='forecast',
                color='bar',
                title='Actual vs Forecasted Consumption',
                hover_data=['date', 'brand']
            )
            fig1.add_trace(px.line(
                x=[0, sim_results['actual_consumption'].max()], 
                y=[0, sim_results['actual_consumption'].max()]
            ).data[0])
            st.plotly_chart(fig1)
            
            # Inventory status visualization
            fig2 = px.bar(
                sim_results,
                x='date',
                y='current_stock',
                color='recommendation',
                title='Daily Inventory Status',
                hover_data=['brand', 'par_level']
            )
            fig2.add_trace(px.line(
                sim_results, 
                x='date', 
                y='par_level', 
                color_discrete_sequence=['red']
            ).data[0])
            st.plotly_chart(fig2)
            
            # Save results
            st.download_button(
                label="Download Simulation Results",
                data=sim_results.to_csv().encode('utf-8'),
                file_name='simulation_results.csv',
                mime='text/csv'
            )
        else:
            st.warning("No simulation results generated")
            st.info("""
            Possible reasons:
            - No data available for selected period
            - Missing required fields in processed data
            - Date range not covered in dataset
            """)
            st.button("View Simulation Log", on_click=lambda: st.text(open('simulation.log').read()))
            
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")

# Data explorer
st.header("Data Explorer")
if st.checkbox("Show Raw Data") and not raw_df.empty:
    st.dataframe(raw_df.head(1000))