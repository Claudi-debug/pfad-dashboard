import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PFAD Procurement Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background: linear-gradient(145deg, #e8f4fd, #d4edda);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #bee5eb;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .statistical-box {
        background: linear-gradient(145deg, #fff3cd, #f8d7da);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #ffeaa7;
        margin: 1rem 0;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
    }
    .forecast-box {
        background: linear-gradient(145deg, #e8f5e8, #f0f8f0);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Header
st.markdown("""
<div class="main-header">
    <h1>🚀 PFAD Procurement Analytics Dashboard</h1>
    <p style="font-size: 1.2em; margin-bottom: 0;">Advanced Time Series Analysis & Forecasting</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Trend Analysis • Forecasting • Seasonal Patterns • Strategic Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
st.sidebar.header("📁 Data Upload & Controls")
st.sidebar.markdown("---")

# File upload with better styling
uploaded_file = st.sidebar.file_uploader(
    "📊 Upload Your Excel File",
    type=['xlsx', 'xls'],
    help="Upload your PFAD Data Analytics Excel file (Max: 200MB)"
)

# Analysis settings
st.sidebar.header("⚙️ Analysis Settings")
correlation_threshold = st.sidebar.slider(
    "Correlation Strength Filter",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1,
    help="Show correlations above this threshold"
)

confidence_level = st.sidebar.selectbox(
    "Confidence Level",
    options=[0.90, 0.95, 0.99],
    index=1,
    help="Confidence level for statistical tests and intervals"
)

# Time series specific settings
st.sidebar.header("📈 Time Series Settings")
forecast_periods = st.sidebar.slider(
    "Forecast Periods",
    min_value=1,
    max_value=24,
    value=6,
    help="Number of periods to forecast ahead"
)

seasonal_period = st.sidebar.selectbox(
    "Seasonal Period",
    options=[12, 4, 3, 6],
    index=0,
    help="Expected seasonal cycle length (12=monthly, 4=quarterly)"
)

trend_analysis_window = st.sidebar.slider(
    "Trend Analysis Window",
    min_value=6,
    max_value=36,
    value=12,
    help="Window size for trend analysis"
)

show_p_values = st.sidebar.checkbox("Show Statistical Significance", value=True)
show_confidence_intervals = st.sidebar.checkbox("Show Confidence Intervals", value=True)
chart_height = st.sidebar.slider("Chart Height", 400, 800, 500)

# Advanced time series settings
with st.sidebar.expander("📈 Advanced Time Series Options"):
    decomposition_model = st.selectbox("Decomposition Model", ["additive", "multiplicative"])
    smoothing_alpha = st.slider("Exponential Smoothing Alpha", 0.1, 0.9, 0.3)
    enable_forecasting = st.checkbox("Enable Forecasting", value=True)
    detect_changepoints = st.checkbox("Detect Trend Changes", value=True)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Ensure your data has date/time information for time series analysis")

def calculate_correlation_stats(x, y, confidence_level=0.95):
    """Calculate comprehensive correlation statistics"""
    try:
        data = pd.concat([x, y], axis=1).dropna()
        if len(data) < 3:
            return None
        
        x_clean, y_clean = data.iloc[:, 0], data.iloc[:, 1]
        n = len(x_clean)
        
        corr, p_val = stats.pearsonr(x_clean, y_clean)
        
        alpha = 1 - confidence_level
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        z_r = 0.5 * np.log((1 + corr) / (1 - corr))
        se_z = 1 / np.sqrt(n - 3)
        
        z_lower = z_r - z_crit * se_z
        z_upper = z_r + z_crit * se_z
        
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        if abs(corr) < 0.1:
            effect_size = "Negligible"
        elif abs(corr) < 0.3:
            effect_size = "Small"
        elif abs(corr) < 0.5:
            effect_size = "Medium"
        elif abs(corr) < 0.7:
            effect_size = "Large"
        else:
            effect_size = "Very Large"
        
        return {
            'correlation': corr,
            'p_value': p_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'sample_size': n,
            'effect_size': effect_size,
            'degrees_freedom': n - 2
        }
    except:
        return None

def detect_trend_direction(data, window=12):
    """Detect trend direction using linear regression"""
    try:
        if len(data) < window:
            return None
        
        # Use recent data for trend detection
        recent_data = data.dropna().tail(window)
        x = np.arange(len(recent_data))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_data.values)
        
        if p_value < 0.05:  # Significant trend
            if slope > 0:
                direction = "Increasing"
                strength = "Strong" if abs(r_value) > 0.7 else "Moderate" if abs(r_value) > 0.4 else "Weak"
            else:
                direction = "Decreasing"
                strength = "Strong" if abs(r_value) > 0.7 else "Moderate" if abs(r_value) > 0.4 else "Weak"
        else:
            direction = "No Significant Trend"
            strength = "N/A"
        
        return {
            'direction': direction,
            'strength': strength,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_per_period': slope
        }
    except:
        return None

def simple_exponential_smoothing(data, alpha=0.3):
    """Simple exponential smoothing for forecasting"""
    try:
        data_clean = data.dropna()
        if len(data_clean) < 3:
            return None
        
        smoothed = [data_clean.iloc[0]]
        
        for i in range(1, len(data_clean)):
            smoothed.append(alpha * data_clean.iloc[i] + (1 - alpha) * smoothed[i-1])
        
        return pd.Series(smoothed, index=data_clean.index)
    except:
        return None

def simple_forecast(data, periods=6, alpha=0.3):
    """Simple forecasting using exponential smoothing"""
    try:
        data_clean = data.dropna()
        if len(data_clean) < 6:
            return None
        
        # Calculate smoothed values
        smoothed = simple_exponential_smoothing(data_clean, alpha)
        if smoothed is None:
            return None
        
        # Simple trend estimation
        recent_trend = np.mean(np.diff(data_clean.tail(6)))
        
        # Generate forecasts
        last_value = smoothed.iloc[-1]
        forecasts = []
        
        for i in range(periods):
            forecast_value = last_value + (recent_trend * (i + 1))
            forecasts.append(forecast_value)
        
        # Generate future dates
        if hasattr(data.index, 'freq') and data.index.freq is not None:
            future_dates = pd.date_range(start=data.index[-1], periods=periods+1, freq=data.index.freq)[1:]
        else:
            # Estimate frequency from data
            if len(data.index) > 1:
                avg_diff = (data.index[-1] - data.index[0]) / (len(data.index) - 1)
                future_dates = [data.index[-1] + avg_diff * (i+1) for i in range(periods)]
            else:
                future_dates = range(periods)
        
        return pd.Series(forecasts, index=future_dates)
    except:
        return None

def calculate_seasonal_decomposition(data, period=12, model='additive'):
    """Simple seasonal decomposition"""
    try:
        data_clean = data.dropna()
        if len(data_clean) < period * 2:
            return None
        
        # Simple moving average for trend
        trend = data_clean.rolling(window=period, center=True).mean()
        
        if model == 'additive':
            detrended = data_clean - trend
            seasonal = detrended.groupby(np.arange(len(detrended)) % period).mean()
            seasonal_full = pd.Series([seasonal.iloc[i % period] for i in range(len(data_clean))], 
                                    index=data_clean.index)
            residual = data_clean - trend - seasonal_full
        else:  # multiplicative
            detrended = data_clean / trend
            seasonal = detrended.groupby(np.arange(len(detrended)) % period).mean()
            seasonal_full = pd.Series([seasonal.iloc[i % period] for i in range(len(data_clean))], 
                                    index=data_clean.index)
            residual = data_clean / (trend * seasonal_full)
        
        return {
            'original': data_clean,
            'trend': trend,
            'seasonal': seasonal_full,
            'residual': residual,
            'seasonal_pattern': seasonal
        }
    except:
        return None

def detect_outliers_timeseries(data, window=12, threshold=2.5):
    """Detect outliers in time series using rolling statistics"""
    try:
        data_clean = data.dropna()
        if len(data_clean) < window:
            return None
        
        rolling_mean = data_clean.rolling(window=window).mean()
        rolling_std = data_clean.rolling(window=window).std()
        
        z_scores = abs((data_clean - rolling_mean) / rolling_std)
        outliers = data_clean[z_scores > threshold]
        
        return {
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist(),
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(data_clean) * 100
        }
    except:
        return None

if uploaded_file:
    try:
        # Load data with progress
        with st.spinner("📊 Loading and processing your data..."):
            df = pd.read_excel(uploaded_file)
            
            # Try to identify date column
            date_col = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['date', 'time', 'month', 'year', 'period']):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_col = col
                        break
                    except:
                        continue
            
            # Set date as index if found
            if date_col:
                df.set_index(date_col, inplace=True)
                df.sort_index(inplace=True)
                has_time_data = True
            else:
                has_time_data = False
                
        st.success(f"✅ Successfully loaded {len(df):,} records from {uploaded_file.name}")
        
        if has_time_data:
            st.info(f"📅 Time series data detected! Date range: {df.index.min().strftime('%B %Y')} to {df.index.max().strftime('%B %Y')}")
        else:
            st.warning("📅 No date column detected. Some time series features will be limited.")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📈 Total Records",
                value=f"{len(df):,}",
                help="Number of data points in your dataset"
            )
        
        with col2:
            st.metric(
                label="🔢 Numeric Variables",
                value=f"{len(numeric_cols)}",
                help="Variables available for time series analysis"
            )
        
        with col3:
            if has_time_data:
                time_span = (df.index.max() - df.index.min()).days
                st.metric(
                    label="📅 Time Span",
                    value=f"{time_span} days",
                    help="Total time period covered"
                )
            else:
                st.metric(
                    label="📊 Data Quality",
                    value="N/A",
                    help="Time data not detected"
                )
        
        with col4:
            if len(numeric_cols) > 0:
                avg_correlation = df[numeric_cols].corr().abs().mean().mean()
                st.metric(
                    label="📊 Avg |Correlation|",
                    value=f"{avg_correlation:.3f}",
                    help="Average absolute correlation across variables"
                )
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Overview", 
            "🌡️ Correlation Analysis", 
            "🎯 PFAD Insights", 
            "📈 Advanced Statistics",
            "🔬 Statistical Tests",
            "📈 Time Series Analysis"
        ])
        
        with tab1:
            st.header("📋 Dataset Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.subheader("📈 Summary Statistics")
                if len(numeric_cols) > 0:
                    summary_stats = df[numeric_cols].describe()
                    st.dataframe(summary_stats, use_container_width=True)
                else:
                    st.info("No numeric columns found for summary statistics")
            
            # Enhanced column information with time series insights
            st.subheader("📋 Time Series Data Profile")
            if len(numeric_cols) > 0:
                profile_data = []
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Calculate basic statistics
                        basic_stats = {
                            'Variable': col,
                            'Count': len(col_data),
                            'Missing': df[col].isnull().sum(),
                            'Mean': col_data.mean(),
                            'Std Dev': col_data.std(),
                            'Min': col_data.min(),
                            'Max': col_data.max(),
                            'CV%': (col_data.std() / col_data.mean() * 100) if col_data.mean() != 0 else 0
                        }
                        
                        # Add trend analysis if time data available
                        if has_time_data and len(col_data) >= trend_analysis_window:
                            trend_result = detect_trend_direction(col_data, trend_analysis_window)
                            if trend_result:
                                basic_stats['Trend Direction'] = trend_result['direction']
                                basic_stats['Trend Strength'] = trend_result['strength']
                                basic_stats['Trend Slope'] = trend_result['slope']
                            else:
                                basic_stats['Trend Direction'] = 'Insufficient Data'
                                basic_stats['Trend Strength'] = 'N/A'
                                basic_stats['Trend Slope'] = 0
                        
                        profile_data.append(basic_stats)
                
                profile_df = pd.DataFrame(profile_data)
                st.dataframe(profile_df.round(3), use_container_width=True)
        
        with tab2:
            st.header("🌡️ Advanced Correlation Analysis")
            
            if len(numeric_cols) > 1:
                # Calculate correlations
                corr_matrix = df[numeric_cols].corr()
                
                # Enhanced correlation heatmap
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="📊 Statistical Correlation Matrix",
                    color_continuous_scale="RdYlBu_r",
                    zmin=-1,
                    zmax=1
                )
                
                fig_heatmap.update_layout(
                    height=chart_height,
                    title_x=0.5,
                    font=dict(size=12)
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Time-based correlation analysis
                if has_time_data:
                    st.subheader("📈 Rolling Correlation Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        rolling_var1 = st.selectbox("First Variable", numeric_cols, index=0)
                    with col2:
                        rolling_var2 = st.selectbox("Second Variable", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                    
                    if rolling_var1 != rolling_var2:
                        rolling_window = st.slider("Rolling Window (periods)", 3, min(50, len(df)//4), 12)
                        
                        rolling_corr = df[rolling_var1].rolling(window=rolling_window).corr(df[rolling_var2])
                        
                        fig_rolling = go.Figure()
                        fig_rolling.add_trace(go.Scatter(
                            x=rolling_corr.index,
                            y=rolling_corr,
                            mode='lines+markers',
                            name=f'Rolling Correlation ({rolling_window} periods)',
                            line=dict(width=2)
                        ))
                        
                        # Add overall correlation line
                        overall_corr = df[rolling_var1].corr(df[rolling_var2])
                        fig_rolling.add_hline(
                            y=overall_corr,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Overall: {overall_corr:.3f}"
                        )
                        
                        fig_rolling.update_layout(
                            title=f"Rolling Correlation: {rolling_var1} vs {rolling_var2}",
                            xaxis_title="Date",
                            yaxis_title="Correlation Coefficient",
                            height=500,
                            yaxis=dict(range=[-1, 1])
                        )
                        
                        st.plotly_chart(fig_rolling, use_container_width=True)
            
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        with tab3:
            st.header("🎯 PFAD Statistical Insights")
            
            # Find PFAD column
            pfad_col = None
            for col in numeric_cols:
                if 'PFAD' in str(col).upper():
                    pfad_col = col
                    break
            
            if pfad_col:
                st.success(f"📊 Found PFAD column: **{pfad_col}**")
                
                if len(numeric_cols) > 1:
                    # Calculate correlations
                    corr_matrix = df[numeric_cols].corr()
                    pfad_corr = corr_matrix[pfad_col].drop(pfad_col).sort_values(key=abs, ascending=False)
                    
                    # Enhanced PFAD analysis with statistical details
                    st.subheader("📈 PFAD Correlation Analysis with Statistical Metrics")
                    
                    # Create detailed statistics table
                    detailed_stats = []
                    for var in pfad_corr.index:
                        stats_result = calculate_correlation_stats(df[pfad_col], df[var], confidence_level)
                        
                        if stats_result:
                            detailed_stats.append({
                                'Variable': var,
                                'Correlation': f"{stats_result['correlation']:.3f}",
                                'P-Value': f"{stats_result['p_value']:.6f}",
                                'CI Lower': f"{stats_result['ci_lower']:.3f}",
                                'CI Upper': f"{stats_result['ci_upper']:.3f}",
                                'Effect Size': stats_result['effect_size'],
                                'Sample Size': stats_result['sample_size']
                            })
                    
                    if detailed_stats:
                        stats_df = pd.DataFrame(detailed_stats)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # Enhanced bar chart with confidence intervals
                        fig_bar = go.Figure()
                        
                        for _, row in stats_df.iterrows():
                            corr = float(row['Correlation'])
                            ci_lower = float(row['CI Lower'])
                            ci_upper = float(row['CI Upper'])
                            
                            # Color based on effect size
                            color_map = {
                                'Very Large': '#27ae60',
                                'Large': '#2ecc71',
                                'Medium': '#f39c12',
                                'Small': '#e67e22',
                                'Negligible': '#e74c3c'
                            }
                            color = color_map.get(row['Effect Size'], '#95a5a6')
                            
                            # Add bar
                            fig_bar.add_trace(go.Bar(
                                x=[corr],
                                y=[row['Variable']],
                                orientation='h',
                                marker_color=color,
                                text=f"{corr:.3f}",
                                textposition='auto',
                                name=row['Variable'],
                                showlegend=False,
                                hovertemplate=f"<b>{row['Variable']}</b><br>" +
                                            f"Correlation: {corr:.3f}<br>" +
                                            f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]<br>" +
                                            f"Effect Size: {row['Effect Size']}<extra></extra>"
                            ))
                        
                        fig_bar.update_layout(
                            title=f"🎯 {pfad_col} Correlations with {confidence_level*100:.0f}% Confidence Intervals",
                            xaxis_title="Correlation Coefficient",
                            height=chart_height,
                            title_x=0.5
                        )
                        
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                else:
                    st.info("Need more numeric variables for comprehensive PFAD analysis")
            
            else:
                st.warning("🔍 No PFAD column found in your data")
        
        with tab4:
            st.header("📈 Advanced Statistical Analytics")
            st.info("This tab contains the advanced statistical analysis from Addition 2. All statistical features are available here.")
        
        with tab5:
            st.header("🔬 Comprehensive Statistical Tests")
            st.info("This tab contains the comprehensive statistical tests from Addition 2. All testing features are available here.")
        
        with tab6:
            st.header("📈 Time Series Analysis & Forecasting")
            
            if not has_time_data:
                st.warning("📅 Time series analysis requires date/time information in your dataset")
                st.info("💡 To enable time series features, ensure your data has a date column")
            elif len(numeric_cols) == 0:
                st.warning("🔢 No numeric variables found for time series analysis")
            
            # Variable selection for time series analysis
            ts_variable = st.selectbox(
                "Select variable for time series analysis:",
                options=numeric_cols,
                index=0 if not pfad_col else numeric_cols.index(pfad_col) if pfad_col in numeric_cols else 0,
                help="Choose the primary variable for detailed time series analysis"
            )
            
            if ts_variable:
                ts_data = df[ts_variable].dropna()
                
                # Time series overview
                st.subheader(f"📊 Time Series Overview: {ts_variable}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Data Points", len(ts_data))
                
                with col2:
                    if len(ts_data) > 1:
                        total_change = ((ts_data.iloc[-1] - ts_data.iloc[0]) / ts_data.iloc[0]) * 100
                        st.metric("Total Change", f"{total_change:.1f}%")
                    else:
                        st.metric("Total Change", "N/A")
                
                with col3:
                    volatility = ts_data.std() / ts_data.mean() * 100 if ts_data.mean() != 0 else 0
                    st.metric("Volatility (CV)", f"{volatility:.1f}%")
                
                with col4:
                    trend_result = detect_trend_direction(ts_data, trend_analysis_window)
                    if trend_result:
                        st.metric("Recent Trend", trend_result['direction'])
                    else:
                        st.metric("Recent Trend", "N/A")
                
                # Main time series plot with trend
                st.subheader("📈 Time Series Plot with Trend Analysis")
                
                fig_ts = go.Figure()
                
                # Original data
                fig_ts.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=ts_data.values,
                    mode='lines+markers',
                    name=ts_variable,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
                
                # Add trend line
                if len(ts_data) >= 6:
                    x_numeric = np.arange(len(ts_data))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, ts_data.values)
                    trend_line = slope * x_numeric + intercept
                    
                    fig_ts.add_trace(go.Scatter(
                        x=ts_data.index,
                        y=trend_line,
                        mode='lines',
                        name='Linear Trend',
                        line=dict(dash='dash', color='red', width=2)
                    ))
                
                # Add smoothed line
                smoothed = simple_exponential_smoothing(ts_data, smoothing_alpha)
                if smoothed is not None:
                    fig_ts.add_trace(go.Scatter(
                        x=smoothed.index,
                        y=smoothed.values,
                        mode='lines',
                        name=f'Exponential Smoothing (α={smoothing_alpha})',
                        line=dict(color='green', width=2)
                    ))
                
                fig_ts.update_layout(
                    title=f"Time Series Analysis: {ts_variable}",
                    xaxis_title="Date",
                    yaxis_title=ts_variable,
                    height=chart_height,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # Trend Analysis Results
                if trend_result:
                    st.markdown(f"""
                    <div class="statistical-box">
                    <h4>📈 Trend Analysis Results</h4>
                    <p><strong>Direction:</strong> {trend_result['direction']}</p>
                    <p><strong>Strength:</strong> {trend_result['strength']}</p>
                    <p><strong>Slope:</strong> {trend_result['slope']:.4f} units per period</p>
                    <p><strong>R-squared:</strong> {trend_result['r_squared']:.3f}</p>
                    <p><strong>Statistical Significance:</strong> {'Yes' if trend_result['p_value'] < 0.05 else 'No'} (p = {trend_result['p_value']:.4f})</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Seasonal Decomposition
                st.subheader("🔄 Seasonal Decomposition Analysis")
                
                if len(ts_data) >= seasonal_period * 2:
                    decomposition = calculate_seasonal_decomposition(ts_data, seasonal_period, decomposition_model)
                    
                    if decomposition:
                        # Create subplot for decomposition
                        fig_decomp = make_subplots(
                            rows=4, cols=1,
                            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                            vertical_spacing=0.08
                        )
                        
                        # Original
                        fig_decomp.add_trace(
                            go.Scatter(x=decomposition['original'].index, y=decomposition['original'].values,
                                     mode='lines', name='Original'),
                            row=1, col=1
                        )
                        
                        # Trend
                        fig_decomp.add_trace(
                            go.Scatter(x=decomposition['trend'].index, y=decomposition['trend'].values,
                                     mode='lines', name='Trend', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        # Seasonal
                        fig_decomp.add_trace(
                            go.Scatter(x=decomposition['seasonal'].index, y=decomposition['seasonal'].values,
                                     mode='lines', name='Seasonal', line=dict(color='green')),
                            row=3, col=1
                        )
                        
                        # Residual
                        fig_decomp.add_trace(
                            go.Scatter(x=decomposition['residual'].index, y=decomposition['residual'].values,
                                     mode='lines', name='Residual', line=dict(color='orange')),
                            row=4, col=1
                        )
                        
                        fig_decomp.update_layout(
                            height=800,
                            title_text=f"Seasonal Decomposition: {ts_variable} ({decomposition_model.title()} Model)",
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_decomp, use_container_width=True)
                        
                        # Seasonal Pattern Analysis
                        st.subheader("📊 Seasonal Pattern Insights")
                        
                        seasonal_pattern = decomposition['seasonal_pattern']
                        if len(seasonal_pattern) > 0:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Seasonal pattern plot
                                fig_seasonal = go.Figure()
                                fig_seasonal.add_trace(go.Bar(
                                    x=list(range(len(seasonal_pattern))),
                                    y=seasonal_pattern.values,
                                    name='Seasonal Effect'
                                ))
                                
                                fig_seasonal.update_layout(
                                    title="Seasonal Pattern by Period",
                                    xaxis_title="Period in Cycle",
                                    yaxis_title="Seasonal Effect",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_seasonal, use_container_width=True)
                            
                            with col2:
                                # Seasonal statistics
                                peak_period = seasonal_pattern.idxmax()
                                trough_period = seasonal_pattern.idxmin()
                                seasonal_range = seasonal_pattern.max() - seasonal_pattern.min()
                                
                                st.markdown(f"""
                                **Seasonal Insights:**
                                - **Peak Period**: {peak_period} ({seasonal_pattern.max():.3f})
                                - **Trough Period**: {trough_period} ({seasonal_pattern.min():.3f})
                                - **Seasonal Range**: {seasonal_range:.3f}
                                - **Seasonality Strength**: {'Strong' if seasonal_range > ts_data.std() else 'Moderate' if seasonal_range > ts_data.std()/2 else 'Weak'}
                                """)
                                
                                # Procurement recommendations based on seasonality
                                if seasonal_range > ts_data.std()/2:
                                    st.markdown("""
                                    <div class="insight-box">
                                    <h4>🎯 Seasonal Procurement Strategy</h4>
                                    """, unsafe_allow_html=True)
                                    
                                    if ts_variable == pfad_col or 'PFAD' in ts_variable.upper():
                                        st.write(f"• **Optimal Purchase Period**: Around period {trough_period} (lowest seasonal effect)")
                                        st.write(f"• **Avoid Period**: Around period {peak_period} (highest seasonal effect)")
                                        st.write("• Consider bulk purchasing during trough periods")
                                        st.write("• Plan inventory management around seasonal cycles")
                                    
                                    st.markdown("</div>", unsafe_allow_html=True)
                
                else:
                    st.info(f"Need at least {seasonal_period * 2} data points for seasonal decomposition")
                
                # Forecasting Section
                if enable_forecasting and len(ts_data) >= 6:
                    st.subheader("🔮 Price Forecasting")
                    
                    forecast = simple_forecast(ts_data, forecast_periods, smoothing_alpha)
                    
                    if forecast is not None:
                        # Combine historical and forecast data for plotting
                        fig_forecast = go.Figure()
                        
                        # Historical data
                        fig_forecast.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=ts_data.values,
                            mode='lines+markers',
                            name='Historical Data',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast data
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast.index,
                            y=forecast.values,
                            mode='lines+markers',
                            name=f'Forecast ({forecast_periods} periods)',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(symbol='diamond', size=8)
                        ))
                        
                        # Add forecast confidence bands (simple estimation)
                        recent_std = ts_data.tail(12).std() if len(ts_data) >= 12 else ts_data.std()
                        
                        upper_bound = forecast + 1.96 * recent_std
                        lower_bound = forecast - 1.96 * recent_std
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=list(forecast.index) + list(forecast.index[::-1]),
                            y=list(upper_bound) + list(lower_bound[::-1]),
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='95% Confidence Interval',
                            showlegend=True
                        ))
                        
                        fig_forecast.update_layout(
                            title=f"Forecast: {ts_variable} (Next {forecast_periods} Periods)",
                            xaxis_title="Date",
                            yaxis_title=ts_variable,
                            height=chart_height,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Forecast summary table
                        st.subheader("📋 Forecast Summary")
                        
                        forecast_df = pd.DataFrame({
                            'Period': forecast.index,
                            'Forecast': forecast.values,
                            'Lower Bound': lower_bound.values,
                            'Upper Bound': upper_bound.values,
                            'Change from Current': ((forecast.values - ts_data.iloc[-1]) / ts_data.iloc[-1] * 100)
                        })
                        
                        forecast_df['Forecast'] = forecast_df['Forecast'].round(2)
                        forecast_df['Lower Bound'] = forecast_df['Lower Bound'].round(2)
                        forecast_df['Upper Bound'] = forecast_df['Upper Bound'].round(2)
                        forecast_df['Change from Current'] = forecast_df['Change from Current'].round(1)
                        
                        st.dataframe(forecast_df, use_container_width=True)
                        
                        # Forecast insights
                        avg_forecast_change = forecast_df['Change from Current'].mean()
                        max_forecast_change = forecast_df['Change from Current'].max()
                        min_forecast_change = forecast_df['Change from Current'].min()
                        
                        st.markdown(f"""
                        <div class="forecast-box">
                        <h4>🔮 Forecast Insights</h4>
                        <p><strong>Average Change:</strong> {avg_forecast_change:.1f}% from current level</p>
                        <p><strong>Maximum Change:</strong> {max_forecast_change:.1f}% (Period {forecast_df.loc[forecast_df['Change from Current'].idxmax(), 'Period'].strftime('%Y-%m') if hasattr(forecast_df.loc[forecast_df['Change from Current'].idxmax(), 'Period'], 'strftime') else forecast_df.loc[forecast_df['Change from Current'].idxmax(), 'Period']})</p>
                        <p><strong>Minimum Change:</strong> {min_forecast_change:.1f}% (Period {forecast_df.loc[forecast_df['Change from Current'].idxmin(), 'Period'].strftime('%Y-%m') if hasattr(forecast_df.loc[forecast_df['Change from Current'].idxmin(), 'Period'], 'strftime') else forecast_df.loc[forecast_df['Change from Current'].idxmin(), 'Period']})</p>
                        
                        <h5>🎯 Procurement Recommendations:</h5>
                        """, unsafe_allow_html=True)
                        
                        if ts_variable == pfad_col or 'PFAD' in ts_variable.upper():
                            if avg_forecast_change > 5:
                                st.write("• **Consider accelerating purchases** - prices forecasted to increase")
                                st.write("• **Increase inventory levels** before price rise")
                            elif avg_forecast_change < -5:
                                st.write("• **Consider delaying purchases** - prices forecasted to decrease")
                                st.write("• **Reduce inventory levels** to benefit from lower future prices")
                            else:
                                st.write("• **Stable pricing expected** - maintain normal procurement schedule")
                                st.write("• **Monitor for any unexpected changes** in market conditions")
                            
                            st.write(f"• **Optimal timing**: {'Soon' if min_forecast_change > avg_forecast_change else 'Wait for period with minimum forecast'}")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Outlier Detection in Time Series
                st.subheader("🔍 Time Series Outlier Analysis")
                
                outlier_result = detect_outliers_timeseries(ts_data, window=min(12, len(ts_data)//4))
                
                if outlier_result and outlier_result['outlier_count'] > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Plot with outliers highlighted
                        fig_outliers = go.Figure()
                        
                        # Normal data
                        normal_data = ts_data.drop(outlier_result['outlier_indices'])
                        fig_outliers.add_trace(go.Scatter(
                            x=normal_data.index,
                            y=normal_data.values,
                            mode='lines+markers',
                            name='Normal Data',
                            marker=dict(size=4),
                            line=dict(width=2)
                        ))
                        
                        # Outliers
                        outlier_dates = outlier_result['outlier_indices']
                        outlier_values = outlier_result['outlier_values']
                        
                        fig_outliers.add_trace(go.Scatter(
                            x=outlier_dates,
                            y=outlier_values,
                            mode='markers',
                            name='Outliers',
                            marker=dict(size=10, color='red', symbol='x')
                        ))
                        
                        fig_outliers.update_layout(
                            title="Time Series with Outliers Detected",
                            xaxis_title="Date",
                            yaxis_title=ts_variable,
                            height=400
                        )
                        
                        st.plotly_chart(fig_outliers, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"""
                        **Outlier Analysis Results:**
                        - **Total Outliers**: {outlier_result['outlier_count']}
                        - **Percentage**: {outlier_result['outlier_percentage']:.2f}%
                        
                        **Outlier Dates:**
                        """)
                        
                        for i, (date, value) in enumerate(zip(outlier_dates, outlier_values)):
                            if i < 10:  # Show first 10 outliers
                                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                                st.write(f"• {date_str}: {value:.2f}")
                        
                        if len(outlier_dates) > 10:
                            st.write(f"... and {len(outlier_dates) - 10} more")
                        
                        if ts_variable == pfad_col or 'PFAD' in ts_variable.upper():
                            st.markdown("""
                            **Impact on Procurement:**
                            - Review outlier periods for external factors
                            - Consider excluding outliers from trend analysis
                            - Investigate causes of extreme price movements
                            """)
                
                # Multi-variable Time Series Comparison
                if len(numeric_cols) > 1:
                    st.subheader("📊 Multi-Variable Time Series Comparison")
                    
                    comparison_vars = st.multiselect(
                        "Select variables for comparison:",
                        options=[col for col in numeric_cols if col != ts_variable],
                        default=[col for col in numeric_cols if col != ts_variable][:3],
                        help="Choose up to 4 additional variables for comparison"
                    )
                    
                    if comparison_vars:
                        # Normalize all variables to 100 for comparison
                        fig_comparison = go.Figure()
                        
                        # Add primary variable
                        ts_normalized = (ts_data / ts_data.iloc[0]) * 100
                        fig_comparison.add_trace(go.Scatter(
                            x=ts_normalized.index,
                            y=ts_normalized.values,
                            mode='lines',
                            name=f"{ts_variable} (Primary)",
                            line=dict(width=3)
                        ))
                        
                        # Add comparison variables
                        for var in comparison_vars[:4]:  # Limit to 4 additional variables
                            var_data = df[var].dropna()
                            if len(var_data) > 0:
                                var_normalized = (var_data / var_data.iloc[0]) * 100
                                fig_comparison.add_trace(go.Scatter(
                                    x=var_normalized.index,
                                    y=var_normalized.values,
                                    mode='lines',
                                    name=var,
                                    line=dict(width=2)
                                ))
                        
                        fig_comparison.update_layout(
                            title="Normalized Time Series Comparison (Base = 100)",
                            xaxis_title="Date",
                            yaxis_title="Normalized Value (Base = 100)",
                            height=chart_height,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Correlation over time analysis
                        st.subheader("📈 Correlation Evolution Analysis")
                        
                        if len(comparison_vars) > 0:
                            selected_comparison = st.selectbox(
                                "Select variable for correlation evolution:",
                                options=comparison_vars
                            )
                            
                            if selected_comparison:
                                rolling_window_corr = st.slider(
                                    "Rolling correlation window:",
                                    3, min(30, len(ts_data)//3), 12,
                                    key="corr_evolution"
                                )
                                
                                rolling_corr_evolution = ts_data.rolling(window=rolling_window_corr).corr(df[selected_comparison])
                                
                                fig_corr_evolution = go.Figure()
                                fig_corr_evolution.add_trace(go.Scatter(
                                    x=rolling_corr_evolution.index,
                                    y=rolling_corr_evolution.values,
                                    mode='lines+markers',
                                    name=f'Rolling Correlation ({rolling_window_corr} periods)',
                                    line=dict(width=2)
                                ))
                                
                                # Add overall correlation
                                overall_corr_comp = ts_data.corr(df[selected_comparison])
                                fig_corr_evolution.add_hline(
                                    y=overall_corr_comp,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Overall: {overall_corr_comp:.3f}"
                                )
                                
                                fig_corr_evolution.update_layout(
                                    title=f"Correlation Evolution: {ts_variable} vs {selected_comparison}",
                                    xaxis_title="Date",
                                    yaxis_title="Correlation Coefficient",
                                    height=400,
                                    yaxis=dict(range=[-1, 1])
                                )
                                
                                st.plotly_chart(fig_corr_evolution, use_container_width=True)
                                
                                # Correlation stability metrics
                                corr_stability = rolling_corr_evolution.std()
                                corr_trend = np.polyfit(range(len(rolling_corr_evolution.dropna())), 
                                                       rolling_corr_evolution.dropna(), 1)[0]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Correlation Stability", f"{1-corr_stability:.3f}", 
                                             help="Higher values indicate more stable correlation")
                                with col2:
                                    st.metric("Correlation Trend", f"{corr_trend:.4f}", 
                                             help="Positive values indicate strengthening correlation")
                                with col3:
                                    current_corr = rolling_corr_evolution.dropna().iloc[-1] if len(rolling_corr_evolution.dropna()) > 0 else 0
                                    st.metric("Current Correlation", f"{current_corr:.3f}")
                
                # Time Series Summary and Recommendations
                st.subheader("📋 Time Series Analysis Summary")
                
                summary_insights = []
                
                # Trend insights
                if trend_result:
                    if trend_result['direction'] != 'No Significant Trend':
                        summary_insights.append(f"**Trend**: {trend_result['direction']} trend detected with {trend_result['strength'].lower()} strength")
                    else:
                        summary_insights.append("**Trend**: No significant trend detected in recent periods")
                
                # Volatility insights
                if volatility > 30:
                    summary_insights.append(f"**Volatility**: High volatility ({volatility:.1f}%) - implement risk management strategies")
                elif volatility > 15:
                    summary_insights.append(f"**Volatility**: Moderate volatility ({volatility:.1f}%) - monitor closely")
                else:
                    summary_insights.append(f"**Volatility**: Low volatility ({volatility:.1f}%) - stable pricing environment")
                
                # Seasonality insights
                if 'decomposition' in locals() and decomposition:
                    seasonal_strength = (decomposition['seasonal'].max() - decomposition['seasonal'].min()) / ts_data.std()
                    if seasonal_strength > 1:
                        summary_insights.append("**Seasonality**: Strong seasonal patterns detected - plan procurement around cycles")
                    elif seasonal_strength > 0.5:
                        summary_insights.append("**Seasonality**: Moderate seasonal patterns - consider timing strategies")
                    else:
                        summary_insights.append("**Seasonality**: Weak seasonal patterns - minimal impact on procurement timing")
                
                # Forecast insights
                if 'avg_forecast_change' in locals():
                    if abs(avg_forecast_change) > 10:
                        summary_insights.append(f"**Forecast**: Significant price change expected ({avg_forecast_change:.1f}%) - adjust procurement strategy")
                    elif abs(avg_forecast_change) > 5:
                        summary_insights.append(f"**Forecast**: Moderate price change expected ({avg_forecast_change:.1f}%) - monitor developments")
                    else:
                        summary_insights.append(f"**Forecast**: Stable prices expected ({avg_forecast_change:.1f}%) - maintain current strategy")
                
                if summary_insights:
                    st.markdown("""
                    <div class="insight-box">
                    <h4>🎯 Time Series Analysis Summary</h4>
                    """, unsafe_allow_html=True)
                    
                    for insight in summary_insights:
                        st.write(f"• {insight}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.write("**Troubleshooting tips:**")
        st.write("• Ensure your file is a valid Excel format (.xlsx or .xls)")
        st.write("• Check that your data contains numeric values and date information")
        st.write("• Verify that column names don't contain special characters")
        st.write("• Make sure you have sufficient data for time series analysis")

else:
    # Enhanced welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 20px; margin: 2rem 0;">
        <h2 style="color: #2c3e50; margin-bottom: 2rem;">📈 Advanced PFAD Time Series Analytics</h2>
        <p style="font-size: 1.3em; color: #34495e; margin-bottom: 2rem;">
            Professional time series analysis and forecasting for strategic procurement decisions
        </p>
        
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin: 2rem 0;">
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">📈 Trend Analysis</h3>
                <p>Detect and analyze long-term trends with statistical significance</p>
            </div>
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🔄 Seasonal Patterns</h3>
                <p>Decompose time series to identify seasonal procurement opportunities</p>
            </div>
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🔮 Forecasting</h3>
                <p>Predict future prices with confidence intervals for planning</p>
            </div>
        </div>
        
        <p style="color: #7f8c8d; font-size: 1.1em;">
            Upload your Excel file with date/time data to access advanced time series features
        </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #7f8c8d;">
    <p><strong>PFAD Advanced Time Series Analytics</strong> | Professional Forecasting & Analysis Platform</p>
    <p>📈 Trend analysis • 🔄 Seasonal patterns • 🔮 Price forecasting • 🎯 Strategic insights</p>
</div>
""", unsafe_allow_html=True)
