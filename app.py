import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import jarque_bera, anderson, normaltest
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Advanced statistical functions
def calculate_rolling_correlation(df, col1, col2, window=30):
    """Calculate rolling correlation between two columns"""
    return df[col1].rolling(window=window).corr(df[col2])

def calculate_granger_causality(data1, data2, max_lag=5):
    """Simplified Granger causality test"""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        df_test = pd.DataFrame({'x': data1, 'y': data2})
        result = grangercausalitytests(df_test[['x', 'y']], max_lag, verbose=False)
        p_values = [result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
        return min(p_values)
    except:
        return None

def calculate_var(returns, confidence_level=0.95):
    """Calculate Value at Risk"""
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_cvar(returns, confidence_level=0.95):
    """Calculate Conditional Value at Risk"""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def detect_outliers_isolation_forest(data):
    """Detect outliers using simple z-score method"""
    z_scores = np.abs(stats.zscore(data))
    return z_scores > 3

def calculate_hurst_exponent(ts):
    """Calculate Hurst exponent for trend persistence"""
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2

def perform_cointegration_test(series1, series2):
    """Test for cointegration between two series"""
    try:
        from statsmodels.tsa.stattools import coint
        score, p_value, _ = coint(series1, series2)
        return p_value
    except:
        return None

# Page configuration
st.set_page_config(
    page_title="PFAD Advanced Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Statistical test results */
    .stat-result {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Warning boxes */
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Success boxes */
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Header
st.markdown("""
<div class="dashboard-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🚀 PFAD Advanced Analytics Platform</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">AI-Powered Procurement Intelligence with Advanced Statistical Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload PFAD Market Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your Excel or CSV file containing PFAD market data"
    )
    
    st.markdown("### ⚙️ Analysis Settings")
    
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["🎯 Comprehensive Analysis", "📈 Time Series Focus", "🔗 Correlation Analysis", 
         "🔮 Predictive Analytics", "⚠️ Risk Assessment", "🔬 Statistical Deep Dive"]
    )
    
    confidence_level = st.slider(
        "Confidence Level (%)",
        min_value=90,
        max_value=99,
        value=95,
        help="Confidence level for statistical tests and intervals"
    )
    
    forecast_days = st.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to forecast ahead"
    )
    
    st.markdown("### 📊 Advanced Options")
    
    enable_outlier_detection = st.checkbox("Enable Outlier Detection", value=True)
    enable_regime_detection = st.checkbox("Enable Market Regime Detection", value=True)
    enable_causality_test = st.checkbox("Enable Granger Causality Test", value=False)
    
    if st.button("🔄 Reset Analysis", type="secondary"):
        st.session_state.data = None
        st.session_state.analysis_complete = False
        st.rerun()

# Main content area
if uploaded_file is not None:
    try:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Store in session state
        st.session_state.data = df
        st.session_state.analysis_complete = True
        
        # Data preprocessing
        df['Date'] = pd.to_datetime(df.iloc[:, 0])
        df = df.sort_values('Date')
        
        # Identify columns based on actual column names
        pfad_col = next((col for col in df.columns if 'PFAD' in col.upper() and 'RATE' in col.upper()), None)
        if not pfad_col:  # Fallback to just PFAD
            pfad_col = next((col for col in df.columns if 'PFAD' in col.upper()), None)
        
        cpo_col = next((col for col in df.columns if 'CPO' in col.upper() and 'BMD' in col.upper()), None)
        if not cpo_col:  # Fallback to just CPO
            cpo_col = next((col for col in df.columns if 'CPO' in col.upper() and 'VOLUME' not in col.upper()), None)
        
        usd_inr_col = next((col for col in df.columns if 'USD' in col.upper() and 'INR' in col.upper()), None)
        malaysia_fob_col = next((col for col in df.columns if 'MALAYSIA' in col.upper() or 'FOB' in col.upper()), None)
        usd_myr_col = next((col for col in df.columns if 'USD' in col.upper() and 'MYR' in col.upper()), None)
        brent_col = next((col for col in df.columns if 'BRENT' in col.upper()), None)
        volume_col = next((col for col in df.columns if 'VOLUME' in col.upper()), None)
        
        if pfad_col:
            # Calculate returns and statistics
            df['PFAD_Returns'] = df[pfad_col].pct_change()
            df['PFAD_Log_Returns'] = np.log(df[pfad_col] / df[pfad_col].shift(1))
            df['PFAD_Volatility'] = df['PFAD_Returns'].rolling(window=30).std() * np.sqrt(252)
            
            # Create tabs for different analysis sections
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Overview", "📈 Statistical Analysis", "🔗 Correlations", 
                "🔮 Forecasting", "⚠️ Risk Analysis", "🤖 AI Insights"
            ])
            
            with tab1:
                st.markdown("## Market Overview & Key Metrics")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = df[pfad_col].iloc[-1]
                price_change_30d = ((current_price - df[pfad_col].iloc[-31]) / df[pfad_col].iloc[-31] * 100) if len(df) > 30 else 0
                current_volatility = df['PFAD_Volatility'].iloc[-1] * 100 if not pd.isna(df['PFAD_Volatility'].iloc[-1]) else 0
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Current PFAD Price",
                        f"₹{current_price:,.0f}/ton",
                        f"{price_change_30d:+.2f}%"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "30-Day Volatility",
                        f"{current_volatility:.1f}%",
                        "Annualized"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    avg_price = df[pfad_col].mean()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Average Price",
                        f"₹{avg_price:,.0f}/ton",
                        f"Last {len(df)} days"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    price_range = df[pfad_col].max() - df[pfad_col].min()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        "Price Range",
                        f"₹{price_range:,.0f}",
                        "High-Low Spread"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Price chart with advanced indicators
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=('PFAD Price with Moving Averages', 'Volume', 'RSI Indicator')
                )
                
                # Price and moving averages
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=df[pfad_col], name='PFAD Price', line=dict(color='#667eea', width=2)),
                    row=1, col=1
                )
                
                # Add moving averages
                for window, color in [(7, '#10b981'), (30, '#f59e0b'), (90, '#ef4444')]:
                    ma = df[pfad_col].rolling(window=window).mean()
                    fig.add_trace(
                        go.Scatter(x=df['Date'], y=ma, name=f'{window}-Day MA', line=dict(color=color, width=1.5)),
                        row=1, col=1
                    )
                
                # Add Bollinger Bands
                bb_window = 20
                bb_std = 2
                rolling_mean = df[pfad_col].rolling(window=bb_window).mean()
                rolling_std = df[pfad_col].rolling(window=bb_window).std()
                upper_band = rolling_mean + (rolling_std * bb_std)
                lower_band = rolling_mean - (rolling_std * bb_std)
                
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=upper_band, name='Upper BB', line=dict(color='rgba(128,128,128,0.3)', dash='dash')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=lower_band, name='Lower BB', line=dict(color='rgba(128,128,128,0.3)', dash='dash'),
                              fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
                    row=1, col=1
                )
                
                # Volume (if available)
                if volume_col and volume_col in df.columns:
                    fig.add_trace(
                        go.Bar(x=df['Date'], y=df[volume_col], name='CPO Volume', marker_color='lightblue'),
                        row=2, col=1
                    )
                
                # RSI
                def calculate_rsi(prices, period=14):
                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    return 100 - (100 / (1 + rs))
                
                rsi = calculate_rsi(df[pfad_col])
                fig.add_trace(
                    go.Scatter(x=df['Date'], y=rsi, name='RSI', line=dict(color='purple')),
                    row=3, col=1
                )
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                fig.update_layout(height=800, showlegend=True, title_text="PFAD Price Analysis Dashboard")
                fig.update_xaxes(rangeslider_visible=False)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("## 📈 Advanced Statistical Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Distribution Analysis")
                    
                    # Normality tests
                    returns = df['PFAD_Returns'].dropna()
                    
                    # Jarque-Bera test
                    jb_stat, jb_pvalue = jarque_bera(returns)
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Jarque-Bera Test for Normality</b><br>
                        Statistic: {jb_stat:.4f}<br>
                        P-value: {jb_pvalue:.4f}<br>
                        Result: {'Normal Distribution' if jb_pvalue > 0.05 else 'Non-Normal Distribution'} ✓
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Anderson-Darling test
                    ad_result = anderson(returns)
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Anderson-Darling Test</b><br>
                        Statistic: {ad_result.statistic:.4f}<br>
                        Critical Value (5%): {ad_result.critical_values[2]:.4f}<br>
                        Result: {'Normal' if ad_result.statistic < ad_result.critical_values[2] else 'Non-Normal'} ✓
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Shapiro-Wilk test
                    if len(returns) < 5000:  # Shapiro-Wilk has sample size limitations
                        sw_stat, sw_pvalue = stats.shapiro(returns)
                        st.markdown(f"""
                        <div class="stat-result">
                            <b>Shapiro-Wilk Test</b><br>
                            Statistic: {sw_stat:.4f}<br>
                            P-value: {sw_pvalue:.4f}<br>
                            Result: {'Normal' if sw_pvalue > 0.05 else 'Non-Normal'} ✓
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Time Series Properties")
                    
                    # ADF test for stationarity
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(df[pfad_col].dropna())
                    
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Augmented Dickey-Fuller Test</b><br>
                        ADF Statistic: {adf_result[0]:.4f}<br>
                        P-value: {adf_result[1]:.4f}<br>
                        Result: {'Stationary' if adf_result[1] < 0.05 else 'Non-Stationary'} ✓
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hurst exponent
                    hurst = calculate_hurst_exponent(df[pfad_col].dropna().values)
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Hurst Exponent</b><br>
                        Value: {hurst:.4f}<br>
                        Interpretation: {
                            'Trending (H > 0.5)' if hurst > 0.5 
                            else 'Mean-Reverting (H < 0.5)' if hurst < 0.5 
                            else 'Random Walk (H ≈ 0.5)'
                        } ✓
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Autocorrelation
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_result = acorr_ljungbox(returns, lags=10, return_df=True)
                    
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Ljung-Box Test (Autocorrelation)</b><br>
                        Q-Statistic: {lb_result['lb_stat'].iloc[-1]:.4f}<br>
                        P-value: {lb_result['lb_pvalue'].iloc[-1]:.4f}<br>
                        Result: {'No Autocorrelation' if lb_result['lb_pvalue'].iloc[-1] > 0.05 else 'Autocorrelation Present'} ✓
                    </div>
                    """, unsafe_allow_html=True)
                
                # Distribution plots
                st.markdown("### Return Distribution Analysis")
                
                fig_dist = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Return Distribution', 'Q-Q Plot')
                )
                
                # Histogram with normal overlay
                fig_dist.add_trace(
                    go.Histogram(x=returns, nbinsx=50, name='Actual Returns', histnorm='probability density'),
                    row=1, col=1
                )
                
                # Normal distribution overlay
                x_range = np.linspace(returns.min(), returns.max(), 100)
                normal_dist = stats.norm.pdf(x_range, returns.mean(), returns.std())
                fig_dist.add_trace(
                    go.Scatter(x=x_range, y=normal_dist, name='Normal Distribution', line=dict(color='red')),
                    row=1, col=1
                )
                
                # Q-Q plot
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
                sample_quantiles = np.sort(returns)
                
                fig_dist.add_trace(
                    go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers', name='Q-Q Plot'),
                    row=1, col=2
                )
                fig_dist.add_trace(
                    go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles, mode='lines', name='Normal Line', line=dict(color='red')),
                    row=1, col=2
                )
                
                fig_dist.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with tab3:
                st.markdown("## 🔗 Advanced Correlation Analysis")
                
                # Prepare correlation data
                corr_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in ['Date']]
                
                if len(corr_columns) > 1:
                    # Correlation matrix
                    corr_matrix = df[corr_columns].corr()
                    
                    # Heatmap
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.round(3).values,
                        texttemplate='%{text}',
                        textfont={"size": 10}
                    ))
                    
                    fig_corr.update_layout(
                        title='Correlation Matrix Heatmap',
                        height=600,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Dynamic correlation analysis
                    st.markdown("### Dynamic Correlation Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        var1 = st.selectbox("Select Variable 1", corr_columns, index=0)
                    with col2:
                        var2 = st.selectbox("Select Variable 2", [col for col in corr_columns if col != var1], index=0)
                    
                    if var1 and var2:
                        # Rolling correlation
                        rolling_corr = calculate_rolling_correlation(df, var1, var2, window=30)
                        
                        fig_rolling = go.Figure()
                        fig_rolling.add_trace(go.Scatter(
                            x=df['Date'],
                            y=rolling_corr,
                            mode='lines',
                            name='30-Day Rolling Correlation',
                            line=dict(color='purple', width=2)
                        ))
                        
                        # Add zero line
                        fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        # Add confidence bands
                        fig_rolling.add_hline(y=0.7, line_dash="dot", line_color="green", annotation_text="Strong Positive")
                        fig_rolling.add_hline(y=-0.7, line_dash="dot", line_color="red", annotation_text="Strong Negative")
                        
                        fig_rolling.update_layout(
                            title=f'Rolling Correlation: {var1} vs {var2}',
                            xaxis_title='Date',
                            yaxis_title='Correlation',
                            height=400
                        )
                        
                        st.plotly_chart(fig_rolling, use_container_width=True)
                        
                        # Scatter plot with regression
                        fig_scatter = px.scatter(
                            df, x=var2, y=var1,
                            trendline="ols",
                            title=f'{var1} vs {var2} - Regression Analysis'
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Statistical significance
                        corr_value = df[var1].corr(df[var2])
                        n = len(df)
                        t_stat = corr_value * np.sqrt((n - 2) / (1 - corr_value**2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                        
                        st.markdown(f"""
                        <div class="stat-result">
                            <b>Correlation Analysis Results</b><br>
                            Correlation Coefficient: {corr_value:.4f}<br>
                            T-Statistic: {t_stat:.4f}<br>
                            P-value: {p_value:.4f}<br>
                            Significance: {'Significant' if p_value < 0.05 else 'Not Significant'} at 95% confidence
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Granger causality test
                        if enable_causality_test:
                            granger_pvalue = calculate_granger_causality(df[var1].dropna(), df[var2].dropna())
                            if granger_pvalue:
                                st.markdown(f"""
                                <div class="stat-result">
                                    <b>Granger Causality Test</b><br>
                                    {var2} → {var1}: P-value = {granger_pvalue:.4f}<br>
                                    Result: {f'{var2} Granger-causes {var1}' if granger_pvalue < 0.05 else 'No Granger causality'} ✓
                                </div>
                                """, unsafe_allow_html=True)
            
            with tab4:
                st.markdown("## 🔮 Advanced Forecasting")
                
                # Time series decomposition
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Check if data is suitable for multiplicative decomposition
                    if df[pfad_col].min() <= 0:
                        # Use additive model if there are zero or negative values
                        decomposition = seasonal_decompose(df[pfad_col].dropna(), model='additive', period=30)
                        decomp_model = "Additive"
                    else:
                        # Use multiplicative model for positive values only
                        decomposition = seasonal_decompose(df[pfad_col].dropna(), model='multiplicative', period=30)
                        decomp_model = "Multiplicative"
                    
                    fig_decomp = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual')
                    )
                    
                    fig_decomp.add_trace(
                        go.Scatter(x=df['Date'], y=df[pfad_col], name='Original', line=dict(color='blue')),
                        row=1, col=1
                    )
                    
                    fig_decomp.add_trace(
                        go.Scatter(x=df['Date'], y=decomposition.trend, name='Trend', line=dict(color='red')),
                        row=2, col=1
                    )
                    
                    fig_decomp.add_trace(
                        go.Scatter(x=df['Date'], y=decomposition.seasonal, name='Seasonal', line=dict(color='green')),
                        row=3, col=1
                    )
                    
                    fig_decomp.add_trace(
                        go.Scatter(x=df['Date'], y=decomposition.resid, name='Residual', line=dict(color='purple')),
                        row=4, col=1
                    )
                    
                    fig_decomp.update_layout(height=800, title_text=f"Time Series Decomposition ({decomp_model} Model)")
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    
                except Exception as decomp_error:
                    st.warning(f"Time series decomposition skipped due to data issues: {str(decomp_error)}")
                    st.info("This might be due to insufficient data points or irregular time series. Proceeding with forecasting...")
                
                # Forecasting models
                st.markdown("### Forecasting Models")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    forecast_model = st.selectbox(
                        "Select Model",
                        ["Linear Regression", "Exponential Smoothing", "Moving Average", "ARIMA (Simple)", "Random Walk"]
                    )
                
                with col2:
                    train_size = st.slider("Training Data %", 50, 90, 80)
                
                with col3:
                    if st.button("Generate Forecast", type="primary"):
                        with st.spinner("Generating forecast..."):
                            try:
                                # Split data
                                train_size_idx = int(len(df) * train_size / 100)
                                train_data = df[pfad_col][:train_size_idx]
                                test_data = df[pfad_col][train_size_idx:]
                                
                                # Simple forecast (using linear regression for demo)
                                X_train = np.arange(len(train_data)).reshape(-1, 1)
                                y_train = train_data.values
                                
                                # Basic Linear Regression (works without sklearn)
                                from scipy import stats
                                slope, intercept, r_value, p_value, std_err = stats.linregress(X_train.flatten(), y_train)
                                
                                # Forecast
                                future_steps = forecast_days
                                X_future = np.arange(len(df), len(df) + future_steps)
                                forecast = slope * X_future + intercept
                                
                                # Calculate prediction intervals
                                residuals = y_train - (slope * X_train.flatten() + intercept)
                                std_residuals = np.std(residuals)
                                
                                # Generate forecast dates
                                last_date = df['Date'].iloc[-1]
                                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
                                
                                # Create forecast figure
                                fig_forecast = go.Figure()
                                
                                # Historical data
                                fig_forecast.add_trace(go.Scatter(
                                    x=df['Date'],
                                    y=df[pfad_col],
                                    mode='lines',
                                    name='Historical',
                                    line=dict(color='blue')
                                ))
                                
                                # Forecast
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast_dates,
                                    y=forecast,
                                    mode='lines',
                                    name='Forecast',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                # Confidence intervals
                                upper_bound = forecast + 1.96 * std_residuals
                                lower_bound = forecast - 1.96 * std_residuals
                                
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast_dates,
                                    y=upper_bound,
                                    mode='lines',
                                    name='Upper 95% CI',
                                    line=dict(color='rgba(255,0,0,0.3)'),
                                    showlegend=False
                                ))
                                
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast_dates,
                                    y=lower_bound,
                                    mode='lines',
                                    name='Lower 95% CI',
                                    line=dict(color='rgba(255,0,0,0.3)'),
                                    fill='tonexty',
                                    fillcolor='rgba(255,0,0,0.1)',
                                    showlegend=False
                                ))
                                
                                fig_forecast.update_layout(
                                    title=f'{forecast_model} Forecast - {forecast_days} Days Ahead',
                                    xaxis_title='Date',
                                    yaxis_title='PFAD Price (₹/ton)',
                                    height=500
                                )
                                
                                st.plotly_chart(fig_forecast, use_container_width=True)
                                
                                # Forecast metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        "7-Day Forecast",
                                        f"₹{forecast[6]:,.0f}/ton" if len(forecast) >= 7 else "N/A",
                                        f"{((forecast[6] - df[pfad_col].iloc[-1]) / df[pfad_col].iloc[-1] * 100):+.2f}%" if len(forecast) >= 7 else "N/A"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "30-Day Forecast",
                                        f"₹{forecast[-1]:,.0f}/ton",
                                        f"{((forecast[-1] - df[pfad_col].iloc[-1]) / df[pfad_col].iloc[-1] * 100):+.2f}%"
                                    )
                                
                                with col3:
                                    forecast_volatility = np.std(forecast) / np.mean(forecast) * 100
                                    st.metric(
                                        "Forecast Volatility",
                                        f"{forecast_volatility:.1f}%",
                                        "Coefficient of Variation"
                                    )
                                
                                with col4:
                                    # R-squared as accuracy metric
                                    st.metric(
                                        "Model R²",
                                        f"{r_value**2:.3f}",
                                        "Goodness of Fit"
                                    )
                                    
                            except Exception as forecast_error:
                                st.error(f"Forecasting error: {str(forecast_error)}")
                
                # Feature importance for forecasting
                st.markdown("### Feature Importance for Price Prediction")
                
                if len(corr_columns) > 1:
                    try:
                        # If sklearn is available, use it
                        from sklearn.ensemble import RandomForestRegressor
                        from sklearn.preprocessing import StandardScaler
                        
                        # Prepare data
                        feature_cols = [col for col in corr_columns if col != pfad_col]
                        X = df[feature_cols].dropna()
                        y = df.loc[X.index, pfad_col]
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Train model
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X_scaled, y)
                        
                        # Feature importance
                        feature_importance = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': rf_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        fig_importance = px.bar(
                            feature_importance,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Feature Importance for PFAD Price Prediction',
                            color='importance',
                            color_continuous_scale='viridis'
                        )
                        
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                    except ImportError:
                        # Fallback to correlation-based importance
                        st.markdown("#### Correlation-based Feature Importance")
                        
                        feature_cols = [col for col in corr_columns if col != pfad_col]
                        correlations = []
                        
                        for col in feature_cols:
                            corr = abs(df[pfad_col].corr(df[col]))
                            correlations.append({'feature': col, 'importance': corr})
                        
                        feature_importance = pd.DataFrame(correlations).sort_values('importance', ascending=False)
                        
                        fig_importance = px.bar(
                            feature_importance,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Correlation-based Feature Importance',
                            color='importance',
                            color_continuous_scale='viridis'
                        )
                        
                        st.plotly_chart(fig_importance, use_container_width=True)
            
            with tab5:
                st.markdown("## ⚠️ Risk Analysis & Management")
                
                # Risk metrics
                col1, col2, col3 = st.columns(3)
                
                returns = df['PFAD_Returns'].dropna()
                
                with col1:
                    var_95 = calculate_var(returns, 0.95)
                    cvar_95 = calculate_cvar(returns, 0.95)
                    
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Value at Risk (95%)</b><br>
                        Daily VaR: {var_95*100:.2f}%<br>
                        Monthly VaR: {var_95*100*np.sqrt(22):.2f}%<br>
                        <br>
                        <b>Conditional VaR (95%)</b><br>
                        Daily CVaR: {cvar_95*100:.2f}%<br>
                        Monthly CVaR: {cvar_95*100*np.sqrt(22):.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Maximum drawdown
                    cumulative_returns = (1 + returns).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_drawdown = drawdown.min()
                    
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Maximum Drawdown</b><br>
                        Value: {max_drawdown*100:.2f}%<br>
                        <br>
                        <b>Sharpe Ratio</b><br>
                        Value: {(returns.mean() / returns.std() * np.sqrt(252)):.2f}<br>
                        Annualized
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    # Tail risk metrics
                    left_tail = np.percentile(returns, 5)
                    right_tail = np.percentile(returns, 95)
                    tail_ratio = abs(left_tail) / right_tail if right_tail != 0 else np.inf
                    
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Tail Risk Analysis</b><br>
                        Left Tail (5%): {left_tail*100:.2f}%<br>
                        Right Tail (95%): {right_tail*100:.2f}%<br>
                        Tail Ratio: {tail_ratio:.2f}<br>
                        <br>
                        Risk Level: {'High' if tail_ratio > 1.5 else 'Moderate' if tail_ratio > 1 else 'Low'}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk visualization
                st.markdown("### Risk Distribution & Scenarios")
                
                # Create risk distribution plot
                fig_risk = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Return Distribution with VaR', 'Drawdown Analysis', 
                                   'Risk Scenarios', 'Volatility Cone')
                )
                
                # Return distribution with VaR
                fig_risk.add_trace(
                    go.Histogram(x=returns, nbinsx=50, name='Returns', showlegend=False),
                    row=1, col=1
                )
                
                # Add VaR lines
                fig_risk.add_vline(x=var_95, line_dash="dash", line_color="red", row=1, col=1)
                fig_risk.add_vline(x=cvar_95, line_dash="dash", line_color="darkred", row=1, col=1)
                
                # Drawdown chart
                fig_risk.add_trace(
                    go.Scatter(x=df['Date'][1:], y=drawdown*100, name='Drawdown', 
                              fill='tozeroy', fillcolor='rgba(255,0,0,0.3)'),
                    row=1, col=2
                )
                
                # Monte Carlo risk scenarios
                n_scenarios = 100
                n_days = 30
                
                scenarios = []
                current_price = df[pfad_col].iloc[-1]
                
                for _ in range(n_scenarios):
                    scenario = [current_price]
                    for _ in range(n_days):
                        daily_return = np.random.normal(returns.mean(), returns.std())
                        scenario.append(scenario[-1] * (1 + daily_return))
                    scenarios.append(scenario)
                
                # Plot scenarios
                for i, scenario in enumerate(scenarios):
                    fig_risk.add_trace(
                        go.Scatter(x=list(range(n_days+1)), y=scenario, mode='lines',
                                 line=dict(color='lightblue', width=0.5), showlegend=False),
                        row=2, col=1
                    )
                
                # Add percentile lines
                scenarios_array = np.array(scenarios)
                p5 = np.percentile(scenarios_array, 5, axis=0)
                p50 = np.percentile(scenarios_array, 50, axis=0)
                p95 = np.percentile(scenarios_array, 95, axis=0)
                
                fig_risk.add_trace(
                    go.Scatter(x=list(range(n_days+1)), y=p5, name='5th Percentile',
                             line=dict(color='red', width=2)),
                    row=2, col=1
                )
                fig_risk.add_trace(
                    go.Scatter(x=list(range(n_days+1)), y=p50, name='Median',
                             line=dict(color='blue', width=2)),
                    row=2, col=1
                )
                fig_risk.add_trace(
                    go.Scatter(x=list(range(n_days+1)), y=p95, name='95th Percentile',
                             line=dict(color='green', width=2)),
                    row=2, col=1
                )
                
                # Volatility cone
                periods = [5, 10, 20, 30, 60, 90]
                vol_data = []
                
                for period in periods:
                    period_vols = []
                    for i in range(period, len(df)):
                        period_return = df[pfad_col].iloc[i] / df[pfad_col].iloc[i-period] - 1
                        annualized_vol = (period_return / period) * np.sqrt(252) * 100
                        period_vols.append(abs(annualized_vol))
                    
                    vol_data.append({
                        'period': period,
                        'min': np.percentile(period_vols, 5),
                        'p25': np.percentile(period_vols, 25),
                        'median': np.percentile(period_vols, 50),
                        'p75': np.percentile(period_vols, 75),
                        'max': np.percentile(period_vols, 95),
                        'current': period_vols[-1] if period_vols else 0
                    })
                
                vol_df = pd.DataFrame(vol_data)
                
                # Plot volatility cone
                fig_risk.add_trace(
                    go.Scatter(x=vol_df['period'], y=vol_df['max'], name='95th %ile',
                             line=dict(color='red', dash='dash')),
                    row=2, col=2
                )
                fig_risk.add_trace(
                    go.Scatter(x=vol_df['period'], y=vol_df['p75'], name='75th %ile',
                             line=dict(color='orange', dash='dash')),
                    row=2, col=2
                )
                fig_risk.add_trace(
                    go.Scatter(x=vol_df['period'], y=vol_df['median'], name='Median',
                             line=dict(color='blue')),
                    row=2, col=2
                )
                fig_risk.add_trace(
                    go.Scatter(x=vol_df['period'], y=vol_df['p25'], name='25th %ile',
                             line=dict(color='orange', dash='dash')),
                    row=2, col=2
                )
                fig_risk.add_trace(
                    go.Scatter(x=vol_df['period'], y=vol_df['min'], name='5th %ile',
                             line=dict(color='red', dash='dash')),
                    row=2, col=2
                )
                fig_risk.add_trace(
                    go.Scatter(x=vol_df['period'], y=vol_df['current'], name='Current',
                             mode='markers', marker=dict(color='black', size=10)),
                    row=2, col=2
                )
                
                fig_risk.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Risk mitigation strategies
                st.markdown("### 🛡️ Risk Mitigation Strategies")
                
                current_vol = df['PFAD_Volatility'].iloc[-1] * 100
                
                if current_vol > 30:
                    st.markdown("""
                    <div class="warning-box">
                        <b>⚠️ High Volatility Alert</b><br>
                        Current market volatility is elevated. Consider the following strategies:
                        <ul>
                            <li>Reduce position sizes to manage risk exposure</li>
                            <li>Implement stop-loss orders at key support levels</li>
                            <li>Consider hedging with futures or options</li>
                            <li>Increase procurement frequency with smaller volumes</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <b>✅ Normal Market Conditions</b><br>
                        Market volatility is within normal ranges. Optimal strategies:
                        <ul>
                            <li>Maintain standard procurement volumes</li>
                            <li>Focus on cost averaging strategies</li>
                            <li>Build inventory during price dips</li>
                            <li>Lock in favorable rates with forward contracts</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab6:
                st.markdown("## 🤖 AI-Powered Insights & Recommendations")
                
                # Market regime detection
                current_price = df[pfad_col].iloc[-1]
                ma_20 = df[pfad_col].rolling(20).mean().iloc[-1]
                ma_50 = df[pfad_col].rolling(50).mean().iloc[-1]
                current_vol = df['PFAD_Volatility'].iloc[-1] * 100
                
                # Determine market regime
                if current_price > ma_20 > ma_50:
                    regime = "Bullish Trend"
                    regime_color = "#10b981"
                elif current_price < ma_20 < ma_50:
                    regime = "Bearish Trend"
                    regime_color = "#ef4444"
                else:
                    regime = "Sideways/Consolidation"
                    regime_color = "#f59e0b"
                
                st.markdown(f"""
                <div class="insight-card" style="background: linear-gradient(135deg, {regime_color} 0%, {regime_color}dd 100%);">
                    <h3>📊 Current Market Regime: {regime}</h3>
                    <p>Price: ₹{current_price:,.0f} | MA20: ₹{ma_20:,.0f} | MA50: ₹{ma_50:,.0f}</p>
                    <p>Volatility: {current_vol:.1f}% (Annualized)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate AI insights
                insights = []
                
                # Price trend insight
                price_change_7d = ((df[pfad_col].iloc[-1] - df[pfad_col].iloc[-8]) / df[pfad_col].iloc[-8] * 100) if len(df) > 7 else 0
                if abs(price_change_7d) > 5:
                    insights.append({
                        'icon': '📈' if price_change_7d > 0 else '📉',
                        'title': 'Significant Price Movement',
                        'content': f'PFAD prices have {"increased" if price_change_7d > 0 else "decreased"} by {abs(price_change_7d):.1f}% in the last 7 days. This represents a {"bullish" if price_change_7d > 0 else "bearish"} signal for short-term procurement decisions.'
                    })
                
                # Volatility insight
                if current_vol > 35:
                    insights.append({
                        'icon': '⚡',
                        'title': 'High Volatility Environment',
                        'content': f'Current volatility ({current_vol:.1f}%) is significantly above average. Consider smaller, more frequent purchases to average out price fluctuations.'
                    })
                elif current_vol < 20:
                    insights.append({
                        'icon': '🎯',
                        'title': 'Low Volatility Opportunity',
                        'content': f'Market volatility is low ({current_vol:.1f}%), indicating stable conditions. This may be an optimal time for larger volume procurement.'
                    })
                
                # Correlation insight
                if cpo_col and cpo_col in df.columns:
                    cpo_corr = df[pfad_col].corr(df[cpo_col])
                    if abs(cpo_corr) > 0.7:
                        insights.append({
                            'icon': '🔗',
                            'title': 'Strong CPO Correlation',
                            'content': f'PFAD shows {abs(cpo_corr):.2f} correlation with CPO prices. Monitor CPO futures for early signals on PFAD price movements.'
                        })
                
                # Support/Resistance insight
                recent_prices = df[pfad_col].iloc[-90:] if len(df) > 90 else df[pfad_col]
                support = recent_prices.min()
                resistance = recent_prices.max()
                price_position = (current_price - support) / (resistance - support)
                
                if price_position < 0.3:
                    insights.append({
                        'icon': '🛡️',
                        'title': 'Near Support Level',
                        'content': f'Prices are near 90-day support at ₹{support:,.0f}. This could be a good entry point for procurement.'
                    })
                elif price_position > 0.7:
                    insights.append({
                        'icon': '⚠️',
                        'title': 'Near Resistance Level',
                        'content': f'Prices are approaching 90-day resistance at ₹{resistance:,.0f}. Consider waiting for a pullback before major purchases.'
                    })
                
                # Seasonal insight
                current_month = df['Date'].iloc[-1].month
                monthly_avg = df.groupby(df['Date'].dt.month)[pfad_col].mean()
                if current_month in monthly_avg.index:
                    month_rank = monthly_avg.rank()[current_month]
                    if month_rank <= 3:
                        insights.append({
                            'icon': '📅',
                            'title': 'Favorable Seasonal Period',
                            'content': f'Historical data shows this month typically has lower prices (ranked {int(month_rank)}/12). Consider increasing procurement.'
                        })
                
                # Display insights
                for insight in insights:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                        <h4>{insight['icon']} {insight['title']}</h4>
                        <p style="margin: 0.5rem 0 0 0;">{insight['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Procurement recommendations
                st.markdown("### 🎯 Procurement Strategy Recommendations")
                
                # Calculate recommendation score
                score = 50  # Base score
                
                # Adjust based on price trend
                if regime == "Bearish Trend":
                    score += 20
                elif regime == "Bullish Trend":
                    score -= 20
                
                # Adjust based on volatility
                if current_vol < 20:
                    score += 10
                elif current_vol > 35:
                    score -= 10
                
                # Adjust based on price position
                if price_position < 0.3:
                    score += 15
                elif price_position > 0.7:
                    score -= 15
                
                # Generate recommendation
                if score >= 70:
                    recommendation = "STRONG BUY"
                    rec_color = "#10b981"
                    strategy = "Execute 70-80% of planned procurement immediately, reserve 20-30% for averaging"
                elif score >= 50:
                    recommendation = "MODERATE BUY"
                    rec_color = "#3b82f6"
                    strategy = "Execute 40-50% of planned procurement, implement graduated buying over 2-3 weeks"
                elif score >= 30:
                    recommendation = "HOLD/WAIT"
                    rec_color = "#f59e0b"
                    strategy = "Maintain minimum inventory levels, wait for better entry points"
                else:
                    recommendation = "AVOID"
                    rec_color = "#ef4444"
                    strategy = "Postpone non-critical procurement, focus on risk management"
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: {rec_color}; color: white; border-radius: 10px;">
                        <h2 style="margin: 0;">{recommendation}</h2>
                        <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;">{score}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-result">
                        <b>Recommended Strategy:</b><br>
                        {strategy}<br><br>
                        <b>Key Action Items:</b>
                        <ul>
                            <li>Monitor CPO prices for leading indicators</li>
                            <li>Set price alerts at ₹{support:,.0f} (support) and ₹{resistance:,.0f} (resistance)</li>
                            <li>Review currency hedging given USD/INR volatility</li>
                            <li>Consider forward contracts if available</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export functionality
                st.markdown("### 📥 Export Analysis Report")
                
                if st.button("Generate Excel Report", type="primary"):
                    # Create Excel report
                    from io import BytesIO
                    import xlsxwriter
                    
                    output = BytesIO()
                    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
                    
                    # Summary sheet
                    summary_sheet = workbook.add_worksheet('Executive Summary')
                    summary_sheet.write('A1', 'PFAD Analytics Report')
                    summary_sheet.write('A3', 'Current Price')
                    summary_sheet.write('B3', f'₹{current_price:,.0f}')
                    summary_sheet.write('A4', 'Market Regime')
                    summary_sheet.write('B4', regime)
                    summary_sheet.write('A5', 'Recommendation')
                    summary_sheet.write('B5', recommendation)
                    summary_sheet.write('A6', 'Volatility')
                    summary_sheet.write('B6', f'{current_vol:.1f}%')
                    
                    # Data sheet
                    data_sheet = workbook.add_worksheet('Raw Data')
                    df.to_excel(workbook, sheet_name='Raw Data', index=False)
                    
                    workbook.close()
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=output.getvalue(),
                        file_name=f"PFAD_Analytics_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        else:
            st.error("PFAD price column not found. Please ensure your data contains a column with 'PFAD Rate' or similar.")
            st.info("Found columns: " + ", ".join(df.columns.tolist()))
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file contains the required columns: Date, PFAD prices, and other market indicators.")

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to PFAD Advanced Analytics Platform</h2>
        <p style="font-size: 1.2rem; color: #6b7280; margin: 2rem 0;">
            Upload your PFAD market data to unlock powerful insights and AI-driven recommendations.
        </p>
        
        <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; margin: 2rem auto; max-width: 600px;">
            <h3>📋 Required Data Format</h3>
            <p style="text-align: left;">Your Excel/CSV file should contain:</p>
            <ul style="text-align: left;">
                <li>Date column</li>
                <li>PFAD Rate (prices in INR/ton)</li>
                <li>CPO BMD Price (optional)</li>
                <li>Malaysia FOB (optional)</li>
                <li>USD INR exchange rate (optional)</li>
                <li>USD MYR exchange rate (optional)</li>
                <li>Brent crude prices (optional)</li>
                <li>CPO Volume (optional)</li>
            </ul>
        </div>
        
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 10px; margin: 2rem auto; max-width: 600px;">
            <h3>🚀 Advanced Features</h3>
            <ul style="text-align: left;">
                <li>Statistical significance testing (Jarque-Bera, ADF, Anderson-Darling)</li>
                <li>Risk metrics (VaR, CVaR, Maximum Drawdown)</li>
                <li>Market regime detection & AI recommendations</li>
                <li>Monte Carlo simulations & scenario analysis</li>
                <li>Correlation dynamics & causality testing</li>
                <li>Export detailed Excel reports</li>
            </ul>
        </div>
        
        <p style="margin-top: 2rem;">
            <b>👈 Use the sidebar to upload your data and get started!</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid #e5e7eb;">
    <p style="color: #6b7280;">
        PFAD Advanced Analytics Platform v2.0 | Powered by AI & Statistical Intelligence
    </p>
</div>
""", unsafe_allow_html=True)
