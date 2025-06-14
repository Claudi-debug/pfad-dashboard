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
    <p style="font-size: 1.2em; margin-bottom: 0;">AI-Powered Statistical Analysis for Strategic Decision Making</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Advanced Statistics • Professional Analytics • Strategic Intelligence</p>
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

show_p_values = st.sidebar.checkbox("Show Statistical Significance", value=True)
show_confidence_intervals = st.sidebar.checkbox("Show Confidence Intervals", value=True)
chart_height = st.sidebar.slider("Chart Height", 400, 800, 500)

# Time series settings
st.sidebar.header("📈 Time Series Settings")
forecast_periods = st.sidebar.slider("Forecast Periods", 1, 24, 6, help="Number of periods to forecast ahead")
trend_window = st.sidebar.slider("Trend Analysis Window", 6, 36, 12, help="Window size for trend analysis")

# Advanced statistical settings
with st.sidebar.expander("🔬 Advanced Statistical Options"):
    normality_test = st.checkbox("Perform Normality Tests", value=False)
    outlier_detection = st.checkbox("Detect Statistical Outliers", value=True)
    bootstrap_samples = st.slider("Bootstrap Samples", 100, 2000, 1000)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Enable advanced statistical options for deeper analysis")

def calculate_correlation_stats(x, y, confidence_level=0.95):
    """Calculate comprehensive correlation statistics"""
    try:
        # Remove NaN values
        data = pd.concat([x, y], axis=1).dropna()
        if len(data) < 3:
            return None
        
        x_clean, y_clean = data.iloc[:, 0], data.iloc[:, 1]
        n = len(x_clean)
        
        # Basic correlation and p-value
        corr, p_val = stats.pearsonr(x_clean, y_clean)
        
        # Confidence interval for correlation
        alpha = 1 - confidence_level
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Fisher z-transformation
        z_r = 0.5 * np.log((1 + corr) / (1 - corr))
        se_z = 1 / np.sqrt(n - 3)
        
        z_lower = z_r - z_crit * se_z
        z_upper = z_r + z_crit * se_z
        
        # Transform back to correlation scale
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        # Effect size interpretation
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
        
        # Statistical power (approximate)
        power = stats.norm.cdf(z_crit - abs(z_r * np.sqrt(n - 3))) + stats.norm.cdf(-z_crit - abs(z_r * np.sqrt(n - 3)))
        power = 1 - power
        
        return {
            'correlation': corr,
            'p_value': p_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'sample_size': n,
            'effect_size': effect_size,
            'statistical_power': power,
            'degrees_freedom': n - 2
        }
    except:
        return None

def perform_normality_test(data, alpha=0.05):
    """Perform Shapiro-Wilk normality test"""
    try:
        if len(data) < 3:
            return None
        
        # Use sample if data is too large
        if len(data) > 5000:
            sample_data = np.random.choice(data.dropna(), 5000, replace=False)
        else:
            sample_data = data.dropna()
        
        stat, p_val = stats.shapiro(sample_data)
        
        return {
            'statistic': stat,
            'p_value': p_val,
            'is_normal': p_val > alpha,
            'interpretation': 'Normal distribution' if p_val > alpha else 'Non-normal distribution'
        }
    except:
        return None

def detect_outliers(data, method='iqr'):
    """Detect statistical outliers using IQR or Z-score method"""
    try:
        data_clean = data.dropna()
        if len(data_clean) < 4:
            return None
        
        if method == 'iqr':
            Q1 = data_clean.quantile(0.25)
            Q3 = data_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_clean))
            outliers = data_clean[z_scores > 3]
        
        outlier_percentage = (len(outliers) / len(data_clean)) * 100
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': outlier_percentage,
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.tolist(),
            'method': method
        }
    except:
        return None

def bootstrap_correlation(x, y, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval for correlation"""
    try:
        data = pd.concat([x, y], axis=1).dropna()
        if len(data) < 10:
            return None
        
        bootstrap_corrs = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            sample = data.iloc[indices]
            
            # Calculate correlation
            if len(sample) > 2:
                corr, _ = stats.pearsonr(sample.iloc[:, 0], sample.iloc[:, 1])
                bootstrap_corrs.append(corr)
        
        if len(bootstrap_corrs) == 0:
            return None
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_corrs, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_corrs, (1 - alpha/2) * 100)
        
        return {
            'bootstrap_correlations': bootstrap_corrs,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_correlation': np.mean(bootstrap_corrs),
            'std_correlation': np.std(bootstrap_corrs)
        }
    except:
        return None

def analyze_time_series_trend(data, variable, window_size=12):
    """Analyze time series trends with statistical significance"""
    try:
        if 'Date' not in data.columns or variable not in data.columns:
            return None
        
        # Clean and sort data
        ts_data = data[['Date', variable]].dropna().sort_values('Date')
        if len(ts_data) < window_size:
            return None
        
        # Calculate trend using linear regression
        x_numeric = pd.to_numeric(ts_data['Date'])
        y_values = ts_data[variable]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_values)
        
        # Trend direction and strength
        trend_direction = "Upward" if slope > 0 else "Downward" if slope < 0 else "Flat"
        trend_strength = "Strong" if abs(r_value) > 0.7 else "Moderate" if abs(r_value) > 0.4 else "Weak"
        
        # Calculate percentage change
        first_val = y_values.iloc[0]
        last_val = y_values.iloc[-1]
        total_change = ((last_val / first_val) - 1) * 100 if first_val != 0 else 0
        
        # Rolling statistics
        rolling_mean = ts_data[variable].rolling(window=window_size).mean()
        rolling_std = ts_data[variable].rolling(window=window_size).std()
        volatility = (rolling_std / rolling_mean).mean() * 100
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'total_change_pct': total_change,
            'volatility': volatility,
            'data_points': len(ts_data),
            'is_significant': p_value < 0.05
        }
    except:
        return None

def simple_forecast(data, variable, periods=6):
    """Simple linear trend forecasting"""
    try:
        if 'Date' not in data.columns or variable not in data.columns:
            return None
        
        # Clean and sort data
        ts_data = data[['Date', variable]].dropna().sort_values('Date')
        if len(ts_data) < 10:
            return None
        
        # Use recent data for forecasting
        recent_data = ts_data.tail(min(24, len(ts_data)))
        
        # Linear regression on recent data
        x_numeric = pd.to_numeric(recent_data['Date'])
        y_values = recent_data[variable]
        
        slope, intercept, _, _, _ = stats.linregress(x_numeric, y_values)
        
        # Generate forecast dates
        last_date = ts_data['Date'].max()
        forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
        forecast_x = pd.to_numeric(pd.Series(forecast_dates))
        
        # Calculate forecasts
        forecast_values = slope * forecast_x + intercept
        
        # Calculate confidence intervals (simple approach)
        residuals = y_values - (slope * x_numeric + intercept)
        mse = np.mean(residuals**2)
        forecast_std = np.sqrt(mse)
        
        forecast_upper = forecast_values + 1.96 * forecast_std
        forecast_lower = forecast_values - 1.96 * forecast_std
        
        return {
            'forecast_dates': forecast_dates,
            'forecast_values': forecast_values,
            'forecast_upper': forecast_upper,
            'forecast_lower': forecast_lower,
            'forecast_std': forecast_std
        }
    except:
        return None

def generate_ai_market_insights(df, pfad_col, numeric_cols):
    """Generate comprehensive AI-powered market insights"""
    insights = {
        'market_efficiency': None,
        'volatility_assessment': None,
        'correlation_insights': None,
        'risk_factors': [],
        'opportunities': [],
        'strategic_recommendations': [],
        'market_regime': None,
        'predictability_score': None
    }
    
    try:
        if pfad_col and len(numeric_cols) > 1:
            # Market efficiency analysis
            corr_matrix = df[numeric_cols].corr()
            pfad_corr = corr_matrix[pfad_col].drop(pfad_col)
            
            avg_abs_corr = pfad_corr.abs().mean()
            strong_correlations = (pfad_corr.abs() > 0.7).sum()
            
            if avg_abs_corr > 0.6:
                insights['market_efficiency'] = {
                    'level': 'High',
                    'score': avg_abs_corr,
                    'description': 'Market shows high interconnectedness with predictable price relationships',
                    'implication': 'Systematic procurement strategies recommended'
                }
            elif avg_abs_corr > 0.3:
                insights['market_efficiency'] = {
                    'level': 'Moderate',
                    'score': avg_abs_corr,
                    'description': 'Market shows moderate efficiency with some predictable patterns',
                    'implication': 'Mixed strategy of systematic and opportunistic procurement'
                }
            else:
                insights['market_efficiency'] = {
                    'level': 'Low',
                    'score': avg_abs_corr,
                    'description': 'Market shows low efficiency with potential arbitrage opportunities',
                    'implication': 'Focus on tactical, opportunistic procurement strategies'
                }
            
            # Volatility assessment
            if pfad_col in df.columns:
                pfad_data = df[pfad_col].dropna()
                if len(pfad_data) > 1:
                    volatility = pfad_data.std() / pfad_data.mean() * 100 if pfad_data.mean() != 0 else 0
                    price_range = (pfad_data.max() - pfad_data.min()) / pfad_data.mean() * 100 if pfad_data.mean() != 0 else 0
                    
                    if volatility > 20:
                        risk_level = 'High'
                        risk_desc = 'High price volatility requires active risk management'
                    elif volatility > 10:
                        risk_level = 'Moderate'
                        risk_desc = 'Moderate volatility allows for strategic timing'
                    else:
                        risk_level = 'Low'
                        risk_desc = 'Low volatility enables predictable procurement planning'
                    
                    insights['volatility_assessment'] = {
                        'level': risk_level,
                        'volatility_pct': volatility,
                        'price_range_pct': price_range,
                        'description': risk_desc
                    }
            
            # Correlation insights
            strongest_positive = pfad_corr[pfad_corr > 0].idxmax() if len(pfad_corr[pfad_corr > 0]) > 0 else None
            strongest_negative = pfad_corr[pfad_corr < 0].idxmin() if len(pfad_corr[pfad_corr < 0]) > 0 else None
            
            insights['correlation_insights'] = {
                'strongest_positive': {
                    'variable': strongest_positive,
                    'correlation': pfad_corr[strongest_positive] if strongest_positive else None
                },
                'strongest_negative': {
                    'variable': strongest_negative,
                    'correlation': pfad_corr[strongest_negative] if strongest_negative else None
                },
                'total_significant': strong_correlations
            }
            
            # Predictability score
            predictability_factors = [
                avg_abs_corr * 0.4,  # Correlation strength
                min(1.0, strong_correlations / 3) * 0.3,  # Number of strong correlations
                max(0, 1 - volatility / 30) * 0.3  # Volatility (inverted)
            ]
            insights['predictability_score'] = sum(predictability_factors)
            
        return insights
        
    except Exception as e:
        return insights

def generate_procurement_recommendations(df, pfad_col, insights, forecast_data=None):
    """Generate AI-powered procurement recommendations"""
    recommendations = {
        'immediate_actions': [],
        'strategic_initiatives': [],
        'risk_management': [],
        'timing_recommendations': [],
        'supplier_strategy': [],
        'inventory_optimization': []
    }
    
    try:
        # Immediate actions based on current data
        if pfad_col in df.columns:
            recent_data = df[pfad_col].dropna().tail(5)
            if len(recent_data) > 1:
                recent_trend = (recent_data.iloc[-1] / recent_data.iloc[0] - 1) * 100
                
                if recent_trend > 5:
                    recommendations['immediate_actions'].append({
                        'priority': 'High',
                        'action': 'Accelerate Procurement',
                        'reason': f'Recent {recent_trend:.1f}% price increase detected',
                        'timeline': 'Next 1-2 weeks'
                    })
                elif recent_trend < -5:
                    recommendations['immediate_actions'].append({
                        'priority': 'Medium',
                        'action': 'Delay Non-Critical Purchases',
                        'reason': f'Recent {abs(recent_trend):.1f}% price decrease suggests further drops',
                        'timeline': 'Next 2-4 weeks'
                    })
        
        # Strategic initiatives based on market efficiency
        if insights.get('market_efficiency'):
            efficiency = insights['market_efficiency']
            
            if efficiency['level'] == 'High':
                recommendations['strategic_initiatives'].append({
                    'initiative': 'Systematic Procurement Program',
                    'description': 'Implement rule-based procurement using statistical indicators',
                    'expected_benefit': '5-15% cost reduction through timing optimization',
                    'implementation_time': '2-3 months'
                })
            elif efficiency['level'] == 'Low':
                recommendations['strategic_initiatives'].append({
                    'initiative': 'Opportunistic Procurement Strategy',
                    'description': 'Focus on market timing and relationship-based negotiations',
                    'expected_benefit': '3-8% cost savings through market inefficiency exploitation',
                    'implementation_time': '1-2 months'
                })
        
        # Timing recommendations based on correlations
        if insights.get('correlation_insights'):
            corr_data = insights['correlation_insights']
            
            if corr_data['strongest_positive']['variable']:
                var_name = corr_data['strongest_positive']['variable']
                correlation = corr_data['strongest_positive']['correlation']
                
                recommendations['timing_recommendations'].append({
                    'indicator': var_name,
                    'relationship': f'Strong positive correlation ({correlation:.3f})',
                    'timing_rule': f'Purchase when {var_name} is trending downward',
                    'confidence': 'High' if abs(correlation) > 0.7 else 'Medium'
                })
        
        return recommendations
        
    except Exception as e:
        return recommendations

def calculate_market_regime(df, pfad_col, window=12):
    """Determine current market regime (trending, ranging, volatile)"""
    try:
        if pfad_col not in df.columns:
            return None
        
        data = df[pfad_col].dropna()
        if len(data) < window * 2:
            return None
        
        recent_data = data.tail(window)
        
        # Calculate regime indicators
        trend_strength = abs(stats.linregress(range(len(recent_data)), recent_data)[2])  # R-value
        volatility = recent_data.std() / recent_data.mean() * 100 if recent_data.mean() != 0 else 0
        price_momentum = (recent_data.iloc[-1] / recent_data.iloc[0] - 1) * 100
        
        # Determine regime
        if trend_strength > 0.7:
            if price_momentum > 5:
                regime = 'Strong Uptrend'
                description = 'Prices are in a strong upward trend'
                strategy = 'Accelerate purchases, consider forward contracts'
            elif price_momentum < -5:
                regime = 'Strong Downtrend'
                description = 'Prices are in a strong downward trend'
                strategy = 'Delay purchases, negotiate better terms'
            else:
                regime = 'Sideways Trend'
                description = 'Prices are trending but with limited movement'
                strategy = 'Maintain regular procurement schedule'
        elif volatility > 15:
            regime = 'High Volatility'
            description = 'Market is experiencing high volatility'
            strategy = 'Use dollar-cost averaging, implement hedging'
        else:
            regime = 'Ranging Market'
            description = 'Prices are moving within a defined range'
            strategy = 'Buy at range lows, sell at range highs'
        
        return {
            'regime': regime,
            'description': description,
            'strategy': strategy,
            'trend_strength': trend_strength,
            'volatility': volatility,
            'momentum': price_momentum,
            'confidence': 'High' if trend_strength > 0.6 or volatility > 20 else 'Medium'
        }
        
    except Exception as e:
        return None

def generate_competitive_analysis(df, numeric_cols):
    """Generate competitive market position analysis"""
    try:
        analysis = {
            'market_position': None,
            'benchmark_comparison': [],
            'competitive_advantages': [],
            'improvement_areas': []
        }
        
        if len(numeric_cols) > 2:
            # Calculate relative performance metrics
            corr_matrix = df[numeric_cols].corr()
            avg_correlations = corr_matrix.abs().mean().sort_values(ascending=False)
            
            # Market position based on correlation strength
            top_performer = avg_correlations.index[0]
            top_score = avg_correlations.iloc[0]
            
            if top_score > 0.6:
                position = 'Market Leader'
                desc = 'Strong relationships with most market variables'
            elif top_score > 0.4:
                position = 'Market Participant'
                desc = 'Moderate integration with market dynamics'
            else:
                position = 'Market Outlier'
                desc = 'Limited correlation with market variables'
            
            analysis['market_position'] = {
                'position': position,
                'description': desc,
                'key_variable': top_performer,
                'score': top_score
            }
            
            # Benchmark comparisons
            for i, (var, score) in enumerate(avg_correlations.head(3).items()):
                analysis['benchmark_comparison'].append({
                    'rank': i + 1,
                    'variable': var,
                    'market_integration_score': score,
                    'percentile': (len(avg_correlations) - i) / len(avg_correlations) * 100
                })
        
        return analysis
        
    except Exception as e:
        return None

if uploaded_file:
    try:
        # Load data with progress
        with st.spinner("📊 Loading and processing your data..."):
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Successfully loaded {len(df):,} records from {uploaded_file.name}")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Check for date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        has_date_data = len(date_cols) > 0
        
        if has_date_data and date_cols[0] not in df.columns:
            # Try to convert the first date-like column
            try:
                df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                has_date_data = not df['Date'].isnull().all()
            except:
                has_date_data = False
        elif 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                has_date_data = not df['Date'].isnull().all()
            except:
                has_date_data = False
        
        # Find PFAD column
        pfad_col = None
        for col in numeric_cols:
            if 'PFAD' in str(col).upper():
                pfad_col = col
                break
        
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
                help="Variables available for statistical analysis"
            )
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric(
                label="📋 Data Quality",
                value=f"{100-missing_pct:.1f}%",
                help="Percentage of complete data"
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Data Overview", 
            "🌡️ Correlation Analysis", 
            "🎯 PFAD Insights", 
            "📈 Advanced Statistics",
            "🔬 Statistical Tests",
            "📈 Time Series Analysis",
            "🤖 AI Insights Engine"
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
            
            # Enhanced column information with statistical insights
            st.subheader("📋 Statistical Data Profile")
            if len(numeric_cols) > 0:
                profile_data = []
                for col in numeric_cols:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        profile_data.append({
                            'Variable': col,
                            'Count': len(col_data),
                            'Missing': df[col].isnull().sum(),
                            'Mean': col_data.mean(),
                            'Std Dev': col_data.std(),
                            'Min': col_data.min(),
                            'Max': col_data.max(),
                            'Skewness': stats.skew(col_data),
                            'Kurtosis': stats.kurtosis(col_data),
                            'CV%': (col_data.std() / col_data.mean() * 100) if col_data.mean() != 0 else 0
                        })
                
                profile_df = pd.DataFrame(profile_data)
                st.dataframe(profile_df.round(3), use_container_width=True)
                
                # Statistical insights
                st.markdown("""
                <div class="statistical-box">
                <h4>📊 Key Statistical Insights</h4>
                """, unsafe_allow_html=True)
                
                high_var_cols = profile_df[profile_df['CV%'] > 50]['Variable'].tolist()
                skewed_cols = profile_df[abs(profile_df['Skewness']) > 1]['Variable'].tolist()
                
                if high_var_cols:
                    st.write(f"**High Variability Variables** (CV > 50%): {', '.join(high_var_cols[:3])}")
                
                if skewed_cols:
                    st.write(f"**Highly Skewed Variables** (|skewness| > 1): {', '.join(skewed_cols[:3])}")
                
                st.write(f"**Average Coefficient of Variation**: {profile_df['CV%'].mean():.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.header("🌡️ Advanced Correlation Analysis")
            
            if len(numeric_cols) > 1:
                # Calculate correlations
                corr_matrix = df[numeric_cols].corr()
                
                # Enhanced correlation heatmap with statistical annotations
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
                
                fig_heatmap.update_traces(
                    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        with tab3:
            st.header("🎯 PFAD Statistical Insights")
            
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
                                'Sample Size': stats_result['sample_size'],
                                'Statistical Power': f"{stats_result['statistical_power']:.3f}"
                            })
                    
                    if detailed_stats:
                        stats_df = pd.DataFrame(detailed_stats)
                        st.dataframe(stats_df, use_container_width=True)
                    
                else:
                    st.info("Need more numeric variables for comprehensive PFAD analysis")
            
            else:
                st.warning("🔍 No PFAD column found in your data")
                st.info("💡 Make sure your PFAD column contains 'PFAD' in the name")
        
        with tab4:
            st.header("📈 Advanced Statistical Analytics")
            
            if len(numeric_cols) > 1 and pfad_col:
                # Bootstrap analysis
                st.subheader("🔄 Bootstrap Confidence Analysis")
                
                corr_matrix = df[numeric_cols].corr()
                pfad_corr = corr_matrix[pfad_col].drop(pfad_col).sort_values(key=abs, ascending=False)
                
                selected_var_bootstrap = st.selectbox(
                    "Select variable for bootstrap analysis:",
                    options=pfad_corr.abs().sort_values(ascending=False).index.tolist()
                )
                
                if selected_var_bootstrap:
                    with st.spinner("Performing bootstrap analysis..."):
                        bootstrap_result = bootstrap_correlation(
                            df[pfad_col], 
                            df[selected_var_bootstrap], 
                            bootstrap_samples, 
                            confidence_level
                        )
                    
                    if bootstrap_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Bootstrap distribution
                            fig_bootstrap = px.histogram(
                                x=bootstrap_result['bootstrap_correlations'],
                                nbins=50,
                                title=f"Bootstrap Distribution (n={bootstrap_samples})",
                                labels={'x': 'Correlation Coefficient', 'y': 'Frequency'}
                            )
                            
                            # Add confidence interval lines
                            fig_bootstrap.add_vline(
                                x=bootstrap_result['ci_lower'], 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text=f"CI Lower: {bootstrap_result['ci_lower']:.3f}"
                            )
                            fig_bootstrap.add_vline(
                                x=bootstrap_result['ci_upper'], 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text=f"CI Upper: {bootstrap_result['ci_upper']:.3f}"
                            )
                            
                            st.plotly_chart(fig_bootstrap, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"""
                            **Bootstrap Results:**
                            - **Mean Correlation**: {bootstrap_result['mean_correlation']:.3f}
                            - **Standard Error**: {bootstrap_result['std_correlation']:.3f}
                            - **{confidence_level*100:.0f}% CI Lower**: {bootstrap_result['ci_lower']:.3f}
                            - **{confidence_level*100:.0f}% CI Upper**: {bootstrap_result['ci_upper']:.3f}
                            - **Bootstrap Samples**: {len(bootstrap_result['bootstrap_correlations'])}
                            
                            **Interpretation:**
                            The bootstrap method provides a robust estimate of the correlation 
                            and its uncertainty without assumptions about the data distribution.
                            """)
            
            else:
                st.info("PFAD column and additional numeric variables needed for advanced analytics")
        
        with tab5:
            st.header("🔬 Comprehensive Statistical Tests")
            
            if len(numeric_cols) > 0:
                # Normality testing
                if normality_test:
                    st.subheader("📊 Normality Assessment")
                    
                    normality_var = st.selectbox(
                        "Select variable for normality testing:",
                        options=numeric_cols,
                        key="normality_var"
                    )
                    
                    if normality_var:
                        normality_result = perform_normality_test(df[normality_var])
                        
                        if normality_result:
                            st.markdown(f"""
                            <div class="statistical-box">
                            <h4>📊 Shapiro-Wilk Normality Test Results</h4>
                            <p><strong>Test Statistic:</strong> {normality_result['statistic']:.4f}</p>
                            <p><strong>P-Value:</strong> {normality_result['p_value']:.6f}</p>
                            <p><strong>Result:</strong> {normality_result['interpretation']}</p>
                            <p><strong>Interpretation:</strong> {'Data appears normally distributed' if normality_result['is_normal'] else 'Data significantly deviates from normal distribution'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Correlation significance testing matrix
                st.subheader("🎯 Comprehensive Correlation Significance Matrix")
                
                if len(numeric_cols) > 1:
                    # Create comprehensive significance table
                    sig_pairs = []
                    
                    for i, var1 in enumerate(numeric_cols):
                        for j, var2 in enumerate(numeric_cols):
                            if i < j:  # Avoid duplicates
                                stats_result = calculate_correlation_stats(df[var1], df[var2], confidence_level)
                                
                                if stats_result:
                                    sig_pairs.append({
                                        'Variable 1': var1,
                                        'Variable 2': var2,
                                        'Correlation': stats_result['correlation'],
                                        'P-Value': stats_result['p_value'],
                                        'CI Lower': stats_result['ci_lower'],
                                        'CI Upper': stats_result['ci_upper'],
                                        'Effect Size': stats_result['effect_size'],
                                        'Sample Size': stats_result['sample_size'],
                                        'Significant': 'Yes' if stats_result['p_value'] < 0.05 else 'No',
                                        'Power': stats_result['statistical_power']
                                    })
                    
                    if sig_pairs:
                        sig_df = pd.DataFrame(sig_pairs)
                        sig_df = sig_df.sort_values('P-Value')
                        
                        st.dataframe(sig_df.round(4), use_container_width=True)
                        
                        # Summary statistics
                        total_pairs = len(sig_pairs)
                        significant_pairs = len([p for p in sig_pairs if p['P-Value'] < 0.05])
                        strong_significant = len([p for p in sig_pairs if p['P-Value'] < 0.05 and abs(p['Correlation']) > 0.7])
                        
                        st.markdown(f"""
                        <div class="insight-box">
                        <h4>📊 Statistical Summary</h4>
                        <p><strong>Total Variable Pairs:</strong> {total_pairs}</p>
                        <p><strong>Statistically Significant:</strong> {significant_pairs} ({significant_pairs/total_pairs*100:.1f}%)</p>
                        <p><strong>Strong & Significant:</strong> {strong_significant} ({strong_significant/total_pairs*100:.1f}%)</p>
                        <p><strong>Confidence Level:</strong> {confidence_level*100:.0f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            else:
                st.info("Numeric variables needed for statistical testing")
        
        with tab6:
            st.header("📈 Professional Time Series Analysis")
            
            if has_date_data and len(numeric_cols) > 0:
                # Time series overview metrics
                st.subheader("📊 Time Series Dataset Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    date_span = (df['Date'].max() - df['Date'].min()).days
                    st.metric("Date Span (Days)", f"{date_span:,}")
                
                with col2:
                    data_frequency = len(df) / (date_span / 30) if date_span > 0 else 0
                    st.metric("Avg Data Points/Month", f"{data_frequency:.1f}")
                
                with col3:
                    missing_dates = df['Date'].isnull().sum()
                    st.metric("Missing Dates", f"{missing_dates:,}")
                
                with col4:
                    if pfad_col and pfad_col in df.columns:
                        pfad_trend_result = analyze_time_series_trend(df, pfad_col, trend_window)
                        if pfad_trend_result:
                            st.metric("PFAD Trend", pfad_trend_result['trend_direction'])
                        else:
                            st.metric("PFAD Trend", "N/A")
                    else:
                        st.metric("PFAD Trend", "No PFAD Data")
                
                # Variable selection for detailed analysis
                st.subheader("🎯 Detailed Variable Analysis")
                
                ts_variable = st.selectbox(
                    "Select Variable for Time Series Analysis:",
                    options=numeric_cols,
                    help="Choose a variable to analyze trends, patterns, and forecasts"
                )
                
                if ts_variable:
                    # Time series plot with trend analysis
                    ts_data = df[['Date', ts_variable]].dropna().sort_values('Date')
                    
                    if len(ts_data) >= 10:
                        # Main time series plot
                        fig_ts = go.Figure()
                        
                        # Add main data line
                        fig_ts.add_trace(go.Scatter(
                            x=ts_data['Date'],
                            y=ts_data[ts_variable],
                            mode='lines+markers',
                            name=ts_variable,
                            line=dict(width=2, color='blue'),
                            marker=dict(size=4)
                        ))
                        
                        # Add trend line
                        if len(ts_data) > 1:
                            x_numeric = pd.to_numeric(ts_data['Date'])
                            z = np.polyfit(x_numeric, ts_data[ts_variable], 1)
                            p = np.poly1d(z)
                            
                            fig_ts.add_trace(go.Scatter(
                                x=ts_data['Date'],
                                y=p(x_numeric),
                                mode='lines',
                                name='Linear Trend',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                        
                        fig_ts.update_layout(
                            title=f"📈 Time Series Analysis: {ts_variable}",
                            xaxis_title="Date",
                            yaxis_title=ts_variable,
                            height=chart_height,
                            hovermode='x unified',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_ts, use_container_width=True)
                        
                        # Statistical trend analysis
                        trend_result = analyze_time_series_trend(df, ts_variable, trend_window)
                        
                        if trend_result:
                            st.subheader("📊 Statistical Trend Analysis")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Trend Direction", trend_result['trend_direction'])
                            
                            with col2:
                                st.metric("Trend Strength", trend_result['trend_strength'])
                            
                            with col3:
                                st.metric("R-squared", f"{trend_result['r_squared']:.3f}")
                            
                            with col4:
                                significance = "Significant" if trend_result['is_significant'] else "Not Significant"
                                st.metric("Statistical Significance", significance)
                        
                        # Forecasting section
                        st.subheader("🔮 Price Forecasting")
                        
                        forecast_result = simple_forecast(df, ts_variable, forecast_periods)
                        
                        if forecast_result:
                            # Forecast plot
                            fig_forecast = go.Figure()
                            
                            # Historical data
                            fig_forecast.add_trace(go.Scatter(
                                x=ts_data['Date'],
                                y=ts_data[ts_variable],
                                mode='lines+markers',
                                name='Historical Data',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Forecast line
                            fig_forecast.add_trace(go.Scatter(
                                x=forecast_result['forecast_dates'],
                                y=forecast_result['forecast_values'],
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            fig_forecast.update_layout(
                                title=f"📈 {ts_variable} Forecast ({forecast_periods} periods ahead)",
                                xaxis_title="Date",
                                yaxis_title=ts_variable,
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        else:
                            st.warning("⚠️ Insufficient data for reliable forecasting")
                    
                    else:
                        st.warning(f"⚠️ Need at least 10 data points for time series analysis. Current: {len(ts_data)}")
            
            else:
                # No time series data available
                st.warning("⚠️ Time Series Analysis Not Available")
                
                if not has_date_data:
                    st.info("""
                    **Missing Date Information**
                    
                    To enable time series analysis, your dataset needs:
                    • A column with date/time information
                    • Dates should be in a recognizable format (YYYY-MM-DD, MM/DD/YYYY, etc.)
                    • Column should be named 'Date' or contain 'date' in the name
                    """)
        
        with tab7:
            st.header("🤖 AI Insights Engine")
            st.markdown("*Powered by Advanced Analytics & Machine Learning*")
            
            if len(numeric_cols) > 1:
                # Generate AI insights
                with st.spinner("🤖 AI is analyzing your data and generating insights..."):
                    ai_insights = generate_ai_market_insights(df, pfad_col, numeric_cols)
                    procurement_recs = generate_procurement_recommendations(df, pfad_col, ai_insights)
                    market_regime = calculate_market_regime(df, pfad_col, trend_window) if pfad_col else None
                    competitive_analysis = generate_competitive_analysis(df, numeric_cols)
                
                # AI Dashboard Overview
                st.subheader("🎯 AI Market Intelligence Dashboard")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if ai_insights.get('market_efficiency'):
                        efficiency = ai_insights['market_efficiency']
                        st.metric(
                            "Market Efficiency",
                            efficiency['level'],
                            f"{efficiency['score']:.3f}",
                            help="AI assessment of market predictability and efficiency"
                        )
                    else:
                        st.metric("Market Efficiency", "N/A")
                
                with col2:
                    if ai_insights.get('volatility_assessment'):
                        volatility = ai_insights['volatility_assessment']
                        st.metric(
                            "Risk Level",
                            volatility['level'],
                            f"{volatility['volatility_pct']:.1f}%",
                            help="AI-assessed risk level based on price volatility"
                        )
                    else:
                        st.metric("Risk Level", "N/A")
                
                with col3:
                    if ai_insights.get('predictability_score'):
                        pred_score = ai_insights['predictability_score']
                        pred_level = "High" if pred_score > 0.7 else "Medium" if pred_score > 0.4 else "Low"
                        st.metric(
                            "AI Predictability",
                            pred_level,
                            f"{pred_score:.3f}",
                            help="AI-calculated score for market predictability (0-1 scale)"
                        )
                    else:
                        st.metric("AI Predictability", "N/A")
                
                with col4:
                    if market_regime:
                        st.metric(
                            "Market Regime",
                            market_regime['regime'],
                            f"{market_regime['confidence']} Confidence",
                            help="AI-identified current market regime and trend direction"
                        )
                    else:
                        st.metric("Market Regime", "N/A")
                
                # AI-Powered Procurement Recommendations
                st.subheader("💡 AI-Generated Procurement Recommendations")
                
                # Immediate Actions
                if procurement_recs['immediate_actions']:
                    st.markdown("#### ⚡ Immediate Actions (Next 1-4 weeks)")
                    for action in procurement_recs['immediate_actions']:
                        priority_color = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#27ae60'}.get(action['priority'], '#34495e')
                        st.markdown(f"""
                        <div style="background: {priority_color}15; border-left: 4px solid {priority_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 5px;">
                            <strong style="color: {priority_color};">[{action['priority']} Priority]</strong> {action['action']}<br>
                            <small><strong>Reason:</strong> {action['reason']}<br>
                            <strong>Timeline:</strong> {action['timeline']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Strategic Initiatives
                if procurement_recs['strategic_initiatives']:
                    st.markdown("#### 🎯 Strategic Initiatives (2-6 months)")
                    for initiative in procurement_recs['strategic_initiatives']:
                        st.markdown(f"""
                        <div class="insight-box">
                            <h4>📈 {initiative['initiative']}</h4>
                            <p><strong>Description:</strong> {initiative['description']}</p>
                            <p><strong>Expected Benefit:</strong> {initiative['expected_benefit']}</p>
                            <p><strong>Implementation Time:</strong> {initiative['implementation_time']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Smart Timing Recommendations
                if procurement_recs['timing_recommendations']:
                    st.subheader("⏰ AI Smart Timing Recommendations")
                    
                    for timing in procurement_recs['timing_recommendations']:
                        confidence_color = {'High': '#27ae60', 'Medium': '#f39c12', 'Low': '#e74c3c'}.get(timing['confidence'], '#34495e')
                        st.markdown(f"""
                        <div style="background: {confidence_color}15; border: 2px solid {confidence_color}; padding: 1.5rem; margin: 1rem 0; border-radius: 10px;">
                            <h4 style="color: {confidence_color};">📊 Leading Indicator: {timing['indicator']}</h4>
                            <p><strong>Relationship:</strong> {timing['relationship']}</p>
                            <p><strong>AI Timing Rule:</strong> {timing['timing_rule']}</p>
                            <p><strong>Confidence Level:</strong> <span style="color: {confidence_color}; font-weight: bold;">{timing['confidence']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # AI Executive Summary
                st.subheader("📋 AI Executive Summary")
                
                st.markdown("""
                <div class="insight-box">
                <h3>🤖 AI-Generated Executive Summary</h3>
                """, unsafe_allow_html=True)
                
                # Generate executive summary
                summary_points = []
                
                if ai_insights.get('market_efficiency'):
                    efficiency = ai_insights['market_efficiency']
                    summary_points.append(f"Market efficiency is **{efficiency['level'].lower()}** ({efficiency['score']:.3f}), {efficiency['implication'].lower()}")
                
                if ai_insights.get('volatility_assessment'):
                    volatility = ai_insights['volatility_assessment']
                    summary_points.append(f"Price volatility is **{volatility['level'].lower()}** ({volatility['volatility_pct']:.1f}%), {volatility['description'].lower()}")
                
                if market_regime:
                    summary_points.append(f"Current market regime is **{market_regime['regime'].lower()}** with {market_regime['confidence'].lower()} confidence")
                
                # Display summary points
                for i, point in enumerate(summary_points, 1):
                    st.write(f"{i}. {point}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            else:
                # Insufficient data for AI analysis
                st.warning("⚠️ AI Insights Engine requires more data for comprehensive analysis")
                
                st.markdown("""
                <div class="insight-box">
                <h3>🤖 AI Engine Requirements</h3>
                <p>To unlock the full power of AI analytics, ensure your dataset contains:</p>
                <ul>
                    <li><strong>Multiple Variables:</strong> At least 3-4 numeric variables for correlation analysis</li>
                    <li><strong>PFAD Data:</strong> Primary commodity data for procurement insights</li>
                    <li><strong>Sufficient Records:</strong> Minimum 20-30 data points for reliable AI predictions</li>
                    <li><strong>Data Quality:</strong> Clean, consistent data with minimal missing values</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.write("**Troubleshooting tips:**")
        st.write("• Ensure your file is a valid Excel format (.xlsx or .xls)")
        st.write("• Check that your data contains numeric values")
        st.write("• Verify that column names don't contain special characters")
        st.write("• Make sure you have sufficient data for statistical analysis")

else:
    # Enhanced welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 20px; margin: 2rem 0;">
        <h2 style="color: #2c3e50; margin-bottom: 2rem;">📊 Advanced PFAD Statistical Analytics</h2>
        <p style="font-size: 1.3em; color: #34495e; margin-bottom: 2rem;">
            Professional-grade statistical analysis for strategic procurement decisions
        </p>
        
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin: 2rem 0;">
            <div style="flex: 1; min-width: 180px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🔬 Statistical Tests</h3>
                <p>Confidence intervals, p-values, and significance testing</p>
            </div>
            <div style="flex: 1; min-width: 180px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🔄 Bootstrap Analysis</h3>
                <p>Robust correlation estimates with bootstrap sampling</p>
            </div>
            <div style="flex: 1; min-width: 180px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">⚡ Power Analysis</h3>
                <p>Statistical power and effect size interpretation</p>
            </div>
            <div style="flex: 1; min-width: 180px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">📈 Time Series</h3>
                <p>Trend analysis, forecasting, and temporal insights</p>
            </div>
            <div style="flex: 1; min-width: 180px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🤖 AI Engine</h3>
                <p>Machine learning insights and smart recommendations</p>
            </div>
        </div>
        
        <p style="color: #7f8c8d; font-size: 1.1em;">
            Upload your Excel file to access comprehensive analytical features
        </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #7f8c8d;">
    <p><strong>PFAD Advanced Statistical Analytics</strong> | Professional Statistical Analysis Platform</p>
    <p>🔬 Statistical rigor • 📊 Professional insights • ⚡ Evidence-based decisions • 📈 Time series forecasting • 🤖 AI-powered intelligence</p>
</div>
""", unsafe_allow_html=True)
