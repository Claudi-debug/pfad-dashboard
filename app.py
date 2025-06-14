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

# Custom CSS for professional styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .correlation-strong {
        color: #28a745;
        font-weight: bold;
    }
    .correlation-moderate {
        color: #ffc107;
        font-weight: bold;
    }
    .correlation-weak {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
        📊 PFAD Procurement Analytics Dashboard
    </h1>
    <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;">
        AI-Powered Correlation Analysis for Strategic Procurement Decisions
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Dashboard Controls")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Excel File", 
    type=['xlsx', 'xls'],
    help="Upload your PFAD data file for analysis"
)

# Analysis settings
st.sidebar.subheader("📊 Analysis Settings")
chart_height = st.sidebar.slider("Chart Height", 300, 800, 500)
confidence_level = st.sidebar.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
correlation_threshold = st.sidebar.slider("Correlation Threshold", 0.0, 1.0, 0.3, 0.1)

# Time series settings
st.sidebar.subheader("📈 Time Series Settings")
forecast_periods = st.sidebar.slider("Forecast Periods", 1, 24, 6)
trend_window = st.sidebar.slider("Trend Analysis Window", 6, 36, 12)

def load_and_prepare_data(uploaded_file):
    """Load and prepare data from uploaded file"""
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)
        
        # Convert column names to strings and clean them
        df.columns = [str(col).strip() for col in df.columns]
        
        # Column mapping for standardization
        column_mapping = {
            'Date': 'Date',
            'Imported PFAD - CIF in INR / ton (IA)': 'PFAD_Rate',
            'CPO BMD Price (MYR)': 'CPO_BMD_MYR',
            'Malaysia FOB USD': 'Malaysia_FOB_USD',
            'USD/INR': 'USD_INR',
            'USD/MYR': 'USD_MYR',
            'Brent crude (USD)': 'Brent_Crude_USD',
            'CPO Volume': 'CPO_Volume'
        }
        
        # Find matching columns (case-insensitive)
        actual_mapping = {}
        for expected, standard in column_mapping.items():
            for col in df.columns:
                if expected.lower() in col.lower() or col.lower() in expected.lower():
                    actual_mapping[col] = standard
                    break
        
        # Rename columns
        df = df.rename(columns=actual_mapping)
        
        # Convert date column if it exists
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        if date_columns:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]], errors='coerce')
            if date_columns[0] != 'Date':
                df = df.rename(columns={date_columns[0]: 'Date'})
        
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        return df, True, None
        
    except Exception as e:
        return None, False, str(e)

def create_enhanced_heatmap(correlation_matrix, height=500):
    """Create an enhanced correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=correlation_matrix.round(3).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(
            title="Correlation Coefficient",
            titleside="right"
        )
    ))
    
    fig.update_layout(
        title="📊 Enhanced Correlation Matrix",
        height=height,
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'},
        font=dict(size=12)
    )
    
    return fig

def create_pfad_correlation_chart(correlation_matrix, threshold=0.3):
    """Create PFAD-specific correlation analysis"""
    if 'PFAD_Rate' not in correlation_matrix.columns:
        return None, None
    
    pfad_corr = correlation_matrix['PFAD_Rate'].drop('PFAD_Rate', errors='ignore')
    pfad_corr_sorted = pfad_corr.reindex(pfad_corr.abs().sort_values(ascending=False).index)
    
    # Color coding based on correlation strength
    colors = []
    for corr in pfad_corr_sorted:
        if abs(corr) >= 0.7:
            colors.append('#28a745')  # Strong - Green
        elif abs(corr) >= 0.4:
            colors.append('#ffc107')  # Moderate - Yellow
        else:
            colors.append('#dc3545')  # Weak - Red
    
    fig = go.Figure(data=go.Bar(
        x=pfad_corr_sorted.values,
        y=pfad_corr_sorted.index,
        orientation='h',
        marker_color=colors,
        text=[f"{val:.3f}" for val in pfad_corr_sorted.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="🎯 PFAD Correlations (Ranked by Strength)",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Variables",
        height=400
    )
    
    # Add reference lines
    fig.add_vline(x=0.7, line_dash="dash", line_color="green", annotation_text="Strong")
    fig.add_vline(x=-0.7, line_dash="dash", line_color="green")
    fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Moderate")
    fig.add_vline(x=-0.4, line_dash="dash", line_color="orange")
    
    return fig, pfad_corr_sorted

def create_scatter_plots(df, correlation_matrix, top_n=4):
    """Create scatter plots for top correlations"""
    # Get top correlations (excluding self-correlations)
    corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr_val = correlation_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                corr_pairs.append((col1, col2, abs(corr_val), corr_val))
    
    # Sort by absolute correlation and take top N
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = corr_pairs[:top_n]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{pair[0]} vs {pair[1]} (r={pair[3]:.3f})" for pair in top_pairs],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )
    
    for idx, (col1, col2, abs_corr, corr) in enumerate(top_pairs):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        if col1 in df.columns and col2 in df.columns:
            x_data = df[col1].dropna()
            y_data = df[col2].dropna()
            
            # Ensure both series have same length
            common_idx = x_data.index.intersection(y_data.index)
            x_data = x_data.loc[common_idx]
            y_data = y_data.loc[common_idx]
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    name=f"{col1} vs {col2}",
                    marker=dict(size=6, opacity=0.7),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add trend line
            if len(x_data) > 1:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=p(x_data),
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(height=600, title_text="🔍 Top Correlation Scatter Plots")
    return fig

def calculate_bootstrap_correlation(x, y, n_bootstrap=1000):
    """Calculate bootstrap confidence interval for correlation"""
    try:
        x_clean = x.dropna()
        y_clean = y.dropna()
        
        # Get common indices
        common_idx = x_clean.index.intersection(y_clean.index)
        if len(common_idx) < 3:
            return None
            
        x_common = x_clean.loc[common_idx]
        y_common = y_clean.loc[common_idx]
        
        bootstrap_corrs = []
        n_samples = len(x_common)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_boot = x_common.iloc[indices]
            y_boot = y_common.iloc[indices]
            
            corr, _ = stats.pearsonr(x_boot, y_boot)
            if not np.isnan(corr):
                bootstrap_corrs.append(corr)
        
        if len(bootstrap_corrs) == 0:
            return None
            
        bootstrap_corrs = np.array(bootstrap_corrs)
        
        return {
            'mean_correlation': np.mean(bootstrap_corrs),
            'std_error': np.std(bootstrap_corrs),
            'confidence_interval': np.percentile(bootstrap_corrs, [2.5, 97.5]),
            'bootstrap_correlations': bootstrap_corrs
        }
    except:
        return None

def create_significance_matrix(df, confidence_level=0.95):
    """Create statistical significance matrix"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_vars = len(numeric_cols)
    
    p_matrix = np.ones((n_vars, n_vars))
    corr_matrix = np.zeros((n_vars, n_vars))
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i != j:
                x = df[col1].dropna()
                y = df[col2].dropna()
                
                # Get common indices
                common_idx = x.index.intersection(y.index)
                if len(common_idx) >= 3:
                    x_common = x.loc[common_idx]
                    y_common = y.loc[common_idx]
                    
                    try:
                        corr, p_val = stats.pearsonr(x_common, y_common)
                        corr_matrix[i, j] = corr
                        p_matrix[i, j] = p_val
                    except:
                        pass
    
    # Create significance labels
    alpha = 1 - confidence_level
    sig_matrix = np.where(p_matrix < alpha, '***', 
                 np.where(p_matrix < alpha*2, '**',
                 np.where(p_matrix < alpha*5, '*', 'ns')))
    
    return pd.DataFrame(p_matrix, index=numeric_cols, columns=numeric_cols), \
           pd.DataFrame(sig_matrix, index=numeric_cols, columns=numeric_cols)

def create_rolling_correlation(df, target_col='PFAD_Rate', window=12):
    """Create rolling correlation analysis"""
    if target_col not in df.columns or 'Date' not in df.columns:
        return None
    
    # Ensure data is sorted by date
    df_sorted = df.sort_values('Date')
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    rolling_corrs = {}
    
    for col in numeric_cols:
        rolling_corr = df_sorted[target_col].rolling(window=window).corr(df_sorted[col])
        rolling_corrs[col] = rolling_corr
    
    # Create plot
    fig = go.Figure()
    
    for col, corr_series in rolling_corrs.items():
        fig.add_trace(go.Scatter(
            x=df_sorted['Date'],
            y=corr_series,
            mode='lines',
            name=col,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f"📈 Rolling Correlation with {target_col} (Window: {window} periods)",
        xaxis_title="Date",
        yaxis_title="Correlation Coefficient",
        height=500,
        hovermode='x unified'
    )
    
    # Add reference lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Strong Positive")
    fig.add_hline(y=-0.7, line_dash="dash", line_color="green", annotation_text="Strong Negative")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No Correlation")
    
    return fig

def generate_business_insights(correlation_matrix, df):
    """Generate AI-powered business insights"""
    insights = []
    
    if 'PFAD_Rate' in correlation_matrix.columns:
        pfad_corr = correlation_matrix['PFAD_Rate'].drop('PFAD_Rate', errors='ignore')
        
        # Strongest positive correlation
        strongest_pos = pfad_corr[pfad_corr > 0].idxmax() if any(pfad_corr > 0) else None
        if strongest_pos:
            corr_val = pfad_corr[strongest_pos]
            insights.append({
                'type': 'positive',
                'title': f"Strongest Positive Driver: {strongest_pos}",
                'description': f"PFAD rates show a strong positive correlation ({corr_val:.3f}) with {strongest_pos}. When {strongest_pos} increases, PFAD rates tend to increase as well.",
                'recommendation': f"Monitor {strongest_pos} closely as an early indicator for PFAD price movements. Consider hedging strategies when {strongest_pos} shows upward trends."
            })
        
        # Strongest negative correlation
        strongest_neg = pfad_corr[pfad_corr < 0].idxmin() if any(pfad_corr < 0) else None
        if strongest_neg:
            corr_val = pfad_corr[strongest_neg]
            insights.append({
                'type': 'negative',
                'title': f"Strongest Negative Driver: {strongest_neg}",
                'description': f"PFAD rates show a strong negative correlation ({corr_val:.3f}) with {strongest_neg}. When {strongest_neg} increases, PFAD rates tend to decrease.",
                'recommendation': f"Use {strongest_neg} as a counter-indicator. When {strongest_neg} is high, it may be a good time to procure PFAD at lower rates."
            })
    
    # Market efficiency insight
    avg_correlation = correlation_matrix.abs().mean().mean()
    if avg_correlation > 0.6:
        insights.append({
            'type': 'efficiency',
            'title': "High Market Efficiency",
            'description': f"Average correlation strength is {avg_correlation:.3f}, indicating a highly interconnected market.",
            'recommendation': "Market shows high efficiency. Price movements are predictable based on other commodity trends. Ideal for systematic procurement strategies."
        })
    else:
        insights.append({
            'type': 'efficiency',
            'title': "Market Opportunities",
            'description': f"Average correlation strength is {avg_correlation:.3f}, indicating potential arbitrage opportunities.",
            'recommendation': "Lower correlations suggest market inefficiencies. Look for tactical procurement opportunities when relationships temporarily diverge."
        })
    
    return insights

def create_time_series_analysis(df):
    """Create comprehensive time series analysis"""
    if 'Date' not in df.columns:
        return None, "No date column found"
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        return None, "No numeric columns found"
    
    # Sort by date
    df_ts = df.sort_values('Date').copy()
    
    # Remove rows with NaN dates
    df_ts = df_ts.dropna(subset=['Date'])
    
    if len(df_ts) < 10:
        return None, "Insufficient data for time series analysis"
    
    return df_ts, None

# Main application
if uploaded_file is not None:
    # Load data
    with st.spinner("📁 Loading and preparing data..."):
        df, success, error = load_and_prepare_data(uploaded_file)
    
    if not success:
        st.error(f"❌ Error loading data: {error}")
        st.stop()
    
    # Display basic info
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>📊 Total Records</h3>
            <h2>{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        date_range = "N/A"
        if 'Date' in df.columns:
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            if pd.notnull(min_date) and pd.notnull(max_date):
                date_range = f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}"
        
        st.markdown(f"""
        <div class="metric-container">
            <h3>📅 Date Range</h3>
            <h2>{date_range}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        strongest_corr = 0
        if len(correlation_matrix) > 1:
            # Get upper triangle of correlation matrix (excluding diagonal)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            upper_triangle = correlation_matrix.where(mask)
            strongest_corr = upper_triangle.abs().max().max()
        
        st.markdown(f"""
        <div class="metric-container">
            <h3>🔗 Strongest Correlation</h3>
            <h2>{strongest_corr:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volatility = 0
        if 'PFAD_Rate' in df.columns:
            volatility = df['PFAD_Rate'].std() / df['PFAD_Rate'].mean() * 100 if df['PFAD_Rate'].mean() != 0 else 0
        
        st.markdown(f"""
        <div class="metric-container">
            <h3>📈 PFAD Volatility</h3>
            <h2>{volatility:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", 
        "🌡️ Correlation Analysis", 
        "🎯 PFAD Insights",
        "📈 Advanced Statistics",
        "🔬 Statistical Tests",
        "📈 Time Series Analysis"
    ])
    
    with tab1:
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Information:**")
            st.write(f"- **Rows:** {len(df)}")
            st.write(f"- **Columns:** {len(df.columns)}")
            st.write(f"- **Numeric Columns:** {len(numeric_cols)}")
            st.write(f"- **Missing Values:** {df.isnull().sum().sum()}")
        
        with col2:
            st.write("**Column Information:**")
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = df[col].isnull().sum()
                st.write(f"- **{col}:** {dtype} ({missing} missing)")
        
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("🌡️ Correlation Analysis")
        
        if len(numeric_cols) >= 2:
            # Enhanced correlation heatmap
            fig_heatmap = create_enhanced_heatmap(correlation_matrix, chart_height)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Correlation table
            st.subheader("📊 Detailed Correlation Matrix")
            st.dataframe(correlation_matrix.round(3), use_container_width=True)
            
            # Top correlations
            st.subheader("🔝 Strongest Correlations")
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    corr_val = correlation_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        corr_pairs.append({
                            'Variable 1': col1,
                            'Variable 2': col2,
                            'Correlation': corr_val,
                            'Strength': 'Strong' if abs(corr_val) >= 0.7 else 'Moderate' if abs(corr_val) >= 0.4 else 'Weak'
                        })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df.head(10), use_container_width=True)
            
        else:
            st.warning("⚠️ Need at least 2 numeric columns for correlation analysis")
    
    with tab3:
        st.subheader("🎯 PFAD-Specific Analysis")
        
        if 'PFAD_Rate' in df.columns:
            # PFAD correlation chart
            fig_pfad, pfad_corr = create_pfad_correlation_chart(correlation_matrix, correlation_threshold)
            if fig_pfad:
                st.plotly_chart(fig_pfad, use_container_width=True)
            
            # PFAD insights
            st.subheader("💡 PFAD Business Insights")
            insights = generate_business_insights(correlation_matrix, df)
            
            for insight in insights:
                icon = "🔺" if insight['type'] == 'positive' else "🔻" if insight['type'] == 'negative' else "⚡"
                st.markdown(f"""
                <div class="insight-box">
                    <h4>{icon} {insight['title']}</h4>
                    <p><strong>Analysis:</strong> {insight['description']}</p>
                    <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Scatter plots for PFAD
            if len(numeric_cols) >= 2:
                fig_scatter = create_scatter_plots(df, correlation_matrix, 4)
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("⚠️ PFAD_Rate column not found in the dataset")
    
    with tab4:
        st.subheader("📈 Advanced Statistical Analysis")
        
        if len(numeric_cols) >= 2:
            # Bootstrap analysis for key correlations
            st.subheader("🔄 Bootstrap Confidence Intervals")
            
            if 'PFAD_Rate' in df.columns:
                bootstrap_results = []
                pfad_corr = correlation_matrix['PFAD_Rate'].drop('PFAD_Rate', errors='ignore')
                top_corr_vars = pfad_corr.abs().nlargest(3).index
                
                for var in top_corr_vars:
                    if var in df.columns:
                        bootstrap_result = calculate_bootstrap_correlation(
                            df['PFAD_Rate'], df[var], n_bootstrap=1000
                        )
                        
                        if bootstrap_result:
                            bootstrap_results.append({
                                'Variable': var,
                                'Mean Correlation': bootstrap_result['mean_correlation'],
                                'Std Error': bootstrap_result['std_error'],
                                'CI Lower': bootstrap_result['confidence_interval'][0],
                                'CI Upper': bootstrap_result['confidence_interval'][1]
                            })
                
                if bootstrap_results:
                    bootstrap_df = pd.DataFrame(bootstrap_results)
                    st.dataframe(bootstrap_df.round(4), use_container_width=True)
            
            # Rolling correlation analysis
            if 'Date' in df.columns and 'PFAD_Rate' in df.columns:
                st.subheader("📊 Rolling Correlation Analysis")
                fig_rolling = create_rolling_correlation(df, 'PFAD_Rate', trend_window)
                if fig_rolling:
                    st.plotly_chart(fig_rolling, use_container_width=True)
        else:
            st.warning("⚠️ Need more numeric data for advanced statistical analysis")
    
    with tab5:
        st.subheader("🔬 Statistical Significance Testing")
        
        if len(numeric_cols) >= 2:
            # P-value matrix
            p_matrix, sig_matrix = create_significance_matrix(df, confidence_level)
            
            st.subheader("📊 P-Value Matrix")
            st.write(f"**Significance Level:** {1-confidence_level:.3f}")
            st.write("**Legend:** *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
            
            # Display significance matrix
            fig_sig = go.Figure(data=go.Heatmap(
                z=p_matrix.values,
                x=p_matrix.columns,
                y=p_matrix.index,
                colorscale='Reds_r',
                text=sig_matrix.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(
                    title="P-Value",
                    titleside="right"
                )
            ))
            
            fig_sig.update_layout(
                title="🔬 Statistical Significance Matrix",
                height=chart_height,
                xaxis={'side': 'bottom'},
                yaxis={'side': 'left'}
            )
            
            st.plotly_chart(fig_sig, use_container_width=True)
            
            # Detailed p-value table
            st.subheader("📋 Detailed P-Values")
            st.dataframe(p_matrix.round(4), use_container_width=True)
            
            # Statistical summary
            significant_pairs = (p_matrix < (1-confidence_level)).sum().sum() - len(p_matrix)
            total_pairs = len(p_matrix) * (len(p_matrix) - 1)
            
            st.info(f"📊 **Statistical Summary:** {significant_pairs} out of {total_pairs} variable pairs show significant correlation at {confidence_level*100}% confidence level")
        else:
            st.warning("⚠️ Need at least 2 numeric columns for significance testing")
    
    with tab6:
        st.subheader("📈 Time Series Analysis")
        
        df_ts, ts_error = create_time_series_analysis(df)
        
        if df_ts is not None:
            # Time series overview
            st.subheader("📊 Time Series Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Data Points", len(df_ts))
            
            with col2:
                date_span = (df_ts['Date'].max() - df_ts['Date'].min()).days
                st.metric("Date Span (Days)", date_span)
            
            with col3:
                if 'PFAD_Rate' in df_ts.columns:
                    total_change = ((df_ts['PFAD_Rate'].iloc[-1] / df_ts['PFAD_Rate'].iloc[0]) - 1) * 100
                    st.metric("PFAD Total Change", f"{total_change:.1f}%")
                else:
                    st.metric("PFAD Data", "Not Available")
            
            with col4:
                if 'PFAD_Rate' in df_ts.columns:
                    volatility = df_ts['PFAD_Rate'].pct_change().std() * 100
                    st.metric("PFAD Volatility", f"{volatility:.1f}%")
                else:
                    st.metric("Volatility", "N/A")
            
            # Time series variable selection
            numeric_ts_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_ts_cols:
                ts_variable = st.selectbox("Select Variable for Analysis", numeric_ts_cols)
                
                if ts_variable:
                    # Time series plot
                    fig_ts = go.Figure()
                    
                    fig_ts.add_trace(go.Scatter(
                        x=df_ts['Date'],
                        y=df_ts[ts_variable],
                        mode='lines+markers',
                        name=ts_variable,
                        line=dict(width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Add trend line
                    if len(df_ts) > 1:
                        x_numeric = pd.to_numeric(df_ts['Date'])
                        z = np.polyfit(x_numeric, df_ts[ts_variable].fillna(method='ffill'), 1)
                        p = np.poly1d(z)
                        
                        fig_ts.add_trace(go.Scatter(
                            x=df_ts['Date'],
                            y=p(x_numeric),
                            mode='lines',
                            name='Trend',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                    
                    fig_ts.update_layout(
                        title=f"Time Series Analysis: {ts_variable}",
                        xaxis_title="Date",
                        yaxis_title=ts_variable,
                        height=chart_height,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                    # Trend analysis
                    st.subheader("📊 Trend Analysis")
                    
                    # Calculate trend statistics
                    if len(df_ts) > 1:
                        x_numeric = pd.to_numeric(df_ts['Date'])
                        y_data = df_ts[ts_variable].fillna(method='ffill')
                        
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_data)
                        
                        trend_direction = "Upward" if slope > 0 else "Downward" if slope < 0 else "Flat"
                        trend_strength = "Strong" if abs(r_value) > 0.7 else "Moderate" if abs(r_value) > 0.4 else "Weak"
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Trend Direction", trend_direction)
                        
                        with col2:
                            st.metric("Trend Strength", trend_strength)
                        
                        with col3:
                            st.metric("R-squared", f"{r_value**2:.3f}")
                        
                        st.write(f"**Statistical Significance:** p-value = {p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.success("✅ Trend is statistically significant")
                        else:
                            st.warning("⚠️ Trend is not statistically significant")
                    
                    # Simple forecasting
                    st.subheader("🔮 Simple Price Forecasting")
                    
                    if len(df_ts) >= 10:
                        # Use last few points for simple linear extrapolation
                        recent_data = df_ts.tail(min(20, len(df_ts)))
                        
                        if len(recent_data) > 1:
                            x_recent = pd.to_numeric(recent_data['Date'])
                            y_recent = recent_data[ts_variable].fillna(method='ffill')
                            
                            # Fit trend to recent data
                            slope, intercept, _, _, _ = stats.linregress(x_recent, y_recent)
                            
                            # Generate forecast dates
                            last_date = df_ts['Date'].max()
                            forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, forecast_periods+1)]
                            forecast_x = pd.to_numeric(pd.Series(forecast_dates))
                            
                            # Calculate forecasts
                            forecast_values = slope * forecast_x + intercept
                            
                            # Create forecast plot
                            fig_forecast = go.Figure()
                            
                            # Historical data
                            fig_forecast.add_trace(go.Scatter(
                                x=df_ts['Date'],
                                y=df_ts[ts_variable],
                                mode='lines+markers',
                                name='Historical',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Forecast
                            fig_forecast.add_trace(go.Scatter(
                                x=forecast_dates,
                                y=forecast_values,
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            fig_forecast.update_layout(
                                title=f"📈 {ts_variable} Forecast ({forecast_periods} periods ahead)",
                                xaxis_title="Date",
                                yaxis_title=ts_variable,
                                height=400
                            )
                            
                            st.plotly_chart(fig_forecast, use_container_width=True)
                            
                            # Forecast table
                            forecast_df = pd.DataFrame({
                                'Date': forecast_dates,
                                'Forecast': forecast_values.round(2)
                            })
                            
                            st.subheader("📋 Forecast Values")
                            st.dataframe(forecast_df, use_container_width=True)
                            
                            # Business insights for forecasting
                            st.subheader("💡 Forecasting Insights")
                            
                            current_value = df_ts[ts_variable].iloc[-1]
                            forecast_avg = forecast_values.mean()
                            change_pct = ((forecast_avg / current_value) - 1) * 100
                            
                            if change_pct > 5:
                                st.warning(f"⚠️ **Price Increase Expected**: Forecast suggests {ts_variable} may increase by ~{change_pct:.1f}% on average. Consider forward purchasing.")
                            elif change_pct < -5:
                                st.success(f"✅ **Price Decrease Expected**: Forecast suggests {ts_variable} may decrease by ~{abs(change_pct):.1f}% on average. Consider delaying purchases.")
                            else:
                                st.info(f"📊 **Stable Prices Expected**: Forecast suggests {ts_variable} will remain relatively stable (~{change_pct:.1f}% change).")
            
            # Multi-variable comparison
            if len(numeric_ts_cols) > 1:
                st.subheader("📊 Multi-Variable Trend Comparison")
                
                # Normalize all variables to start at 100
                fig_multi = go.Figure()
                
                for col in numeric_ts_cols[:5]:  # Limit to 5 variables for readability
                    if col in df_ts.columns:
                        normalized_values = (df_ts[col] / df_ts[col].iloc[0]) * 100
                        
                        fig_multi.add_trace(go.Scatter(
                            x=df_ts['Date'],
                            y=normalized_values,
                            mode='lines',
                            name=col,
                            line=dict(width=2)
                        ))
                
                fig_multi.update_layout(
                    title="📈 Normalized Price Trends (Base = 100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Value (Base = 100)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_multi, use_container_width=True)
        else:
            st.warning(f"⚠️ Time series analysis not available: {ts_error}")
            st.info("💡 To enable time series features, ensure your data has a properly formatted date column.")

else:
    # Welcome message
    st.info("""
    👋 **Welcome to the PFAD Procurement Analytics Dashboard!**
    
    📁 **To get started:**
    1. Upload your Excel file using the sidebar
    2. Ensure your file contains the following columns:
       - Date (for time series analysis)
       - Imported PFAD - CIF in INR / ton (IA)
       - CPO BMD Price (MYR)
       - Malaysia FOB USD
       - USD/INR, USD/MYR
       - Brent crude (USD)
       - CPO Volume
    
    🎯 **What you'll get:**
    - **Correlation Analysis:** Discover relationships between variables
    - **PFAD Insights:** Specific analysis for PFAD procurement
    - **Statistical Testing:** Professional-grade statistical analysis
    - **Time Series Analysis:** Trend analysis and forecasting
    - **Business Recommendations:** AI-powered procurement insights
    
    📊 **Dashboard Features:**
    - Interactive visualizations
    - Statistical significance testing
    - Bootstrap confidence intervals
    - Rolling correlation analysis
    - Price forecasting
    - Export capabilities
    
    Start by uploading your data file! 🚀
    """)
