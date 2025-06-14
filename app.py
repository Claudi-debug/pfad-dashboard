import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PFAD Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; text-align: center; margin: 0;">📊 PFAD Procurement Analytics</h1>
    <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; opacity: 0.9;">Strategic Procurement Decision Support</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("📁 Upload Excel File", type=['xlsx', 'xls'])

# Settings
chart_height = st.sidebar.slider("Chart Height", 300, 800, 500)
confidence_level = st.sidebar.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)

def load_data(file):
    """Load and clean data"""
    try:
        df = pd.read_excel(file)
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Standard column mapping
        mapping = {
            'Date': 'Date',
            'Imported PFAD - CIF in INR / ton (IA)': 'PFAD_Rate',
            'CPO BMD Price (MYR)': 'CPO_BMD_MYR',
            'Malaysia FOB USD': 'Malaysia_FOB_USD',
            'USD/INR': 'USD_INR',
            'USD/MYR': 'USD_MYR',
            'Brent crude (USD)': 'Brent_Crude_USD',
            'CPO Volume': 'CPO_Volume'
        }
        
        # Find and rename columns
        for old_name, new_name in mapping.items():
            for col in df.columns:
                if old_name.lower() in col.lower():
                    df = df.rename(columns={col: new_name})
                    break
        
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        return df, None
    except Exception as e:
        return None, str(e)

def create_correlation_heatmap(corr_matrix):
    """Simple correlation heatmap"""
    fig = px.imshow(
        corr_matrix,
        text_auto='.3f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(
        title="🌡️ Correlation Matrix",
        height=chart_height
    )
    return fig

def create_pfad_bar_chart(corr_matrix):
    """PFAD correlation bar chart"""
    if 'PFAD_Rate' not in corr_matrix.columns:
        return None
    
    pfad_corr = corr_matrix['PFAD_Rate'].drop('PFAD_Rate').sort_values(key=abs, ascending=False)
    
    colors = ['green' if abs(x) >= 0.7 else 'orange' if abs(x) >= 0.4 else 'red' for x in pfad_corr]
    
    fig = go.Figure(data=go.Bar(
        x=pfad_corr.values,
        y=pfad_corr.index,
        orientation='h',
        marker_color=colors,
        text=[f"{val:.3f}" for val in pfad_corr.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="🎯 PFAD Correlations (Ranked)",
        xaxis_title="Correlation",
        height=400
    )
    return fig

def create_scatter_plots(df, corr_matrix):
    """Top correlation scatter plots"""
    # Get top 4 correlations
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val):
                correlations.append((col1, col2, abs(corr_val), corr_val))
    
    correlations.sort(key=lambda x: x[2], reverse=True)
    top_4 = correlations[:4]
    
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=[f"{pair[0]} vs {pair[1]} (r={pair[3]:.3f})" for pair in top_4])
    
    for idx, (col1, col2, _, corr) in enumerate(top_4):
        row, col = (idx // 2) + 1, (idx % 2) + 1
        
        if col1 in df.columns and col2 in df.columns:
            fig.add_trace(
                go.Scatter(x=df[col1], y=df[col2], mode='markers', 
                          name=f"{col1} vs {col2}", showlegend=False),
                row=row, col=col
            )
    
    fig.update_layout(height=600, title_text="🔍 Top Correlation Scatter Plots")
    return fig

def calculate_statistics(df, corr_matrix):
    """Calculate basic statistics"""
    stats_data = []
    
    if 'PFAD_Rate' in corr_matrix.columns:
        pfad_corr = corr_matrix['PFAD_Rate'].drop('PFAD_Rate')
        
        for var in pfad_corr.index:
            if var in df.columns:
                corr_val = pfad_corr[var]
                
                # Calculate p-value
                x = df['PFAD_Rate'].dropna()
                y = df[var].dropna()
                common_idx = x.index.intersection(y.index)
                
                if len(common_idx) >= 3:
                    try:
                        _, p_val = stats.pearsonr(x.loc[common_idx], y.loc[common_idx])
                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    except:
                        p_val, significance = np.nan, "na"
                else:
                    p_val, significance = np.nan, "na"
                
                stats_data.append({
                    'Variable': var,
                    'Correlation': corr_val,
                    'P-Value': p_val,
                    'Significance': significance,
                    'Strength': 'Strong' if abs(corr_val) >= 0.7 else 'Moderate' if abs(corr_val) >= 0.4 else 'Weak'
                })
    
    return pd.DataFrame(stats_data)

def create_time_series_plot(df):
    """Time series analysis"""
    if 'Date' not in df.columns:
        return None
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None
    
    # Select variable for time series
    ts_var = st.selectbox("Select Variable for Time Series", numeric_cols)
    
    if ts_var:
        df_clean = df[['Date', ts_var]].dropna()
        df_clean = df_clean.sort_values('Date')
        
        fig = go.Figure()
        
        # Add main line
        fig.add_trace(go.Scatter(
            x=df_clean['Date'],
            y=df_clean[ts_var],
            mode='lines+markers',
            name=ts_var,
            line=dict(width=2)
        ))
        
        # Add trend line
        if len(df_clean) > 1:
            x_numeric = pd.to_numeric(df_clean['Date'])
            z = np.polyfit(x_numeric, df_clean[ts_var], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=df_clean['Date'],
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=f"📈 Time Series: {ts_var}",
            xaxis_title="Date",
            yaxis_title=ts_var,
            height=chart_height
        )
        
        return fig
    return None

def generate_insights(corr_matrix, df):
    """Generate business insights"""
    insights = []
    
    if 'PFAD_Rate' in corr_matrix.columns:
        pfad_corr = corr_matrix['PFAD_Rate'].drop('PFAD_Rate')
        
        # Strongest positive correlation
        if len(pfad_corr[pfad_corr > 0]) > 0:
            strongest_pos = pfad_corr[pfad_corr > 0].idxmax()
            corr_val = pfad_corr[strongest_pos]
            insights.append({
                'title': f"🔺 Strongest Positive Driver: {strongest_pos}",
                'description': f"PFAD rates have a {corr_val:.3f} correlation with {strongest_pos}",
                'recommendation': f"Monitor {strongest_pos} as a leading indicator for PFAD price increases"
            })
        
        # Strongest negative correlation
        if len(pfad_corr[pfad_corr < 0]) > 0:
            strongest_neg = pfad_corr[pfad_corr < 0].idxmin()
            corr_val = pfad_corr[strongest_neg]
            insights.append({
                'title': f"🔻 Strongest Negative Driver: {strongest_neg}",
                'description': f"PFAD rates have a {corr_val:.3f} correlation with {strongest_neg}",
                'recommendation': f"When {strongest_neg} is high, consider purchasing PFAD at potentially lower rates"
            })
        
        # Market efficiency
        avg_abs_corr = pfad_corr.abs().mean()
        if avg_abs_corr > 0.5:
            insights.append({
                'title': "⚡ High Market Efficiency",
                'description': f"Average correlation strength is {avg_abs_corr:.3f}",
                'recommendation': "Market shows predictable patterns. Systematic procurement strategies recommended."
            })
    
    return insights

# Main Application
if uploaded_file:
    # Load data
    df, error = load_data(uploaded_file)
    
    if error:
        st.error(f"❌ Error: {error}")
        st.stop()
    
    # Success message
    st.success(f"✅ Data loaded! Shape: {df.shape}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr() if len(numeric_cols) >= 2 else pd.DataFrame()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Records</h3>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        date_range = "N/A"
        if 'Date' in df.columns:
            dates = df['Date'].dropna()
            if len(dates) > 0:
                date_range = f"{dates.min().strftime('%Y-%m')} to {dates.max().strftime('%Y-%m')}"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>📅 Period</h3>
            <h2>{date_range}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        strongest_corr = 0
        if not correlation_matrix.empty:
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            upper = correlation_matrix.where(mask)
            strongest_corr = upper.abs().max().max() if not upper.empty else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔗 Max Correlation</h3>
            <h2>{strongest_corr:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volatility = 0
        if 'PFAD_Rate' in df.columns:
            pfad_data = df['PFAD_Rate'].dropna()
            if len(pfad_data) > 0 and pfad_data.mean() != 0:
                volatility = (pfad_data.std() / pfad_data.mean()) * 100
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 PFAD Volatility</h3>
            <h2>{volatility:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🌡️ Correlations", "🎯 PFAD Analysis", 
        "📈 Statistics", "🔬 Tests", "📈 Time Series"
    ])
    
    with tab1:
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Info:**")
            st.write(f"- Rows: {len(df)}")
            st.write(f"- Columns: {len(df.columns)}")
            st.write(f"- Numeric: {len(numeric_cols)}")
            st.write(f"- Missing: {df.isnull().sum().sum()}")
        
        with col2:
            st.write("**Columns:**")
            for col in df.columns[:8]:  # Show first 8 columns
                missing = df[col].isnull().sum()
                st.write(f"- {col}: {missing} missing")
        
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        if len(numeric_cols) > 0:
            st.subheader("📈 Summary Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    with tab2:
        st.subheader("🌡️ Correlation Analysis")
        
        if not correlation_matrix.empty and len(correlation_matrix) >= 2:
            # Correlation heatmap
            fig_heatmap = create_correlation_heatmap(correlation_matrix)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Correlation table
            st.subheader("📊 Correlation Matrix")
            st.dataframe(correlation_matrix.round(3), use_container_width=True)
            
            # Top correlations
            st.subheader("🔝 Strongest Correlations")
            pairs = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    col1, col2 = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    val = correlation_matrix.iloc[i, j]
                    if not np.isnan(val):
                        pairs.append({
                            'Variable 1': col1,
                            'Variable 2': col2,
                            'Correlation': val,
                            'Strength': 'Strong' if abs(val) >= 0.7 else 'Moderate' if abs(val) >= 0.4 else 'Weak'
                        })
            
            if pairs:
                pairs_df = pd.DataFrame(pairs).sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(pairs_df.head(10), use_container_width=True)
        else:
            st.warning("⚠️ Need at least 2 numeric columns for correlation analysis")
    
    with tab3:
        st.subheader("🎯 PFAD Analysis")
        
        if 'PFAD_Rate' in correlation_matrix.columns:
            # PFAD correlation chart
            fig_pfad = create_pfad_bar_chart(correlation_matrix)
            if fig_pfad:
                st.plotly_chart(fig_pfad, use_container_width=True)
            
            # Business insights
            st.subheader("💡 Business Insights")
            insights = generate_insights(correlation_matrix, df)
            
            for insight in insights:
                st.markdown(f"""
                <div class="insight-box">
                    <h4>{insight['title']}</h4>
                    <p><strong>Analysis:</strong> {insight['description']}</p>
                    <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Scatter plots
            if len(correlation_matrix) >= 2:
                fig_scatter = create_scatter_plots(df, correlation_matrix)
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("⚠️ PFAD_Rate column not found")
    
    with tab4:
        st.subheader("📈 Statistical Analysis")
        
        if not correlation_matrix.empty:
            # Statistics table
            stats_df = calculate_statistics(df, correlation_matrix)
            if not stats_df.empty:
                st.subheader("📊 PFAD Correlation Statistics")
                st.dataframe(stats_df.round(4), use_container_width=True)
                
                # Summary
                strong_count = len(stats_df[stats_df['Strength'] == 'Strong'])
                moderate_count = len(stats_df[stats_df['Strength'] == 'Moderate'])
                weak_count = len(stats_df[stats_df['Strength'] == 'Weak'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Strong Correlations", strong_count)
                with col2:
                    st.metric("Moderate Correlations", moderate_count)
                with col3:
                    st.metric("Weak Correlations", weak_count)
        else:
            st.warning("⚠️ No correlation data available")
    
    with tab5:
        st.subheader("🔬 Statistical Tests")
        
        if not correlation_matrix.empty and 'PFAD_Rate' in correlation_matrix.columns:
            # P-value analysis
            st.subheader("📊 Significance Testing")
            
            stats_df = calculate_statistics(df, correlation_matrix)
            if not stats_df.empty:
                # Filter out NaN p-values
                valid_stats = stats_df.dropna(subset=['P-Value'])
                
                if not valid_stats.empty:
                    # Count significant correlations
                    alpha = 1 - confidence_level
                    significant = valid_stats[valid_stats['P-Value'] < alpha]
                    
                    st.write(f"**Confidence Level:** {confidence_level*100}%")
                    st.write(f"**Significance Threshold:** p < {alpha}")
                    st.write(f"**Significant Correlations:** {len(significant)} out of {len(valid_stats)}")
                    
                    # Show significant correlations
                    if not significant.empty:
                        st.subheader("✅ Statistically Significant Correlations")
                        st.dataframe(significant[['Variable', 'Correlation', 'P-Value', 'Significance']], use_container_width=True)
                    
                    # Show all p-values
                    st.subheader("📋 All P-Values")
                    st.dataframe(valid_stats[['Variable', 'Correlation', 'P-Value', 'Significance']], use_container_width=True)
                else:
                    st.warning("⚠️ No valid p-values calculated")
        else:
            st.warning("⚠️ Need PFAD_Rate column for significance testing")
    
    with tab6:
        st.subheader("📈 Time Series Analysis")
        
        if 'Date' in df.columns:
            # Time series plot
            fig_ts = create_time_series_plot(df)
            if fig_ts:
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # Time series insights
                st.subheader("📊 Time Series Insights")
                
                numeric_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols_list:
                    selected_var = st.selectbox("Analyze Variable", numeric_cols_list, key="ts_insight")
                    
                    if selected_var:
                        ts_data = df[['Date', selected_var]].dropna().sort_values('Date')
                        
                        if len(ts_data) > 1:
                            # Calculate basic trend
                            first_val = ts_data[selected_var].iloc[0]
                            last_val = ts_data[selected_var].iloc[-1]
                            total_change = ((last_val / first_val) - 1) * 100 if first_val != 0 else 0
                            
                            # Volatility
                            volatility = ts_data[selected_var].pct_change().std() * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Change", f"{total_change:.1f}%")
                            with col2:
                                st.metric("Volatility", f"{volatility:.1f}%")
                            with col3:
                                trend = "📈 Rising" if total_change > 5 else "📉 Falling" if total_change < -5 else "➡️ Stable"
                                st.metric("Trend", trend)
                            
                            # Business insight
                            if abs(total_change) > 10:
                                if total_change > 0:
                                    st.success(f"✅ **Price Increase Detected:** {selected_var} has increased by {total_change:.1f}% over the period. Consider forward contracts or alternative sourcing.")
                                else:
                                    st.info(f"📉 **Price Decrease Detected:** {selected_var} has decreased by {abs(total_change):.1f}% over the period. Good time for strategic purchasing.")
                            else:
                                st.info(f"📊 **Stable Pricing:** {selected_var} has remained relatively stable with {total_change:.1f}% change.")
            else:
                st.warning("⚠️ No numeric data available for time series analysis")
        else:
            st.warning("⚠️ No date column found for time series analysis")

else:
    st.info("""
    👋 **Welcome to PFAD Procurement Analytics!**
    
    📁 **Upload your Excel file to start:**
    - Use the file uploader in the sidebar
    - Ensure your file contains PFAD and related commodity data
    - Date column recommended for time series analysis
    
    🎯 **You'll get:**
    - **Correlation Analysis** - Discover price relationships
    - **PFAD Insights** - Specific procurement recommendations  
    - **Statistical Testing** - Professional analysis with p-values
    - **Time Series** - Trend analysis and forecasting insights
    
    📊 **Perfect for:**
    - Treasury teams
    - Procurement managers
    - Strategic planning
    - Risk management
    
    Start by uploading your data! 🚀
    """)
