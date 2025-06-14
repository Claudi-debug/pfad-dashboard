# PFAD Correlation Analysis - Streamlit Cloud Optimized
# Deployed at: https://your-app.streamlit.app

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="PFAD Procurement Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-username/pfad-analytics-dashboard',
        'Report a bug': 'https://github.com/your-username/pfad-analytics-dashboard/issues',
        'About': "# PFAD Procurement Analytics Dashboard\nAI-Powered correlation analysis for strategic procurement decisions."
    }
)

# Custom CSS for professional styling
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Header with animation
st.markdown("""
<div class="main-header">
    <h1>🚀 PFAD Procurement Analytics Dashboard</h1>
    <p style="font-size: 1.2em; margin-bottom: 0;">AI-Powered Correlation Analysis for Strategic Decision Making</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Deployed on Streamlit Cloud ☁️</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📁 Data Upload & Settings")
st.sidebar.markdown("---")

# File upload with enhanced UI
uploaded_file = st.sidebar.file_uploader(
    "📊 Upload Your Excel File",
    type=['xlsx', 'xls'],
    help="Upload your PFAD Data Analytics Excel file (Max: 200MB)",
    accept_multiple_files=False
)

# Analysis settings
st.sidebar.header("⚙️ Analysis Settings")
correlation_threshold = st.sidebar.slider(
    "Correlation Strength Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Minimum correlation strength to highlight in analysis"
)

show_p_values = st.sidebar.checkbox("Show Statistical Significance", value=True)
show_insights = st.sidebar.checkbox("Generate AI Insights", value=True)
normalize_data = st.sidebar.checkbox("Normalize Price Data", value=True)

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    rolling_window = st.slider("Rolling Correlation Window", 3, 24, 12, help="Months for rolling correlation analysis")
    chart_height = st.slider("Chart Height", 400, 800, 500, help="Height of charts in pixels")
    color_scheme = st.selectbox("Color Scheme", ["Default", "Viridis", "Plasma", "Turbo"], help="Color scheme for visualizations")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: For best results, ensure your Excel file has consistent column names and numeric data.")

@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file):
    """Load and process the uploaded Excel file with caching"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file, sheet_name='Sheet1')
        
        # Convert column names to strings and clean
        df.columns = [str(col).strip() for col in df.columns]
        
        # Flexible column mapping
        def find_column(df, possible_names):
            for name in possible_names:
                if name in df.columns:
                    return name
            return None
        
        column_mappings = {
            'Date': ['Date', 'date', 'DATE', 'Month', 'month'],
            'PFAD_Rate': ['Imported PFAD - CIF in INR / ton (IA)', 'PFAD Rate', 'PFAD_Rate', 'PFAD'],
            'CPO_BMD_Price': ['CPO BMD Price (MYR)', 'CPO Price', 'BMD Price', 'CPO_Price'],
            'Malaysia_FOB': ['Malaysia  FOB USD', 'Malaysia FOB USD', 'FOB USD', 'Malaysia FOB'],
            'USD_INR': ['USD/INR', 'USDINR', 'USD_INR', 'INR'],
            'USD_MYR': ['USD/MYR', 'USDMYR', 'USD_MYR', 'MYR'],
            'Brent_Crude': ['Brent crude (USD)', 'Brent Crude', 'Brent', 'Crude'],
            'CPO_Volume': ['CPO Volume', 'Volume', 'CPO_Volume']
        }
        
        # Apply column mapping
        rename_dict = {}
        mapping_results = []
        for new_name, possible_names in column_mappings.items():
            found_col = find_column(df, possible_names)
            if found_col:
                rename_dict[found_col] = new_name
                mapping_results.append(f"✅ '{found_col}' → '{new_name}'")
            else:
                mapping_results.append(f"⚠️ Column for '{new_name}' not found")
        
        df.rename(columns=rename_dict, inplace=True)
        
        # Handle date column
        date_col = find_column(df, ['Date', 'date', 'DATE', 'Month', 'month'])
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df_clean = df[numeric_columns].dropna(how='all').dropna(axis=1, how='all')
        
        return df_clean, True, "Data loaded and processed successfully!", mapping_results
        
    except Exception as e:
        return None, False, f"Error loading data: {str(e)}", []

@st.cache_data(show_spinner=False)
def calculate_correlations(df):
    """Calculate correlation matrix and statistics with caching"""
    correlation_matrix = df.corr()
    
    # Find PFAD column
    pfad_columns = ['PFAD_Rate', 'PFAD', 'Pfad_Rate']
    pfad_col = None
    for col in pfad_columns:
        if col in df.columns:
            pfad_col = col
            break
    
    if pfad_col:
        pfad_correlations = correlation_matrix[pfad_col].drop(pfad_col)
        
        # Calculate p-values
        p_values = {}
        for col in df.columns:
            if col != pfad_col:
                clean_data = df[[pfad_col, col]].dropna()
                if len(clean_data) >= 3:
                    _, p_val = stats.pearsonr(clean_data[pfad_col], clean_data[col])
                    p_values[col] = p_val
                else:
                    p_values[col] = 1.0
        
        return correlation_matrix, pfad_correlations, p_values, pfad_col
    
    return correlation_matrix, None, {}, None

def create_enhanced_heatmap(correlation_matrix, height=500):
    """Create an enhanced correlation heatmap"""
    color_scales = {
        "Default": "RdYlBu_r",
        "Viridis": "Viridis",
        "Plasma": "Plasma", 
        "Turbo": "Turbo"
    }
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="🌡️ Interactive Correlation Matrix",
        color_continuous_scale=color_scales.get(color_scheme, "RdYlBu_r"),
        zmin=-1,
        zmax=1
    )
    
    fig.update_layout(
        height=height,
        title_x=0.5,
        font=dict(size=12),
        coloraxis_colorbar=dict(
            title="Correlation<br>Coefficient",
            titleside="right"
        )
    )
    
    fig.update_traces(
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
    )
    
    return fig

def generate_ai_insights(df, pfad_correlations, pfad_col):
    """Generate comprehensive AI insights"""
    try:
        insights = []
        
        # Data overview
        insights.append("🤖 **AI Analysis Results**")
        insights.append(f"📊 **Dataset Overview**: {len(df)} records across {len(df.columns)} variables")
        
        if hasattr(df.index, 'strftime'):
            insights.append(f"📅 **Time Period**: {df.index.min().strftime('%B %Y')} to {df.index.max().strftime('%B %Y')}")
        
        # Correlation analysis
        strong_corr = pfad_correlations[pfad_correlations.abs() > 0.7]
        moderate_corr = pfad_correlations[(pfad_correlations.abs() > 0.5) & (pfad_correlations.abs() <= 0.7)]
        weak_corr = pfad_correlations[pfad_correlations.abs() <= 0.3]
        
        insights.append("\n🔥 **Correlation Strength Analysis**:")
        insights.append(f"• **Strong relationships** (|r| > 0.7): {len(strong_corr)} variables")
        insights.append(f"• **Moderate relationships** (0.5 < |r| ≤ 0.7): {len(moderate_corr)} variables")
        insights.append(f"• **Weak relationships** (|r| ≤ 0.3): {len(weak_corr)} variables")
        
        if len(strong_corr) > 0:
            insights.append(f"\n💎 **Primary Price Drivers**:")
            for var, corr in strong_corr.head(3).items():
                direction = "positive" if corr > 0 else "negative"
                insights.append(f"• **{var}**: {corr:.3f} ({direction} relationship)")
        
        # Market insights
        top_predictor = pfad_correlations.abs().idxmax()
        top_correlation = pfad_correlations[top_predictor]
        
        insights.append(f"\n🎯 **Strategic Recommendations**:")
        insights.append(f"• **Primary Monitor**: {top_predictor} (correlation: {top_correlation:.3f})")
        
        if abs(top_correlation) > 0.6:
            if top_correlation > 0:
                insights.append("• **Buy Signal**: Consider purchasing when this indicator trends downward")
                insights.append("• **Hold Signal**: Delay purchases during upward trends")
            else:
                insights.append("• **Buy Signal**: Consider purchasing when this indicator trends upward")
                insights.append("• **Hold Signal**: Delay purchases during downward trends")
        
        # Risk assessment
        volatility = df[pfad_col].std() / df[pfad_col].mean() * 100
        predictability = pfad_correlations.abs().max() * 100
        
        insights.append(f"\n⚠️ **Risk Assessment**:")
        insights.append(f"• **Price Volatility**: {volatility:.1f}% coefficient of variation")
        insights.append(f"• **Predictability Score**: {predictability:.1f}%")
        
        if predictability > 80:
            risk_level = "**Low Risk** - High predictability suggests stable market patterns"
        elif predictability > 60:
            risk_level = "**Moderate Risk** - Good predictability with some uncertainty"
        else:
            risk_level = "**High Risk** - Low predictability requires careful monitoring"
        
        insights.append(f"• **Risk Level**: {risk_level}")
        
        # Action items
        insights.append(f"\n🚀 **Recommended Actions**:")
        insights.append("• Set up automated alerts for top 3 correlates")
        insights.append("• Establish procurement thresholds based on key indicators")
        insights.append("• Implement monthly correlation stability reviews")
        
        if predictability > 60:
            insights.append("• Develop predictive models for procurement timing")
        
        insights.append("• Create hedging strategies for volatile market periods")
        
        return "\n".join(insights)
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Main application logic
if uploaded_file is not None:
    # Process data with progress bar
    with st.spinner("🔄 Loading and processing your data..."):
        progress_bar = st.progress(0)
        
        # Load data
        progress_bar.progress(25)
        df_clean, success, message, mapping_results = load_and_process_data(uploaded_file)
        progress_bar.progress(50)
        
        if success:
            st.session_state.df_clean = df_clean
            st.session_state.data_processed = True
            
            # Calculate correlations
            correlation_matrix, pfad_correlations, p_values, pfad_col = calculate_correlations(df_clean)
            progress_bar.progress(75)
            
            # Store results
            st.session_state.analysis_results = {
                'correlation_matrix': correlation_matrix,
                'pfad_correlations': pfad_correlations,
                'p_values': p_values,
                'pfad_col': pfad_col
            }
            progress_bar.progress(100)
            
            # Success message with details
            st.success(f"✅ {message}")
            
            with st.expander("📋 Data Processing Details", expanded=False):
                st.write("**Column Mapping Results:**")
                for result in mapping_results:
                    st.write(result)
                
                st.write(f"\n**Data Quality:**")
                st.write(f"• Original shape: {uploaded_file.name} - Processing completed")
                st.write(f"• Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
                st.write(f"• Missing values: {df_clean.isnull().sum().sum()} total")
            
            progress_bar.empty()
            
        else:
            st.error(f"❌ {message}")
            st.stop()

# Display analysis if data is loaded
if st.session_state.data_processed and st.session_state.df_clean is not None:
    df_clean = st.session_state.df_clean
    analysis_results = st.session_state.analysis_results
    
    correlation_matrix = analysis_results['correlation_matrix']
    pfad_correlations = analysis_results['pfad_correlations']
    p_values = analysis_results['p_values']
    pfad_col = analysis_results['pfad_col']
    
    # Key Metrics Dashboard
    st.markdown("## 📊 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Total Records",
            value=f"{len(df_clean):,}",
            help="Number of data points in the analysis"
        )
    
    with col2:
        if hasattr(df_clean.index, 'strftime'):
            date_range = f"{df_clean.index.min().strftime('%b %Y')} - {df_clean.index.max().strftime('%b %Y')}"
        else:
            date_range = "Date index not available"
        st.metric(
            label="📅 Time Span",
            value=date_range,
            help="Time period covered by the analysis"
        )
    
    with col3:
        if pfad_correlations is not None:
            strongest_var = pfad_correlations.abs().idxmax()
            strongest_val = pfad_correlations[strongest_var]
            st.metric(
                label="🎯 Strongest Correlation",
                value=f"{strongest_val:.3f}",
                delta=f"with {strongest_var}",
                help="Variable with highest correlation to PFAD"
            )
        else:
            st.metric(label="🎯 Strongest Correlation", value="N/A")
    
    with col4:
        if pfad_correlations is not None:
            avg_corr = pfad_correlations.abs().mean()
            volatility = df_clean[pfad_col].std() / df_clean[pfad_col].mean() * 100
            st.metric(
                label="⚡ Market Volatility",
                value=f"{volatility:.1f}%",
                delta=f"Avg |r|: {avg_corr:.3f}",
                help="Price volatility and average correlation strength"
            )
        else:
            st.metric(label="⚡ Market Volatility", value="N/A")
    
    # Main Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Correlation Matrix", "🎯 PFAD Analysis", "📈 Time Series", "🤖 AI Insights"])
    
    with tab1:
        st.header("📊 Comprehensive Correlation Analysis")
        
        # Enhanced correlation heatmap
        fig_heatmap = create_enhanced_heatmap(correlation_matrix, chart_height)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Correlation statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Correlation Statistics")
            stats_df = pd.DataFrame({
                'Mean': correlation_matrix.mean(),
                'Std Dev': correlation_matrix.std(),
                'Min': correlation_matrix.min(),
                'Max': correlation_matrix.max()
            }).round(3)
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Correlation Distribution")
            # Create histogram of correlations
            all_corrs = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
            fig_hist = px.histogram(
                x=all_corrs,
                nbins=20,
                title="Distribution of Correlation Coefficients",
                labels={'x': 'Correlation Coefficient', 'y': 'Frequency'}
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        if pfad_correlations is not None:
            st.header(f"🎯 {pfad_col} Deep Dive Analysis")
            
            # PFAD correlations visualization
            pfad_corr_sorted = pfad_correlations.sort_values(ascending=True)
            
            # Enhanced color coding
            def get_color(val):
                if abs(val) >= 0.8:
                    return '#27ae60'  # Dark green for very strong
                elif abs(val) >= 0.6:
                    return '#2ecc71'  # Green for strong
                elif abs(val) >= 0.4:
                    return '#f39c12'  # Orange for moderate
                elif abs(val) >= 0.2:
                    return '#e67e22'  # Dark orange for weak
                else:
                    return '#e74c3c'  # Red for very weak
            
            colors = [get_color(x) for x in pfad_corr_sorted.values]
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=pfad_corr_sorted.values,
                    y=pfad_corr_sorted.index,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{val:.3f}" for val in pfad_corr_sorted.values],
                    textposition='auto',
                    hovertemplate="<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>"
                )
            ])
            
            fig_bar.update_layout(
                title=f"🎯 {pfad_col} Correlations (Strength Ranked)",
                xaxis_title="Correlation Coefficient",
                height=chart_height,
                title_x=0.5,
                showlegend=False
            )
            
            # Add reference lines with labels
            fig_bar.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.7, 
                             annotation_text="Neutral", annotation_position="top")
            fig_bar.add_vline(x=0.5, line_dash="dash", line_color="green", opacity=0.7,
                             annotation_text="Strong +", annotation_position="top")
            fig_bar.add_vline(x=0.8, line_dash="dot", line_color="darkgreen", opacity=0.7,
                             annotation_text="Very Strong +", annotation_position="top")
            fig_bar.add_vline(x=-0.5, line_dash="dash", line_color="green", opacity=0.7,
                             annotation_text="Strong -", annotation_position="bottom")
            fig_bar.add_vline(x=-0.8, line_dash="dot", line_color="darkgreen", opacity=0.7,
                             annotation_text="Very Strong -", annotation_position="bottom")
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Statistical significance analysis
            if show_p_values:
                st.subheader("📊 Statistical Significance Analysis")
                
                sig_data = []
                for var, corr in pfad_correlations.items():
                    p_val = p_values.get(var, 1.0)
                    
                    if p_val < 0.001:
                        significance = "***"
                        sig_text = "Very Significant"
                    elif p_val < 0.01:
                        significance = "**"
                        sig_text = "Significant"
                    elif p_val < 0.05:
                        significance = "*"
                        sig_text = "Moderately Significant"
                    else:
                        significance = ""
                        sig_text = "Not Significant"
                    
                    if abs(corr) > 0.7:
                        strength = "Very Strong"
                    elif abs(corr) > 0.5:
                        strength = "Strong"
                    elif abs(corr) > 0.3:
                        strength = "Moderate"
                    else:
                        strength = "Weak"
                    
                    sig_data.append({
                        'Variable': var,
                        'Correlation': f"{corr:.3f}",
                        'P-Value': f"{p_val:.6f}",
                        'Significance': significance,
                        'Interpretation': sig_text,
                        'Strength': strength
                    })
                
                sig_df = pd.DataFrame(sig_data)
                
                # Enhanced table display
                st.dataframe(
                    sig_df,
                    use_container_width=True,
                    column_config={
                        "Variable": st.column_config.TextColumn("Variable", help="Market variable name"),
                        "Correlation": st.column_config.NumberColumn("Correlation", help="Pearson correlation coefficient"),
                        "P-Value": st.column_config.NumberColumn("P-Value", help="Statistical significance p-value"),
                        "Significance": st.column_config.TextColumn("Sig.", help="*** p<0.001, ** p<0.01, * p<0.05"),
                        "Strength": st.column_config.TextColumn("Strength", help="Correlation strength category")
                    }
                )
            
            # Interactive scatter plot explorer
            st.subheader("💫 Relationship Explorer")
            
            scatter_var = st.selectbox(
                "Select variable to analyze relationship with PFAD:",
                options=pfad_correlations.abs().sort_values(ascending=False).index.tolist(),
                index=0,
                help="Choose a variable to see its detailed relationship with PFAD rates"
            )
            
            if scatter_var:
                plot_data = df_clean[[pfad_col, scatter_var]].dropna()
                
                if len(plot_data) > 0:
                    corr_val = pfad_correlations[scatter_var]
                    
                    fig_scatter = px.scatter(
                        plot_data,
                        x=scatter_var,
                        y=pfad_col,
                        title=f"📈 {pfad_col} vs {scatter_var} (r = {corr_val:.3f})",
                        trendline="ols",
                        hover_data={scatter_var: ':.2f', pfad_col: ':.2f'},
                        color_discrete_sequence=['#667eea']
                    )
                    
                    fig_scatter.update_layout(height=chart_height, title_x=0.5)
                    fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Add relationship interpretation
                    if abs(corr_val) > 0.7:
                        interpretation = "🔥 **Very Strong Relationship** - This variable is a primary driver of PFAD prices"
                    elif abs(corr_val) > 0.5:
                        interpretation = "🔶 **Strong Relationship** - This variable significantly influences PFAD prices"
                    elif abs(corr_val) > 0.3:
                        interpretation = "🔸 **Moderate Relationship** - This variable has some influence on PFAD prices"
                    else:
                        interpretation = "⚪ **Weak Relationship** - This variable has limited influence on PFAD prices"
                    
                    st.info(interpretation)
                else:
                    st.warning("No data available for this relationship analysis")
        else:
            st.warning("🔍 PFAD Rate column not found in the uploaded data. Please check your column names.")
    
    with tab3:
        st.header("📈 Time Series & Trend Analysis")
        
        if hasattr(df_clean.index, 'strftime'):
            # Price trends comparison
            price_columns = [col for col in df_clean.columns if 'Price' in col or 'Rate' in col or 'FOB' in col]
            
            if len(price_columns) > 1:
                st.subheader("📊 Comparative Price Trends")
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    trend_options = st.multiselect(
                        "Select variables to plot:",
                        options=price_columns,
                        default=price_columns[:4] if len(price_columns) > 4 else price_columns,
                        help="Choose which price variables to display"
                    )
                
                with col1:
                    if trend_options:
                        if normalize_data:
                            # Normalize to base 100
                            df_normalized = df_clean[trend_options].div(df_clean[trend_options].iloc[0]) * 100
                            y_title = "Normalized Price Index (Base = 100)"
                            title_suffix = "(Normalized)"
                        else:
                            df_normalized = df_clean[trend_options]
                            y_title = "Price Value"
                            title_suffix = "(Absolute Values)"
                        
                        fig_trends = go.Figure()
                        
                        for col in trend_options:
                            fig_trends.add_trace(go.Scatter(
                                x=df_normalized.index,
                                y=df_normalized[col],
                                mode='lines+markers',
                                name=col,
                                line=dict(width=3),
                                marker=dict(size=4),
                                hovertemplate=f"<b>{col}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>"
                            ))
                        
                        fig_trends.update_layout(
                            title=f"📈 Price Trends Over Time {title_suffix}",
                            xaxis_title="Date",
                            yaxis_title=y_title,
                            height=chart_height,
                            title_x=0.5,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_trends, use_container_width=True)
                    else:
                        st.info("Please select at least one variable to display trends")
            
            # Rolling correlation analysis
            if pfad_col and len(df_clean) > rolling_window:
                st.subheader("🔄 Dynamic Correlation Analysis")
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    st.write(f"**Rolling Window**: {rolling_window} months")
                    correlation_vars = st.multiselect(
                        "Variables for rolling correlation:",
                        options=pfad_correlations.abs().nlargest(5).index.tolist(),
                        default=pfad_correlations.abs().nlargest(3).index.tolist(),
                        help="Select variables to analyze correlation changes over time"
                    )
                
                with col1:
                    if correlation_vars:
                        fig_rolling = go.Figure()
                        
                        for var in correlation_vars:
                            rolling_corr = df_clean[pfad_col].rolling(window=rolling_window).corr(df_clean[var])
                            fig_rolling.add_trace(go.Scatter(
                                x=rolling_corr.index,
                                y=rolling_corr,
                                mode='lines',
                                name=var,
                                line=dict(width=3),
                                hovertemplate=f"<b>{var}</b><br>Date: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>"
                            ))
                        
                        # Add reference lines with enhanced styling
                        fig_rolling.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.5,
                                            annotation_text="Neutral", annotation_position="bottom right")
                        fig_rolling.add_hline(y=0.5, line_dash="dash", line_color="orange", opacity=0.7,
                                            annotation_text="Moderate", annotation_position="bottom right")
                        fig_rolling.add_hline(y=0.8, line_dash="dot", line_color="green", opacity=0.7,
                                            annotation_text="Strong", annotation_position="bottom right")
                        fig_rolling.add_hline(y=-0.5, line_dash="dash", line_color="orange", opacity=0.7)
                        fig_rolling.add_hline(y=-0.8, line_dash="dot", line_color="green", opacity=0.7)
                        
                        fig_rolling.update_layout(
                            title=f"🔄 {rolling_window}-Month Rolling Correlations with {pfad_col}",
                            xaxis_title="Date",
                            yaxis_title="Correlation Coefficient",
                            height=chart_height,
                            yaxis=dict(range=[-1, 1]),
                            title_x=0.5,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_rolling, use_container_width=True)
                        
                        # Correlation stability analysis
                        st.subheader("⚖️ Correlation Stability Metrics")
                        stability_data = []
                        
                        for var in correlation_vars:
                            rolling_corr = df_clean[pfad_col].rolling(window=rolling_window).corr(df_clean[var])
                            stability_data.append({
                                'Variable': var,
                                'Current Correlation': f"{pfad_correlations[var]:.3f}",
                                'Average Rolling': f"{rolling_corr.mean():.3f}",
                                'Std Deviation': f"{rolling_corr.std():.3f}",
                                'Stability Score': f"{(1 - rolling_corr.std()):.3f}",
                                'Trend': "📈 Increasing" if rolling_corr.iloc[-1] > rolling_corr.iloc[-rolling_window//2] else "📉 Decreasing"
                            })
                        
                        stability_df = pd.DataFrame(stability_data)
                        st.dataframe(stability_df, use_container_width=True)
                    else:
                        st.info("Please select variables to analyze rolling correlations")
            else:
                st.info("Rolling correlation analysis requires more data points or a smaller window size")
        else:
            st.info("📅 Time series analysis requires date information in your dataset")
    
    with tab4:
        if show_insights and pfad_correlations is not None:
            st.header("🤖 AI-Powered Business Intelligence")
            
            # Generate and display insights
            with st.spinner("🧠 Generating AI insights..."):
                insights = generate_ai_insights(df_clean, pfad_correlations, pfad_col)
            
            st.markdown(f"""
            <div class="insight-box">
            {insights.replace('**', '<b>').replace('**', '</b>').replace('\n', '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            # Additional analytics
            st.subheader("🎯 Advanced Analytics Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Market efficiency gauge
                predictability = pfad_correlations.abs().max() * 100
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = predictability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Market Predictability Score"},
                    delta = {'reference': 70},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Risk-return scatter
                if len(pfad_correlations) > 2:
                    risk_return_data = []
                    for var in pfad_correlations.index:
                        if var in df_clean.columns:
                            volatility = df_clean[var].std() / df_clean[var].mean() * 100
                            correlation = abs(pfad_correlations[var])
                            risk_return_data.append({
                                'Variable': var,
                                'Volatility': volatility,
                                'Correlation': correlation
                            })
                    
                    risk_df = pd.DataFrame(risk_return_data)
                    
                    fig_risk = px.scatter(
                        risk_df,
                        x='Volatility',
                        y='Correlation',
                        text='Variable',
                        title="Risk vs Predictive Power",
                        labels={'Volatility': 'Volatility (%)', 'Correlation': 'Absolute Correlation'},
                        color='Correlation',
                        size='Correlation',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig_risk.update_traces(textposition='top center')
                    fig_risk.update_layout(height=300, title_x=0.5)
                    st.plotly_chart(fig_risk, use_container_width=True)
            
            # Procurement strategy recommendations
            st.subheader("📋 Strategic Action Plan")
            
            action_items = []
            
            # Based on correlation strength
            strong_predictors = pfad_correlations[pfad_correlations.abs() > 0.7]
            if len(strong_predictors) > 0:
                action_items.extend([
                    f"🎯 **Immediate Priority**: Set up real-time monitoring for {strong_predictors.abs().idxmax()}",
                    f"⚡ **Alert System**: Configure alerts for ±5% movements in top 3 correlates",
                    f"📊 **Dashboard**: Create executive dashboard tracking {len(strong_predictors)} primary indicators"
                ])
            
            # Based on volatility
            volatility = df_clean[pfad_col].std() / df_clean[pfad_col].mean() * 100
            if volatility > 15:
                action_items.extend([
                    "⚠️ **High Volatility Detected**: Implement hedging strategies",
                    "🔄 **Procurement Frequency**: Consider more frequent, smaller purchases",
                    "💰 **Budget Planning**: Increase contingency reserves by 20%"
                ])
            elif volatility < 8:
                action_items.extend([
                    "✅ **Stable Market**: Opportunity for bulk purchasing strategies",
                    "📈 **Long-term Contracts**: Consider extended contract periods",
                    "🎯 **Precision Timing**: Focus on optimal entry points"
                ])
            
            # Based on predictability
            predictability = pfad_correlations.abs().max() * 100
            if predictability > 75:
                action_items.extend([
                    "🤖 **AI Implementation**: Deploy machine learning forecasting models",
                    "📅 **Procurement Calendar**: Develop algorithm-based purchase scheduling",
                    "🎯 **Optimization**: Target 10-15% cost reduction through timing"
                ])
            
            # Timeline-based recommendations
            action_items.extend([
                "📊 **Week 1**: Implement monitoring dashboard for top correlates",
                "📈 **Month 1**: Establish procurement thresholds and alert systems",
                "🔄 **Quarter 1**: Deploy predictive models and optimization strategies",
                "📋 **Ongoing**: Monthly correlation stability reviews and strategy adjustments"
            ])
            
            for i, item in enumerate(action_items, 1):
                st.markdown(f"{i}. {item}")
        
        else:
            st.info("Enable 'Generate AI Insights' in the sidebar to see detailed business intelligence")

    # Download and sharing section
    st.markdown("---")
    st.header("💾 Export & Sharing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Excel Report", type="primary"):
            with st.spinner("📝 Creating comprehensive report..."):
                # Create Excel report
                import io
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Main correlation matrix
                    correlation_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
                    
                    # PFAD specific analysis
                    if pfad_correlations is not None:
                        pfad_analysis = pd.DataFrame({
                            'Variable': pfad_correlations.index,
                            'Correlation': pfad_correlations.values,
                            'Abs_Correlation': pfad_correlations.abs().values,
                            'P_Value': [p_values.get(var, 1.0) for var in pfad_correlations.index],
                            'Strength': [
                                'Very Strong' if abs(corr) > 0.7 else
                                'Strong' if abs(corr) > 0.5 else
                                'Moderate' if abs(corr) > 0.3 else 'Weak'
                                for corr in pfad_correlations.values
                            ]
                        })
                        pfad_analysis.to_excel(writer, sheet_name='PFAD_Analysis', index=False)
                    
                    # Clean dataset
                    df_clean.to_excel(writer, sheet_name='Clean_Data')
                    
                    # Summary statistics
                    summary_stats = df_clean.describe()
                    summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
                
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"PFAD_Analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    with col2:
        if st.button("📱 Share Dashboard Link"):
            st.info("🌐 **Public URL**: https://your-app.streamlit.app")
            st.code("Share this link with your team for real-time access!")
    
    with col3:
        if st.button("📋 Copy Insights"):
            if pfad_correlations is not None:
                insights_text = generate_ai_insights(df_clean, pfad_correlations, pfad_col)
                st.text_area(
                    "Copy the insights below:",
                    value=insights_text,
                    height=200,
                    help="Select all and copy to share insights"
                )

else:
    # Welcome screen with enhanced design
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 20px; margin: 2rem 0;">
        <h2 style="color: #2c3e50; margin-bottom: 2rem;">📊 Welcome to PFAD Procurement Analytics</h2>
        <p style="font-size: 1.3em; color: #34495e; margin-bottom: 2rem;">
            Transform your procurement strategy with AI-powered correlation analysis
        </p>
        
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; margin: 2rem 0;">
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🌡️ Interactive Analysis</h3>
                <p>Dynamic correlation heatmaps with real-time insights</p>
            </div>
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🤖 AI-Powered</h3>
                <p>Automated business insights and recommendations</p>
            </div>
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">☁️ Cloud-Based</h3>
                <p>Access anywhere, share with your team instantly</p>
            </div>
        </div>
        
        <p style="color: #7f8c8d; font-size: 1.1em;">
            Upload your Excel file using the sidebar to begin your analysis journey
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data format guide
    with st.expander("📋 Data Format Requirements", expanded=False):
        st.markdown("""
        ### 📊 Expected Excel File Structure:
        
        **File Requirements:**
        - **Format**: .xlsx or .xls
        - **Sheet Name**: Sheet1 (default)
        - **Size Limit**: 200MB maximum
        
        **Column Examples:**
        """)
        
        sample_data = pd.DataFrame({
            'Date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01'],
            'Imported PFAD - CIF in INR / ton (IA)': [95000, 97000, 94000, 96500],
            'CPO BMD Price (MYR)': [2800, 2850, 2750, 2820],
            'Malaysia FOB USD': [750, 765, 740, 758],
            'USD/INR': [74.5, 75.2, 74.8, 75.0],
            'USD/MYR': [4.15, 4.18, 4.12, 4.16],
            'Brent crude (USD)': [65, 68, 63, 66],
            'CPO Volume': [15000, 18000, 16000, 17200]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown("""
        **Important Notes:**
        - ✅ Date column should be in standard date format
        - ✅ Numeric columns should contain only numbers
        - ✅ Column names can vary (flexible matching)
        - ✅ Missing values are automatically handled
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #7f8c8d;">
    <p><strong>PFAD Procurement Analytics Dashboard</strong> | Powered by Streamlit Cloud ☁️</p>
    <p>🚀 Advanced correlation analysis • 🤖 AI-driven insights • 📊 Real-time visualization</p>
    <p style="font-size: 0.9em;">Made with ❤️ for strategic procurement decisions</p>
</div>
""", unsafe_allow_html=True)
