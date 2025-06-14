import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
    <p style="font-size: 1.2em; margin-bottom: 0;">AI-Powered Correlation Analysis for Strategic Decision Making</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Advanced Analytics • Real-time Insights • Strategic Intelligence</p>
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

show_all_correlations = st.sidebar.checkbox("Show All Variables", value=True)
chart_height = st.sidebar.slider("Chart Height", 400, 800, 500)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Upload Excel files with PFAD data and market variables for best results")

if uploaded_file:
    try:
        # Load data with progress
        with st.spinner("📊 Loading and processing your data..."):
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Successfully loaded {len(df):,} records from {uploaded_file.name}")
        
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
                label="📊 Variables",
                value=f"{len(df.columns)}",
                help="Total number of columns in your data"
            )
        
        with col3:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            st.metric(
                label="🔢 Numeric Variables",
                value=f"{len(numeric_cols)}",
                help="Variables available for correlation analysis"
            )
        
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric(
                label="📋 Data Quality",
                value=f"{100-missing_pct:.1f}%",
                help="Percentage of complete data"
            )
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🌡️ Correlation Analysis", "🎯 PFAD Insights"])
        
        with tab1:
            st.header("📋 Dataset Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.subheader("📈 Data Summary")
                if len(numeric_cols) > 0:
                    summary_stats = df[numeric_cols].describe()
                    st.dataframe(summary_stats, use_container_width=True)
                else:
                    st.info("No numeric columns found for summary statistics")
            
            # Column information
            st.subheader("📋 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab2:
            st.header("🌡️ Correlation Analysis")
            
            if len(numeric_cols) > 1:
                # Calculate correlations
                corr_matrix = df[numeric_cols].corr()
                
                # Enhanced correlation heatmap
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="📊 Interactive Correlation Matrix",
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
                
                # Correlation insights
                st.subheader("🔍 Correlation Insights")
                
                # Get correlation pairs above threshold
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        var1 = corr_matrix.columns[i]
                        var2 = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) >= correlation_threshold:
                            corr_pairs.append({
                                'Variable 1': var1,
                                'Variable 2': var2,
                                'Correlation': corr_val,
                                'Strength': 'Very Strong' if abs(corr_val) > 0.8 else 'Strong' if abs(corr_val) > 0.6 else 'Moderate'
                            })
                
                if corr_pairs:
                    corr_df = pd.DataFrame(corr_pairs)
                    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                    
                    st.write(f"**Found {len(corr_pairs)} correlations above {correlation_threshold} threshold:**")
                    st.dataframe(corr_df, use_container_width=True)
                else:
                    st.info(f"No correlations found above {correlation_threshold} threshold. Try lowering the threshold.")
            
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis")
        
        with tab3:
            st.header("🎯 PFAD-Specific Analysis")
            
            # Find PFAD column
            pfad_col = None
            for col in numeric_cols:
                if 'PFAD' in str(col).upper():
                    pfad_col = col
                    break
            
            if pfad_col:
                st.success(f"📊 Found PFAD column: **{pfad_col}**")
                
                if len(numeric_cols) > 1:
                    # PFAD correlations
                    pfad_corr = corr_matrix[pfad_col].drop(pfad_col).sort_values(key=abs, ascending=False)
                    
                    # Enhanced bar chart
                    colors = ['#27ae60' if abs(x) > 0.7 else '#f39c12' if abs(x) > 0.4 else '#e74c3c' for x in pfad_corr.values]
                    
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=pfad_corr.values,
                            y=pfad_corr.index,
                            orientation='h',
                            marker_color=colors,
                            text=[f"{val:.3f}" for val in pfad_corr.values],
                            textposition='auto',
                            hovertemplate="<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>"
                        )
                    ])
                    
                    fig_bar.update_layout(
                        title=f"🎯 {pfad_col} Correlations (Ranked by Strength)",
                        xaxis_title="Correlation Coefficient",
                        height=chart_height,
                        title_x=0.5
                    )
                    
                    # Add reference lines
                    fig_bar.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.5)
                    fig_bar.add_vline(x=0.5, line_dash="dash", line_color="orange", opacity=0.7)
                    fig_bar.add_vline(x=0.8, line_dash="dot", line_color="green", opacity=0.7)
                    fig_bar.add_vline(x=-0.5, line_dash="dash", line_color="orange", opacity=0.7)
                    fig_bar.add_vline(x=-0.8, line_dash="dot", line_color="green", opacity=0.7)
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Key insights
                    st.subheader("💡 Key Insights")
                    
                    strong_corr = pfad_corr[abs(pfad_corr) > 0.7]
                    moderate_corr = pfad_corr[(abs(pfad_corr) > 0.4) & (abs(pfad_corr) <= 0.7)]
                    weak_corr = pfad_corr[abs(pfad_corr) <= 0.4]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="🔥 Strong Correlations",
                            value=len(strong_corr),
                            help="Variables with |correlation| > 0.7"
                        )
                        if len(strong_corr) > 0:
                            st.write("**Variables:**")
                            for var, corr in strong_corr.items():
                                st.write(f"• {var}: {corr:.3f}")
                    
                    with col2:
                        st.metric(
                            label="🔶 Moderate Correlations",
                            value=len(moderate_corr),
                            help="Variables with 0.4 < |correlation| ≤ 0.7"
                        )
                        if len(moderate_corr) > 0:
                            st.write("**Variables:**")
                            for var, corr in moderate_corr.head(3).items():
                                st.write(f"• {var}: {corr:.3f}")
                    
                    with col3:
                        st.metric(
                            label="⚪ Weak Correlations",
                            value=len(weak_corr),
                            help="Variables with |correlation| ≤ 0.4"
                        )
                    
                    # Business recommendations
                    if len(strong_corr) > 0 or len(moderate_corr) > 0:
                        st.markdown("""
                        <div class="insight-box">
                        <h3>🎯 Procurement Recommendations</h3>
                        """, unsafe_allow_html=True)
                        
                        if len(strong_corr) > 0:
                            top_var = strong_corr.abs().idxmax()
                            top_corr = strong_corr[top_var]
                            st.write(f"• **Primary Focus**: Monitor {top_var} closely (correlation: {top_corr:.3f})")
                            st.write("• Set up alerts for significant movements in this variable")
                            
                            if top_corr > 0:
                                st.write("• Rising values typically indicate higher PFAD costs")
                                st.write("• Consider purchasing when this indicator trends downward")
                            else:
                                st.write("• Rising values typically indicate lower PFAD costs")
                                st.write("• Consider purchasing when this indicator trends upward")
                        
                        st.write("• Develop monitoring dashboard for top 3 correlates")
                        st.write("• Establish procurement thresholds based on key indicators")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Interactive scatter plot
                    st.subheader("💫 Relationship Explorer")
                    
                    selected_var = st.selectbox(
                        "Select variable to explore relationship with PFAD:",
                        options=pfad_corr.abs().sort_values(ascending=False).index.tolist(),
                        help="Choose a variable to see detailed relationship analysis"
                    )
                    
                    if selected_var:
                        plot_data = df[[pfad_col, selected_var]].dropna()
                        
                        if len(plot_data) > 0:
                            corr_val = pfad_corr[selected_var]
                            
                            fig_scatter = px.scatter(
                                plot_data,
                                x=selected_var,
                                y=pfad_col,
                                title=f"📈 {pfad_col} vs {selected_var} (r = {corr_val:.3f})",
                                trendline="ols",
                                hover_data={selected_var: ':.2f', pfad_col: ':.2f'}
                            )
                            
                            fig_scatter.update_layout(height=500, title_x=0.5)
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            # Relationship interpretation
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
                            st.warning("No data available for this relationship")
                else:
                    st.info("Need more numeric variables for PFAD correlation analysis")
            
            else:
                st.warning("🔍 No PFAD column found in your data")
                st.write("**Available numeric columns:**")
                for col in numeric_cols:
                    st.write(f"• {col}")
                st.info("💡 Make sure your PFAD column contains 'PFAD' in the name")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.write("**Troubleshooting tips:**")
        st.write("• Ensure your file is a valid Excel format (.xlsx or .xls)")
        st.write("• Check that your data contains numeric values")
        st.write("• Verify that column names don't contain special characters")

else:
    # Enhanced welcome screen
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
                <h3 style="color: #667eea;">🎯 PFAD-Specific</h3>
                <p>Specialized analysis for procurement optimization</p>
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

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #7f8c8d;">
    <p><strong>PFAD Procurement Analytics Dashboard</strong> | Enhanced with Advanced Features</p>
    <p>🚀 Interactive analysis • 🎯 Strategic insights • 📊 Professional visualization</p>
</div>
""", unsafe_allow_html=True)
