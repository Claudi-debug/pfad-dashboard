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

if uploaded_file:
    try:
        # Load data with progress
        with st.spinner("📊 Loading and processing your data..."):
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Successfully loaded {len(df):,} records from {uploaded_file.name}")
        
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview", 
            "🌡️ Correlation Analysis", 
            "🎯 PFAD Insights", 
            "📈 Advanced Statistics",
            "🔬 Statistical Tests"
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
                
                # Correlation distribution analysis
                st.subheader("📈 Correlation Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of correlations
                    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                    
                    fig_hist = px.histogram(
                        x=corr_values,
                        nbins=20,
                        title="Distribution of Correlation Coefficients",
                        labels={'x': 'Correlation Coefficient', 'y': 'Frequency'}
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Correlation strength breakdown
                    strong_corr = sum(abs(corr_values) > 0.7)
                    moderate_corr = sum((abs(corr_values) > 0.3) & (abs(corr_values) <= 0.7))
                    weak_corr = sum(abs(corr_values) <= 0.3)
                    
                    breakdown_data = pd.DataFrame({
                        'Strength': ['Strong (>0.7)', 'Moderate (0.3-0.7)', 'Weak (≤0.3)'],
                        'Count': [strong_corr, moderate_corr, weak_corr],
                        'Percentage': [strong_corr/len(corr_values)*100, 
                                     moderate_corr/len(corr_values)*100, 
                                     weak_corr/len(corr_values)*100]
                    })
                    
                    fig_pie = px.pie(
                        breakdown_data, 
                        values='Count', 
                        names='Strength',
                        title="Correlation Strength Distribution"
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Statistical significance matrix
                if show_p_values:
                    st.subheader("🔬 Statistical Significance Matrix")
                    
                    # Create p-value matrix
                    p_value_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
                    
                    for i in corr_matrix.index:
                        for j in corr_matrix.columns:
                            if i != j:
                                stats_result = calculate_correlation_stats(df[i], df[j], confidence_level)
                                if stats_result:
                                    p_value_matrix.loc[i, j] = stats_result['p_value']
                            else:
                                p_value_matrix.loc[i, j] = 0
                    
                    # Convert to numeric
                    p_value_matrix = p_value_matrix.astype(float)
                    
                    # Create significance heatmap
                    significance_matrix = p_value_matrix < 0.05
                    
                    fig_sig = px.imshow(
                        significance_matrix.astype(int),
                        text_auto=False,
                        aspect="auto",
                        title="Statistical Significance Matrix (p < 0.05)",
                        color_continuous_scale=["white", "green"],
                        labels=dict(color="Significant")
                    )
                    
                    fig_sig.update_layout(height=chart_height//2, title_x=0.5)
                    st.plotly_chart(fig_sig, use_container_width=True)
            
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
                                'Sample Size': stats_result['sample_size'],
                                'Statistical Power': f"{stats_result['statistical_power']:.3f}"
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
                            if row['Effect Size'] == 'Very Large':
                                color = '#27ae60'
                            elif row['Effect Size'] == 'Large':
                                color = '#2ecc71'
                            elif row['Effect Size'] == 'Medium':
                                color = '#f39c12'
                            elif row['Effect Size'] == 'Small':
                                color = '#e67e22'
                            else:
                                color = '#e74c3c'
                            
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
                                            f"Effect Size: {row['Effect Size']}<br>" +
                                            f"P-value: {row['P-Value']}<extra></extra>"
                            ))
                            
                            # Add confidence interval error bars
                            if show_confidence_intervals:
                                fig_bar.add_trace(go.Scatter(
                                    x=[ci_lower, ci_upper],
                                    y=[row['Variable'], row['Variable']],
                                    mode='lines',
                                    line=dict(color='black', width=2),
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                        
                        fig_bar.update_layout(
                            title=f"🎯 {pfad_col} Correlations with {confidence_level*100:.0f}% Confidence Intervals",
                            xaxis_title="Correlation Coefficient",
                            height=chart_height,
                            title_x=0.5
                        )
                        
                        # Add reference lines
                        fig_bar.add_vline(x=0, line_dash="solid", line_color="black", opacity=0.5)
                        fig_bar.add_vline(x=0.3, line_dash="dash", line_color="orange", opacity=0.7)
                        fig_bar.add_vline(x=0.7, line_dash="dot", line_color="green", opacity=0.7)
                        fig_bar.add_vline(x=-0.3, line_dash="dash", line_color="orange", opacity=0.7)
                        fig_bar.add_vline(x=-0.7, line_dash="dot", line_color="green", opacity=0.7)
                        
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Statistical interpretation
                        st.markdown("""
                        <div class="statistical-box">
                        <h4>📊 Statistical Interpretation Guide</h4>
                        """, unsafe_allow_html=True)
                        
                        strong_significant = len([s for s in detailed_stats if float(s['P-Value']) < 0.05 and abs(float(s['Correlation'])) > 0.7])
                        total_significant = len([s for s in detailed_stats if float(s['P-Value']) < 0.05])
                        
                        st.write(f"• **Statistically Significant Correlations**: {total_significant}/{len(detailed_stats)} variables")
                        st.write(f"• **Strong & Significant**: {strong_significant} variables (|r| > 0.7, p < 0.05)")
                        st.write(f"• **Confidence Level**: {confidence_level*100:.0f}% confidence intervals shown")
                        st.write("• **Effect Size**: Cohen's guidelines for correlation interpretation")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Business recommendations based on statistical analysis
                        significant_strong = [s for s in detailed_stats if float(s['P-Value']) < 0.05 and abs(float(s['Correlation'])) > 0.5]
                        
                        if significant_strong:
                            st.markdown("""
                            <div class="insight-box">
                            <h3>🎯 Evidence-Based Procurement Recommendations</h3>
                            """, unsafe_allow_html=True)
                            
                            top_predictor = significant_strong[0]
                            st.write(f"• **Primary Indicator**: {top_predictor['Variable']} (r = {top_predictor['Correlation']}, p = {top_predictor['P-Value']})")
                            st.write(f"• **Statistical Confidence**: {confidence_level*100:.0f}% confidence interval: [{top_predictor['CI Lower']}, {top_predictor['CI Upper']}]")
                            st.write(f"• **Effect Size**: {top_predictor['Effect Size']} relationship")
                            st.write(f"• **Statistical Power**: {top_predictor['Statistical Power']} (reliability of detection)")
                            
                            st.write("\n**Strategic Actions:**")
                            st.write("• Implement real-time monitoring for statistically significant variables")
                            st.write("• Set procurement thresholds based on confidence intervals")
                            st.write("• Focus resources on variables with large effect sizes")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    
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
                
                # Rolling correlation analysis
                st.subheader("📊 Rolling Correlation Stability")
                
                if 'Date' in df.columns or hasattr(df.index, 'strftime'):
                    window_size = st.slider("Rolling Window Size", 6, min(50, len(df)//4), 12)
                    
                    rolling_var = st.selectbox(
                        "Select variable for rolling correlation:",
                        options=pfad_corr.abs().sort_values(ascending=False).index.tolist(),
                        key="rolling_var"
                    )
                    
                    if rolling_var and len(df) > window_size:
                        # Calculate rolling correlation
                        rolling_corr = df[pfad_col].rolling(window=window_size).corr(df[rolling_var])
                        
                        # Calculate rolling statistics
                        rolling_mean = rolling_corr.rolling(window=window_size).mean()
                        rolling_std = rolling_corr.rolling(window=window_size).std()
                        
                        fig_rolling = go.Figure()
                        
                        # Add rolling correlation
                        fig_rolling.add_trace(go.Scatter(
                            x=rolling_corr.index,
                            y=rolling_corr,
                            mode='lines',
                            name='Rolling Correlation',
                            line=dict(width=2, color='blue')
                        ))
                        
                        # Add confidence bands
                        upper_band = rolling_mean + 1.96 * rolling_std
                        lower_band = rolling_mean - 1.96 * rolling_std
                        
                        fig_rolling.add_trace(go.Scatter(
                            x=rolling_corr.index,
                            y=upper_band,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            name='Upper CI'
                        ))
                        
                        fig_rolling.add_trace(go.Scatter(
                            x=rolling_corr.index,
                            y=lower_band,
                            mode='lines',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(0,100,80,0.2)',
                            showlegend=False,
                            name='Lower CI'
                        ))
                        
                        # Add overall mean
                        overall_corr = pfad_corr[rolling_var]
                        fig_rolling.add_hline(
                            y=overall_corr, 
                            line_dash="dash", 
                            line_color="red",
                            annotation_text=f"Overall: {overall_corr:.3f}"
                        )
                        
                        fig_rolling.update_layout(
                            title=f"Rolling Correlation: {pfad_col} vs {rolling_var} (Window: {window_size})",
                            xaxis_title="Time Period",
                            yaxis_title="Correlation Coefficient",
                            height=500,
                            yaxis=dict(range=[-1, 1])
                        )
                        
                        st.plotly_chart(fig_rolling, use_container_width=True)
                        
                        # Stability metrics
                        stability_score = 1 - (rolling_corr.std() / abs(rolling_corr.mean())) if rolling_corr.mean() != 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Correlation Stability", f"{stability_score:.3f}", help="Higher values indicate more stable relationships")
                        with col2:
                            st.metric("Std Deviation", f"{rolling_corr.std():.3f}", help="Lower values indicate less volatility")
                        with col3:
                            trend_slope = np.polyfit(range(len(rolling_corr.dropna())), rolling_corr.dropna(), 1)[0]
                            st.metric("Trend Slope", f"{trend_slope:.4f}", help="Positive values indicate strengthening correlation")
                
                else:
                    st.info("Date information needed for rolling correlation analysis")
                
                # Outlier impact analysis
                st.subheader("🔍 Outlier Impact Analysis")
                
                outlier_var = st.selectbox(
                    "Select variable for outlier analysis:",
                    options=[pfad_col] + list(pfad_corr.abs().sort_values(ascending=False).index[:5]),
                    key="outlier_var"
                )
                
                if outlier_var:
                    outlier_result = detect_outliers(df[outlier_var], method='iqr')
                    
                    if outlier_result:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Outlier visualization
                            fig_outlier = go.Figure()
                            
                            # Box plot
                            fig_outlier.add_trace(go.Box(
                                y=df[outlier_var],
                                name=outlier_var,
                                boxpoints='outliers'
                            ))
                            
                            fig_outlier.update_layout(
                                title=f"Outlier Detection: {outlier_var}",
                                height=400
                            )
                            
                            st.plotly_chart(fig_outlier, use_container_width=True)
                        
                        with col2:
                            st.markdown(f"""
                            **Outlier Analysis Results:**
                            - **Total Outliers**: {outlier_result['outlier_count']}
                            - **Percentage**: {outlier_result['outlier_percentage']:.2f}%
                            - **Detection Method**: {outlier_result['method'].upper()}
                            
                            **Impact Assessment:**
                            """)
                            
                            if outlier_result['outlier_count'] > 0:
                                # Calculate correlation with and without outliers
                                if outlier_var != pfad_col and pfad_col:
                                    df_no_outliers = df.drop(outlier_result['outlier_indices'])
                                    
                                    corr_with = df[pfad_col].corr(df[outlier_var])
                                    corr_without = df_no_outliers[pfad_col].corr(df_no_outliers[outlier_var])
                                    
                                    impact = abs(corr_with - corr_without)
                                    
                                    st.write(f"- **Correlation with outliers**: {corr_with:.3f}")
                                    st.write(f"- **Correlation without outliers**: {corr_without:.3f}")
                                    st.write(f"- **Impact magnitude**: {impact:.3f}")
                                    
                                    if impact > 0.1:
                                        st.warning("⚠️ Outliers significantly affect correlation")
                                    else:
                                        st.success("✅ Outliers have minimal impact")
            
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
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Q-Q plot
                                data_clean = df[normality_var].dropna()
                                
                                fig_qq = go.Figure()
                                
                                # Generate theoretical quantiles
                                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data_clean)))
                                sample_quantiles = np.sort(data_clean)
                                
                                fig_qq.add_trace(go.Scatter(
                                    x=theoretical_quantiles,
                                    y=sample_quantiles,
                                    mode='markers',
                                    name='Data Points'
                                ))
                                
                                # Add reference line
                                min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
                                fig_qq.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val * data_clean.std() + data_clean.mean(), 
                                       max_val * data_clean.std() + data_clean.mean()],
                                    mode='lines',
                                    name='Reference Line',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                fig_qq.update_layout(
                                    title="Q-Q Plot (Normal Distribution)",
                                    xaxis_title="Theoretical Quantiles",
                                    yaxis_title="Sample Quantiles",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_qq, use_container_width=True)
                            
                            with col2:
                                # Distribution histogram with normal overlay
                                fig_dist = px.histogram(
                                    df, 
                                    x=normality_var, 
                                    nbins=30,
                                    title="Distribution vs Normal Curve",
                                    density=True
                                )
                                
                                # Add normal distribution overlay
                                x_norm = np.linspace(data_clean.min(), data_clean.max(), 100)
                                y_norm = stats.norm.pdf(x_norm, data_clean.mean(), data_clean.std())
                                
                                fig_dist.add_trace(go.Scatter(
                                    x=x_norm,
                                    y=y_norm,
                                    mode='lines',
                                    name='Normal Distribution',
                                    line=dict(color='red', width=3)
                                ))
                                
                                fig_dist.update_layout(height=400)
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # Normality test results
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
                        
                        # Color coding for significance
                        def color_significance(val):
                            if val == 'Yes':
                                return 'background-color: #d5f4e6'
                            else:
                                return 'background-color: #f8d7da'
                        
                        styled_sig_df = sig_df.style.applymap(color_significance, subset=['Significant'])
                        st.dataframe(styled_sig_df, use_container_width=True)
                        
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
                
                # Power analysis
                st.subheader("⚡ Statistical Power Analysis")
                
                if pfad_col:
                    power_analysis_data = []
                    
                    for var in pfad_corr.index:
                        stats_result = calculate_correlation_stats(df[pfad_col], df[var], confidence_level)
                        
                        if stats_result:
                            power_analysis_data.append({
                                'Variable': var,
                                'Correlation': abs(stats_result['correlation']),
                                'Sample Size': stats_result['sample_size'],
                                'Statistical Power': stats_result['statistical_power'],
                                'Power Category': 'High (>0.8)' if stats_result['statistical_power'] > 0.8 else 
                                                'Medium (0.5-0.8)' if stats_result['statistical_power'] > 0.5 else 'Low (<0.5)'
                            })
                    
                    if power_analysis_data:
                        power_df = pd.DataFrame(power_analysis_data)
                        
                        # Power vs correlation scatter
                        fig_power = px.scatter(
                            power_df,
                            x='Correlation',
                            y='Statistical Power',
                            size='Sample Size',
                            color='Power Category',
                            hover_data=['Variable'],
                            title="Statistical Power vs Correlation Strength"
                        )
                        
                        fig_power.add_hline(y=0.8, line_dash="dash", line_color="green", 
                                          annotation_text="Adequate Power (0.8)")
                        fig_power.add_hline(y=0.5, line_dash="dash", line_color="orange",
                                          annotation_text="Moderate Power (0.5)")
                        
                        fig_power.update_layout(height=500)
                        st.plotly_chart(fig_power, use_container_width=True)
                        
                        # Power analysis interpretation
                        high_power = len([p for p in power_analysis_data if p['Statistical Power'] > 0.8])
                        low_power = len([p for p in power_analysis_data if p['Statistical Power'] < 0.5])
                        
                        st.markdown(f"""
                        <div class="statistical-box">
                        <h4>⚡ Power Analysis Interpretation</h4>
                        <p><strong>High Power Variables:</strong> {high_power} (reliable detection of true effects)</p>
                        <p><strong>Low Power Variables:</strong> {low_power} (may miss true effects)</p>
                        <p><strong>Recommendation:</strong> Focus on high-power relationships for reliable decision making</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            else:
                st.info("Numeric variables needed for statistical testing")
    
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
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🔬 Statistical Tests</h3>
                <p>Confidence intervals, p-values, and significance testing</p>
            </div>
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">🔄 Bootstrap Analysis</h3>
                <p>Robust correlation estimates with bootstrap sampling</p>
            </div>
            <div style="flex: 1; min-width: 250px; margin: 1rem; padding: 1.5rem; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea;">⚡ Power Analysis</h3>
                <p>Statistical power and effect size interpretation</p>
            </div>
        </div>
        
        <p style="color: #7f8c8d; font-size: 1.1em;">
            Upload your Excel file to access advanced statistical analysis features
        </p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #7f8c8d;">
    <p><strong>PFAD Advanced Statistical Analytics</strong> | Professional Statistical Analysis Platform</p>
    <p>🔬 Statistical rigor • 📊 Professional insights • ⚡ Evidence-based decisions</p>
</div>
""", unsafe_allow_html=True)
