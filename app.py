import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 PFAD Analytics Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])

if uploaded_file:
    try:
        # Load data
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} records")
        
        # Show data
        st.subheader("📋 Your Data")
        st.dataframe(df.head())
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Calculate correlations
            corr_matrix = df[numeric_cols].corr()
            
            # Show correlation heatmap
            st.subheader("🌡️ Correlation Matrix")
            fig = px.imshow(corr_matrix, text_auto=True, title="Correlations")
            st.plotly_chart(fig)
            
            # Find PFAD column
            pfad_col = None
            for col in numeric_cols:
                if 'PFAD' in str(col).upper():
                    pfad_col = col
                    break
            
            if pfad_col:
                # PFAD correlations
                pfad_corr = corr_matrix[pfad_col].drop(pfad_col)
                
                st.subheader(f"🎯 {pfad_col} Correlations")
                
                # Simple bar chart
                fig2 = px.bar(
                    x=pfad_corr.values,
                    y=pfad_corr.index,
                    orientation='h',
                    title="PFAD Correlations"
                )
                st.plotly_chart(fig2)
                
                # Show top correlations
                st.subheader("📈 Top Correlations")
                top_corr = pfad_corr.abs().sort_values(ascending=False).head(5)
                for var, corr in top_corr.items():
                    original_corr = pfad_corr[var]
                    st.write(f"**{var}**: {original_corr:.3f}")
            
            else:
                st.warning("No PFAD column found")
                st.write("Available columns:", numeric_cols)
        
        else:
            st.error("Need at least 2 numeric columns for correlation analysis")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please check your Excel file format")

else:
    st.info("👆 Upload your Excel file to start analysis")
