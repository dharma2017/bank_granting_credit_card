# ============================================================================
# STREAMLIT APP FOR CREDIT CARD DEFAULT PREDICTION
# Save this as: streamlit_app.py
# Run with: streamlit run streamlit_app.py
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load all trained model artifacts"""
    artifacts = {}
    required_files = {
        'model': 'best_credit_model.pkl',
        'scaler': 'scaler.pkl',
        'feature_names': 'feature_names.pkl',
        'model_info': 'model_info.pkl',
        'feature_config': 'feature_config.pkl'
    }
    
    optional_files = {
        'confusion_matrix': 'confusion_matrix.pkl',
        'split_info': 'split_info.pkl',
        'model_comparison': 'all_models_comparison.csv'
    }
    
    try:
        # Load required files
        for key, filename in required_files.items():
            if not os.path.exists(filename):
                st.error(f"❌ Required file not found: {filename}")
                return None
            
            if filename.endswith('.pkl'):
                with open(filename, 'rb') as f:
                    artifacts[key] = pickle.load(f) if key != 'model' and key != 'scaler' else joblib.load(filename)
            
        # Load optional files (won't fail if missing)
        for key, filename in optional_files.items():
            try:
                if os.path.exists(filename):
                    if filename.endswith('.pkl'):
                        with open(filename, 'rb') as f:
                            artifacts[key] = pickle.load(f)
                    elif filename.endswith('.csv'):
                        artifacts[key] = pd.read_csv(filename)
            except Exception as e:
                st.warning(f"⚠️ Optional file {filename} could not be loaded: {str(e)}")
        
        return artifacts
        
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {str(e)}")
        return None

# Load all artifacts
artifacts = load_model_artifacts()

if artifacts is None:
    st.stop()

# Extract artifacts
model = artifacts.get('model')
scaler = artifacts.get('scaler')
feature_names = artifacts.get('feature_names')
model_info = artifacts.get('model_info')
feature_config = artifacts.get('feature_config')
confusion_matrix_data = artifacts.get('confusion_matrix')
split_info = artifacts.get('split_info')
model_comparison = artifacts.get('model_comparison')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def engineer_features(data, feature_config):
    """Apply feature engineering to input data"""
    data = data.copy()
    
    # Create engineered features
    payment_cols = feature_config['payment_columns']
    bill_cols = feature_config['bill_columns']
    pay_amt_cols = feature_config['payment_amount_columns']
    
    data['AVG_PAYMENT_STATUS'] = data[payment_cols].mean(axis=1)
    data['MAX_PAYMENT_DELAY'] = data[payment_cols].max(axis=1)
    data['TOTAL_BILL_AMT'] = data[bill_cols].sum(axis=1)
    data['TOTAL_PAY_AMT'] = data[pay_amt_cols].sum(axis=1)
    data['UTILIZATION_RATIO'] = data['TOTAL_BILL_AMT'] / (data['LIMIT_BAL'] + 1)
    
    return data

def make_prediction(input_data, model, scaler, feature_names, feature_config):
    """Make prediction for single or batch input"""
    # Engineer features
    input_with_features = engineer_features(input_data, feature_config)
    
    # Ensure columns match training
    input_with_features = input_with_features[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_with_features)
    
    # Make prediction
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    return predictions, probabilities

# ============================================================================
# HEADER
# ============================================================================

st.title("💳 Credit Card Default Prediction System")
st.markdown("### Predict customer creditworthiness to minimize risk and maximize profit")

# Display model info in sidebar
st.sidebar.header("📊 Model Information")
st.sidebar.metric("Model Type", model_info['model_name'])
st.sidebar.metric("Test Accuracy", f"{model_info['test_accuracy']:.2%}")
st.sidebar.metric("F1 Score", f"{model_info['f1_score']:.3f}")
st.sidebar.metric("Precision", f"{model_info['precision']:.2%}")
st.sidebar.metric("Recall", f"{model_info['recall']:.2%}")

# Add overfitting indicator
if model_info.get('overfitting') == 'Yes':
    st.sidebar.warning("⚠️ Overfitting: Yes")
else:
    st.sidebar.success("✅ Overfitting: No")

st.sidebar.info(f"📅 Trained: {model_info['train_date']}")

# Display dataset info if available
if split_info:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Dataset Info")
    st.sidebar.metric("Training Samples", f"{split_info['train_size']:,}")
    st.sidebar.metric("Test Samples", f"{split_info['test_size']:,}")
    st.sidebar.metric("Train Default Rate", f"{split_info['train_default_rate']:.2%}")

# Add download model info button
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export")

# Convert model info to JSON for download
model_info_json = json.dumps(model_info, indent=2, default=str)
st.sidebar.download_button(
    label="Download Model Info",
    data=model_info_json,
    file_name="model_info.json",
    mime="application/json"
)

# ============================================================================
# MAIN APP TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Single Prediction", 
    "📊 Batch Prediction", 
    "📈 Model Performance",
    "🔍 Model Comparison", 
    "💾 Data Management"
])

# ============================================================================
# TAB 1: SINGLE PREDICTION
# ============================================================================

with tab1:
    st.header("Single Customer Prediction")
    st.markdown("Enter customer details to predict default probability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Basic Information")
        limit_bal = st.number_input("Credit Limit (NT$)", 
                                    min_value=0, 
                                    max_value=1000000, 
                                    value=50000,
                                    step=10000,
                                    help="Amount of given credit in NT dollars")
        sex = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        education = st.selectbox("Education", [1, 2, 3, 4], 
                                format_func=lambda x: ["Graduate School", "University", "High School", "Others"][x-1])
        marriage = st.selectbox("Marital Status", [1, 2, 3],
                               format_func=lambda x: ["Married", "Single", "Others"][x-1])
        age = st.slider("Age", 21, 80, 35)
    
    with col2:
        st.subheader("📅 Payment Status (Last 6 Months)")
        st.caption("Payment Status: -1=Pay duly, 1-9=Delay months")
        pay_0 = st.selectbox("September 2005", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        pay_2 = st.selectbox("August 2005", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        pay_3 = st.selectbox("July 2005", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        pay_4 = st.selectbox("June 2005", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        pay_5 = st.selectbox("May 2005", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        pay_6 = st.selectbox("April 2005", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
    
    with col3:
        st.subheader("💰 Bill Amounts (NT$)")
        bill_amt1 = st.number_input("September 2005 Bill", 0, 500000, 20000, 1000)
        bill_amt2 = st.number_input("August 2005 Bill", 0, 500000, 20000, 1000)
        bill_amt3 = st.number_input("July 2005 Bill", 0, 500000, 20000, 1000)
        bill_amt4 = st.number_input("June 2005 Bill", 0, 500000, 20000, 1000)
        bill_amt5 = st.number_input("May 2005 Bill", 0, 500000, 20000, 1000)
        bill_amt6 = st.number_input("April 2005 Bill", 0, 500000, 20000, 1000)
        
        st.subheader("💵 Payment Amounts (NT$)")
        pay_amt1 = st.number_input("September 2005 Payment", 0, 100000, 2000, 500)
        pay_amt2 = st.number_input("August 2005 Payment", 0, 100000, 2000, 500)
        pay_amt3 = st.number_input("July 2005 Payment", 0, 100000, 2000, 500)
        pay_amt4 = st.number_input("June 2005 Payment", 0, 100000, 2000, 500)
        pay_amt5 = st.number_input("May 2005 Payment", 0, 100000, 2000, 500)
        pay_amt6 = st.number_input("April 2005 Payment", 0, 100000, 2000, 500)
    
    # Add buttons side by side
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        predict_button = st.button("🎯 Predict Default Risk", type="primary", use_container_width=True)
    
    with col_btn2:
        save_input_button = st.button("💾 Save Customer Data", use_container_width=True)
    
    # Predict button
    if predict_button:
        # Create input dataframe
        input_data = pd.DataFrame({
            'LIMIT_BAL': [limit_bal],
            'SEX': [sex],
            'EDUCATION': [education],
            'MARRIAGE': [marriage],
            'AGE': [age],
            'PAY_0': [pay_0],
            'PAY_2': [pay_2],
            'PAY_3': [pay_3],
            'PAY_4': [pay_4],
            'PAY_5': [pay_5],
            'PAY_6': [pay_6],
            'BILL_AMT1': [bill_amt1],
            'BILL_AMT2': [bill_amt2],
            'BILL_AMT3': [bill_amt3],
            'BILL_AMT4': [bill_amt4],
            'BILL_AMT5': [bill_amt5],
            'BILL_AMT6': [bill_amt6],
            'PAY_AMT1': [pay_amt1],
            'PAY_AMT2': [pay_amt2],
            'PAY_AMT3': [pay_amt3],
            'PAY_AMT4': [pay_amt4],
            'PAY_AMT5': [pay_amt5],
            'PAY_AMT6': [pay_amt6]
        })
        
        # Make prediction
        predictions, probabilities = make_prediction(input_data, model, scaler, feature_names, feature_config)
        prediction = predictions[0]
        prediction_proba = probabilities[0]
        
        # Display results
        st.markdown("---")
        st.subheader("📋 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ HIGH RISK")
                st.metric("Default Prediction", "YES", delta="High Risk", delta_color="inverse")
            else:
                st.success("✅ LOW RISK")
                st.metric("Default Prediction", "NO", delta="Low Risk", delta_color="normal")
        
        with col2:
            st.metric("Default Probability", f"{prediction_proba[1]:.2%}")
            st.progress(prediction_proba[1])
        
        with col3:
            st.metric("Non-Default Probability", f"{prediction_proba[0]:.2%}")
            st.progress(prediction_proba[0])
        
        # Risk gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction_proba[1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'lightcoral'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if prediction == 1:
            st.warning("""
            **Risk Mitigation Strategies:**
            - 🔴 Consider reducing credit limit
            - 👁️ Implement closer monitoring
            - 🔒 Require additional collateral
            - 📧 Send early payment reminders
            - 💬 Offer financial counseling
            """)
        else:
            st.info("""
            **Customer Retention Strategies:**
            - ✅ Maintain current credit terms
            - ⬆️ Consider credit limit increase
            - 🎁 Offer rewards program
            - 💰 Provide preferential interest rates
            """)
    
    # Save input button
    if save_input_button:
        # Create customer data dictionary
        customer_data = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'LIMIT_BAL': limit_bal,
            'SEX': sex,
            'EDUCATION': education,
            'MARRIAGE': marriage,
            'AGE': age,
            'PAY_0': pay_0,
            'PAY_2': pay_2,
            'PAY_3': pay_3,
            'PAY_4': pay_4,
            'PAY_5': pay_5,
            'PAY_6': pay_6,
            'BILL_AMT1': bill_amt1,
            'BILL_AMT2': bill_amt2,
            'BILL_AMT3': bill_amt3,
            'BILL_AMT4': bill_amt4,
            'BILL_AMT5': bill_amt5,
            'BILL_AMT6': bill_amt6,
            'PAY_AMT1': pay_amt1,
            'PAY_AMT2': pay_amt2,
            'PAY_AMT3': pay_amt3,
            'PAY_AMT4': pay_amt4,
            'PAY_AMT5': pay_amt5,
            'PAY_AMT6': pay_amt6
        }
        
        # Convert to dataframe
        save_df = pd.DataFrame([customer_data])
        
        # Convert to CSV
        csv = save_df.to_csv(index=False)
        
        st.success("✅ Customer data ready for download!")
        st.download_button(
            label="📥 Download Customer Data (CSV)",
            data=csv,
            file_name=f"customer_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ============================================================================
# TAB 2: BATCH PREDICTION
# ============================================================================

with tab2:
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file with customer data for bulk predictions")
    
    # Show expected format
    with st.expander("📋 View Expected CSV Format"):
        st.markdown("""
        Your CSV file should contain the following columns:
        - **Demographics**: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
        - **Payment Status**: PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
        - **Bill Amounts**: BILL_AMT1 through BILL_AMT6
        - **Payment Amounts**: PAY_AMT1 through PAY_AMT6
        
        **Optional**: ID column (will be preserved in results)
        """)
        
        # Create sample template
        sample_template = pd.DataFrame({
            'ID': [1, 2],
            'LIMIT_BAL': [50000, 100000],
            'SEX': [1, 2],
            'EDUCATION': [2, 1],
            'MARRIAGE': [1, 2],
            'AGE': [35, 28],
            'PAY_0': [0, 1],
            'PAY_2': [0, 0],
            'PAY_3': [0, 0],
            'PAY_4': [0, 0],
            'PAY_5': [0, 0],
            'PAY_6': [0, 0],
            'BILL_AMT1': [20000, 30000],
            'BILL_AMT2': [20000, 30000],
            'BILL_AMT3': [20000, 30000],
            'BILL_AMT4': [20000, 30000],
            'BILL_AMT5': [20000, 30000],
            'BILL_AMT6': [20000, 30000],
            'PAY_AMT1': [2000, 3000],
            'PAY_AMT2': [2000, 3000],
            'PAY_AMT3': [2000, 3000],
            'PAY_AMT4': [2000, 3000],
            'PAY_AMT5': [2000, 3000],
            'PAY_AMT6': [2000, 3000]
        })
        
        st.dataframe(sample_template, use_container_width=True)
        
        # Download template button
        template_csv = sample_template.to_csv(index=False)
        st.download_button(
            label="📥 Download Template CSV",
            data=template_csv,
            file_name="batch_prediction_template.csv",
            mime="text/csv"
        )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            batch_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(batch_data):,} records found.")
            
            # Show preview
            with st.expander("📄 View Data Preview (First 10 rows)"):
                st.dataframe(batch_data.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Process data
                    batch_processed = batch_data.copy()
                    
                    # Drop ID if exists
                    if 'ID' in batch_processed.columns:
                        ids = batch_processed['ID']
                        batch_input = batch_processed.drop('ID', axis=1)
                    else:
                        ids = range(1, len(batch_processed) + 1)
                        batch_input = batch_processed
                    
                    # Make predictions
                    predictions, probabilities = make_prediction(batch_input, model, scaler, feature_names, feature_config)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'ID': ids,
                        'Default_Prediction': predictions,
                        'Default_Probability': probabilities[:, 1],
                        'Non_Default_Probability': probabilities[:, 0],
                        'Risk_Category': pd.cut(probabilities[:, 1], 
                                                bins=[0, 0.3, 0.7, 1.0], 
                                                labels=['Low', 'Medium', 'High'])
                    })
                    
                    # Display summary
                    st.subheader("📊 Prediction Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Customers", f"{len(results_df):,}")
                    with col2:
                        defaults = (predictions == 1).sum()
                        st.metric("Predicted Defaults", f"{defaults:,}")
                    with col3:
                        st.metric("Default Rate", f"{(predictions == 1).mean():.2%}")
                    with col4:
                        st.metric("Avg Risk Score", f"{probabilities[:, 1].mean():.2%}")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Risk distribution pie chart
                        risk_counts = results_df['Risk_Category'].value_counts()
                        fig = px.pie(values=risk_counts.values, 
                                    names=risk_counts.index,
                                    title="Risk Distribution",
                                    color=risk_counts.index,
                                    color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Probability distribution histogram
                        fig = px.histogram(results_df, x='Default_Probability', 
                                         nbins=50,
                                         title="Default Probability Distribution",
                                         labels={'Default_Probability': 'Default Probability'})
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True, height=400)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"credit_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV has all required columns and correct data types.")

# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================

with tab3:
    st.header("Model Performance & Insights")
    
    # Performance Metrics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Performance Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
            'Value': [
                model_info['test_accuracy'],
                model_info['f1_score'],
                model_info['precision'],
                model_info['recall']
            ]
        })
        
        fig = px.bar(metrics_df, x='Metric', y='Value', 
                    title=f"{model_info['model_name']} - Test Performance",
                    color='Value',
                    color_continuous_scale='Blues',
                    text='Value')
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(showlegend=False)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        
        # Train vs Test Comparison
        if 'train_accuracy' in model_info:
            st.subheader("🔄 Train vs Test Performance")
            comparison_df = pd.DataFrame({
                'Dataset': ['Train', 'Test', 'Train', 'Test'],
                'Metric': ['Accuracy', 'Accuracy', 'F1 Score', 'F1 Score'],
                'Value': [
                    model_info.get('train_accuracy', 0),
                    model_info['test_accuracy'],
                    model_info.get('train_accuracy', 0),  # Placeholder
                    model_info['f1_score']
                ]
            })
            
            fig = px.bar(comparison_df, x='Metric', y='Value', color='Dataset',
                        barmode='group', title='Train vs Test Performance')
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ Model Details")
        st.info(f"""
        **Model Type:** {model_info['model_name']}
        
        **Training Date:** {model_info['train_date']}
        
        **Model Architecture:** {model_info.get('model_type', 'N/A')}
        
        **Number of Features:** {model_info['n_features']}
        
        **Target Variable:** {model_info.get('target_variable', 'default.payment.next.month')}
        
        **Class Labels:** 
        - 0: No Default (Good Customer)
        - 1: Default (High Risk)
        """)
        
        # Overfitting check
        if model_info.get('overfitting') == 'No':
            st.success("✅ **No Overfitting Detected**\n\nThe model generalizes well to unseen data.")
        else:
            st.warning("⚠️ **Overfitting Detected**\n\nModel may perform worse on new data.")
    
    st.markdown("---")
    
    # Confusion Matrix Section
    if confusion_matrix_data:
        st.subheader("📉 Confusion Matrix Analysis")
        
        cm = confusion_matrix_data['confusion_matrix']
        class_labels = confusion_matrix_data['class_labels']
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create confusion matrix heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_labels,
                y=class_labels,
                colorscale='Blues',
                text=cm,
                texttemplate='<b>%{text:,}</b>',
                textfont={"size": 20},
                hovertemplate='<b>Predicted:</b> %{x}<br><b>Actual:</b> %{y}<br><b>Count:</b> %{z:,}<extra></extra>',
                showscale=True
            ))
            
            fig.update_layout(
                title='Confusion Matrix - Model Performance on Test Set',
                xaxis_title='<b>Predicted Label</b>',
                yaxis_title='<b>True Label</b>',
                width=600,
                height=500,
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Confusion Matrix Breakdown")
            
            tn = confusion_matrix_data['true_negatives']
            fp = confusion_matrix_data['false_positives']
            fn = confusion_matrix_data['false_negatives']
            tp = confusion_matrix_data['true_positives']
            
            st.metric("✅ True Negatives (TN)", f"{tn:,}", 
                     help="Correctly predicted non-defaults")
            st.metric("❌ False Positives (FP)", f"{fp:,}", 
                     help="Non-defaulters incorrectly flagged as high-risk")
            st.metric("⚠️ False Negatives (FN)", f"{fn:,}", 
                     help="Defaulters missed by the model - CRITICAL!")
            st.metric("✅ True Positives (TP)", f"{tp:,}", 
                     help="Correctly predicted defaults")
        
        # Calculate metrics from confusion matrix
        st.markdown("---")
        st.subheader("📊 Derived Metrics from Confusion Matrix")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}", 
                     help="Overall correct predictions")
        with col2:
            st.metric("Precision", f"{precision:.2%}", 
                     help="Accuracy of positive predictions")
        with col3:
            st.metric("Recall (Sensitivity)", f"{recall:.2%}", 
                     help="% of actual defaults caught")
        with col4:
            st.metric("Specificity", f"{specificity:.2%}", 
                     help="% of non-defaults correctly identified")
        
        # Business interpretation
        st.markdown("---")
        st.subheader("💼 Business Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **📋 Understanding the Matrix:**
            
            - **True Negatives (TN)**: Customers correctly identified as low-risk
              → Good customers approved ✅
            
            - **False Positives (FP)**: Low-risk customers flagged as high-risk
              → Lost business opportunity 📉
            
            - **False Negatives (FN)**: High-risk customers approved
              → Financial loss through defaults 💸⚠️
            
            - **True Positives (TP)**: High-risk customers correctly identified
              → Prevented losses ✅
            """)
        
        with col2:
            st.warning("""
            **💰 Cost-Benefit Analysis:**
            
            **Cost of False Positives (FP):**
            - Lost revenue from rejected good customers
            - Damaged customer relationships
            - Competitive disadvantage
            
            **Cost of False Negatives (FN):**
            - Direct financial losses from defaults
            - Collection costs
            - Legal expenses
            
            ⚖️ **Balancing Act:** In credit risk, FN typically costs more than FP, 
            so minimizing false negatives is critical!
            """)
    
    else:
        st.warning("⚠️ Confusion matrix data not available. Please ensure confusion_matrix.pkl is in the app directory.")
    
    st.markdown("---")
    
    # Feature Information
    st.subheader("🔧 Feature Engineering Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📝 Original Features (23):**")
        original_features = [f for f in feature_names if f not in feature_config.get('engineered_features', [])]
        for i, feat in enumerate(original_features[:12], 1):
            st.text(f"{i}. {feat}")
    
    with col2:
        st.markdown("**✨ Engineered Features (5):**")
        engineered = feature_config.get('engineered_features', [])
        for i, feat in enumerate(engineered, 1):
            st.text(f"{i}. {feat}")
        
        st.markdown("\n**🔄 Preprocessing:**")
        st.text(f"- Scaler: {feature_config.get('scaler_type', 'StandardScaler')}")
        st.text(f"- Total Features: {len(feature_names)}")
    
    # Dataset split information
    if split_info:
        st.markdown("---")
        st.subheader("📊 Dataset Split Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Samples", f"{split_info['train_size']:,}")
        with col2:
            st.metric("Test Samples", f"{split_info['test_size']:,}")
        with col3:
            st.metric("Train Default Rate", f"{split_info['train_default_rate']:.2%}")
        with col4:
            st.metric("Test Default Rate", f"{split_info['test_default_rate']:.2%}")
        
        # Visualization of split
        split_viz_df = pd.DataFrame({
            'Dataset': ['Training', 'Test'],
            'Samples': [split_info['train_size'], split_info['test_size']]
        })
        
        fig = px.pie(split_viz_df, values='Samples', names='Dataset',
                    title='Train-Test Split Distribution',
                    color_discrete_sequence=['#636EFA', '#EF553B'])
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: MODEL COMPARISON
# ============================================================================

with tab4:
    st.header("Model Comparison & Selection")
    
    if model_comparison is not None:
        st.markdown("### 🏆 All Models Performance Comparison")
        
        # Display comparison table
        st.dataframe(model_comparison, use_container_width=True, height=400)
        
        # Highlight best model
        best_idx = model_comparison['F1 Score'].idxmax()
        best_model_name = model_comparison.loc[best_idx, 'Model']
        
        st.success(f"🥇 **Best Model:** {best_model_name} (F1 Score: {model_comparison.loc[best_idx, 'F1 Score']:.4f})")
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Visual Comparison")
        
        # F1 Score comparison
        fig = px.bar(model_comparison.sort_values('F1 Score', ascending=True), 
                    x='F1 Score', y='Model', 
                    orientation='h',
                    title='Model Comparison: F1 Score',
                    color='F1 Score',
                    color_continuous_scale='Viridis',
                    text='F1 Score')
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Train vs Test Accuracy
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Train Accuracy',
            x=model_comparison['Model'],
            y=model_comparison['Train Accuracy'],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Test Accuracy',
            x=model_comparison['Model'],
            y=model_comparison['Test Accuracy'],
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title='Train vs Test Accuracy - Overfitting Check',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            barmode='group',
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Precision-Recall comparison
        fig = px.scatter(model_comparison, x='Precision', y='Recall',
                        text='Model', size='F1 Score',
                        title='Precision vs Recall Trade-off',
                        color='F1 Score',
                        color_continuous_scale='RdYlGn')
        fig.update_traces(textposition='top center')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Overfitting analysis
        st.markdown("---")
        st.subheader("🔍 Overfitting Analysis")
        
        overfitting_counts = model_comparison['Overfitting'].value_counts()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = px.pie(values=overfitting_counts.values, 
                        names=overfitting_counts.index,
                        title="Overfitting Distribution",
                        color_discrete_map={'Yes': 'red', 'No': 'green'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Models with Overfitting:")
            overfit_models = model_comparison[model_comparison['Overfitting'] == 'Yes']['Model'].tolist()
            if overfit_models:
                for model_name in overfit_models:
                    st.warning(f"⚠️ {model_name}")
            else:
                st.success("✅ No models showing overfitting!")
        
        # Download comparison
        st.markdown("---")
        csv = model_comparison.to_csv(index=False)
        st.download_button(
            label="📥 Download Model Comparison (CSV)",
            data=csv,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("⚠️ Model comparison data not available. Please ensure all_models_comparison.csv is in the app directory.")
        
        # Show current model info instead
        st.info(f"""
        **Current Model Information:**
        
        - **Model:** {model_info['model_name']}
        - **Test Accuracy:** {model_info['test_accuracy']:.2%}
        - **F1 Score:** {model_info['f1_score']:.4f}
        - **Precision:** {model_info['precision']:.2%}
        - **Recall:** {model_info['recall']:.2%}
        """)

# ============================================================================
# TAB 5: DATA MANAGEMENT
# ============================================================================

with tab5:
    st.header("💾 Data Management & Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Generate Custom Template")
        st.markdown("Create a custom CSV template for batch predictions")
        
        # Template customization
        include_id = st.checkbox("Include ID column", value=True)
        include_timestamp = st.checkbox("Include Timestamp column", value=True)
        num_rows = st.number_input("Number of sample rows", min_value=1, max_value=100, value=5)
        
        if st.button("📥 Generate Custom Template", type="primary"):
            # Create custom template
            template_data = {}
            
            if include_id:
                template_data['ID'] = range(1, num_rows + 1)
            if include_timestamp:
                template_data['Timestamp'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * num_rows
            
            # Add all required columns with sample data
            template_data.update({
                'LIMIT_BAL': [50000] * num_rows,
                'SEX': [1] * num_rows,
                'EDUCATION': [2] * num_rows,
                'MARRIAGE': [1] * num_rows,
                'AGE': [35] * num_rows,
                'PAY_0': [0] * num_rows,
                'PAY_2': [0] * num_rows,
                'PAY_3': [0] * num_rows,
                'PAY_4': [0] * num_rows,
                'PAY_5': [0] * num_rows,
                'PAY_6': [0] * num_rows,
                'BILL_AMT1': [20000] * num_rows,
                'BILL_AMT2': [20000] * num_rows,
                'BILL_AMT3': [20000] * num_rows,
                'BILL_AMT4': [20000] * num_rows,
                'BILL_AMT5': [20000] * num_rows,
                'BILL_AMT6': [20000] * num_rows,
                'PAY_AMT1': [2000] * num_rows,
                'PAY_AMT2': [2000] * num_rows,
                'PAY_AMT3': [2000] * num_rows,
                'PAY_AMT4': [2000] * num_rows,
                'PAY_AMT5': [2000] * num_rows,
                'PAY_AMT6': [2000] * num_rows
            })
            
            custom_template = pd.DataFrame(template_data)
            
            # Show preview
            st.dataframe(custom_template, use_container_width=True)
            
            # Download
            csv = custom_template.to_csv(index=False)
            st.download_button(
                label="📥 Download Custom Template",
                data=csv,
                file_name=f"custom_template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        st.subheader("📊 Export Model Configuration")
        st.markdown("Download complete model configuration and metadata")
        
        if st.button("📥 Generate Configuration File", type="primary"):
            # Create comprehensive config
            config_export = {
                'model_info': model_info,
                'feature_config': feature_config,
                'feature_names': feature_names,
                'split_info': split_info if split_info else {},
                'confusion_matrix': {
                    'true_negatives': confusion_matrix_data['true_negatives'],
                    'false_positives': confusion_matrix_data['false_positives'],
                    'false_negatives': confusion_matrix_data['false_negatives'],
                    'true_positives': confusion_matrix_data['true_positives']
                } if confusion_matrix_data else {},
                'export_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'app_version': '1.0'
            }
            
            # Convert to JSON
            config_json = json.dumps(config_export, indent=2, default=str)
            
            st.success("✅ Configuration ready for download!")
            st.download_button(
                label="📥 Download Configuration (JSON)",
                data=config_json,
                file_name=f"model_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    st.markdown("---")
    
    # Feature engineering guide
    st.subheader("📚 Feature Engineering Guide")
    
    with st.expander("🔍 View Detailed Feature Engineering Steps"):
        st.markdown("""
        ### Feature Engineering Process
        
        The model uses the following engineered features to improve prediction accuracy:
        
        #### 1. **AVG_PAYMENT_STATUS**
        - **Formula:** Average of (PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
        - **Purpose:** Captures overall payment behavior trend
        - **Interpretation:** Higher values indicate more payment delays
        
        #### 2. **MAX_PAYMENT_DELAY**
        - **Formula:** Maximum of (PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6)
        - **Purpose:** Identifies worst payment behavior
        - **Interpretation:** Indicates maximum delay period
        
        #### 3. **TOTAL_BILL_AMT**
        - **Formula:** Sum of (BILL_AMT1 through BILL_AMT6)
        - **Purpose:** Total outstanding balance
        - **Interpretation:** Higher values indicate more debt burden
        
        #### 4. **TOTAL_PAY_AMT**
        - **Formula:** Sum of (PAY_AMT1 through PAY_AMT6)
        - **Purpose:** Total payments made
        - **Interpretation:** Higher values indicate better payment capacity
        
        #### 5. **UTILIZATION_RATIO**
        - **Formula:** TOTAL_BILL_AMT / (LIMIT_BAL + 1)
        - **Purpose:** Credit utilization rate
        - **Interpretation:** Values close to 1 indicate maxed-out credit
        
        ---
        
        ### Preprocessing Steps
        
        1. **Feature Engineering:** Create 5 new features as described above
        2. **Feature Selection:** Ensure all 28 features are present
        3. **Scaling:** Apply StandardScaler to normalize all features
        4. **Prediction:** Pass scaled features to the model
        
        ---
        
        ### Important Notes
        
        ⚠️ **Critical:** All features must be engineered in the exact same order and manner as during training
        
        ✅ **Best Practice:** Use the provided functions in this app for consistency
        """)
    
    # Data dictionary
    st.markdown("---")
    st.subheader("📖 Data Dictionary")
    
    with st.expander("📋 View Complete Data Dictionary"):
        data_dict = pd.DataFrame({
            'Feature': [
                'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                'BILL_AMT1-6', 'PAY_AMT1-6',
                'AVG_PAYMENT_STATUS', 'MAX_PAYMENT_DELAY', 
                'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT', 'UTILIZATION_RATIO'
            ],
            'Description': [
                'Credit limit in NT dollars',
                'Gender (1=Male, 2=Female)',
                'Education level (1=Grad, 2=Univ, 3=High, 4=Others)',
                'Marital status (1=Married, 2=Single, 3=Others)',
                'Age in years',
                'Payment status September 2005',
                'Payment status August 2005',
                'Payment status July 2005',
                'Payment status June 2005',
                'Payment status May 2005',
                'Payment status April 2005',
                'Bill statement amounts (6 months)',
                'Payment amounts (6 months)',
                'Average payment delay over 6 months',
                'Maximum payment delay observed',
                'Sum of all bill amounts',
                'Sum of all payment amounts',
                'Total bills / Credit limit'
            ],
            'Type': [
                'Numerical', 'Categorical', 'Categorical', 'Categorical', 'Numerical',
                'Categorical', 'Categorical', 'Categorical', 'Categorical', 'Categorical', 'Categorical',
                'Numerical', 'Numerical',
                'Numerical (Engineered)', 'Numerical (Engineered)',
                'Numerical (Engineered)', 'Numerical (Engineered)', 'Numerical (Engineered)'
            ]
        })
        
        st.dataframe(data_dict, use_container_width=True, height=600)
    
    # Model files information
    st.markdown("---")
    st.subheader("📦 Model Files Information")
    
    files_info = {
        'File Name': [
            'best_credit_model.pkl',
            'scaler.pkl',
            'feature_names.pkl',
            'feature_config.pkl',
            'model_info.pkl',
            'confusion_matrix.pkl',
            'split_info.pkl',
            'all_models_comparison.csv'
        ],
        'Description': [
            'Trained machine learning model',
            'StandardScaler for feature preprocessing',
            'List of feature names in correct order',
            'Feature engineering configuration',
            'Model metadata and performance metrics',
            'Confusion matrix and derived metrics',
            'Train-test split information',
            'Performance comparison of all models'
        ],
        'Required': [
            '✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes',
            '⚠️ Optional', '⚠️ Optional', '⚠️ Optional'
        ]
    }
    
    files_df = pd.DataFrame(files_info)
    st.dataframe(files_df, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p style='font-size: 18px;'><b>💳 Credit Card Default Prediction System</b></p>
        <p>Version 1.0 | Built with Streamlit & Machine Learning</p>
        <p style='font-size: 12px;'>
            Model: {model_name} | Accuracy: {accuracy:.2%} | F1: {f1:.4f}<br>
            Last Updated: {date}
        </p>
    </div>
    """.format(
        model_name=model_info['model_name'],
        accuracy=model_info['test_accuracy'],
        f1=model_info['f1_score'],
        date=model_info['train_date']
    ), unsafe_allow_html=True)