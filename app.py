import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model
model = joblib.load("my_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .stTextInput>div>div>input {
            border-radius: 8px;
            padding: 10px;
        }
        .stSelectbox>div>div>div {
            border-radius: 8px;
            padding: 8px;
        }
        .stButton>button {
            border-radius: 8px;
            padding: 10px 24px;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .css-1aumxhk {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
        }
        .stAlert {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("💰 Employee Salary Classifier")
st.markdown("""
    Predict whether an employee earns **>50K** or **≤50K** based on their demographic and employment characteristics.
""")
st.markdown("---")

# Sidebar for categorical inputs
with st.sidebar:
    st.header("Employee Details")
    st.markdown("Please provide the employee's information:")
    
    # Categorical inputs with better organization
    with st.expander("Personal Information", expanded=True):
        age = st.text_input("Age", placeholder="e.g. 35", help="Employee's age in years")
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", [
            "Married-civ-spouse", "Never-married", "Divorced", 
            "Separated", "Widowed", "Married-spouse-absent"
        ])
        relationship = st.selectbox("Relationship", [
            "Husband", "Not-in-family", "Own-child", 
            "Unmarried", "Wife", "Other-relative"
        ])
    
    with st.expander("Employment Information", expanded=True):
        workclass = st.selectbox("Work Class", [
            "Private", "Self-emp-not-inc", "Local-gov", 
            "Others", "State-gov", "Self-emp-inc", "Federal-gov"
        ])
        education = st.selectbox("Education Level", [
            "Bachelors", "Masters", "Prof-school", "HS-grad", 
            "Assoc-voc", "Some-college", "10th", "11th", 
            "12th", "Doctorate", "Assoc-acdm"
        ])
        occupation = st.selectbox("Occupation", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
            "Transport-moving", "Priv-house-serv", "Protective-serv"
        ])
        hours_per_week = st.text_input("Hours per Week", placeholder="e.g. 40", help="Typical weekly working hours")

# Main content area for numerical inputs
with st.container():
    st.subheader("Additional Financial Information")
    col1, col2 = st.columns(2)
    
    with col1:
        fnlwgt = st.text_input("Final Weight (fnlwgt)", placeholder="e.g. 250000", 
                             help="Statistical weight assigned by the Census Bureau")
        gain = st.text_input("Capital Gain", placeholder="e.g. 5000", 
                           help="Income from capital investments")
    
    with col2:
        loss = st.text_input("Capital Loss", placeholder="e.g. 2000", 
                           help="Loss from capital investments")
    
    st.markdown("---")

# Prediction button and results
if st.button("Predict Salary Range", type="primary"):
    # Input validation
    required_fields = {
        "Age": age,
        "Hours per Week": hours_per_week,
        "Final Weight": fnlwgt,
        "Capital Gain": gain,
        "Capital Loss": loss
    }
    
    missing_fields = [field for field, value in required_fields.items() if not value.strip()]
    
    if missing_fields:
        st.error(f"Please fill in all required fields: {', '.join(missing_fields)}")
    else:
        try:
            # Create input DataFrame
            input_df = pd.DataFrame({
                'age': [int(age)],
                'workclass': [workclass],
                'fnlwgt': [int(fnlwgt)],
                'education': [education],
                'marital-status': [marital],
                'occupation': [occupation],
                'relationship': [relationship],
                'gender': [gender],
                'capital-gain': [int(gain)],
                'capital-loss': [int(loss)],
                'hours-per-week': [int(hours_per_week)],
            })

            # Data processing
            input_df['net-capital'] = input_df['capital-gain'] - input_df['capital-loss']
            input_df.drop(columns=['capital-gain', 'capital-loss', 'gender'], inplace=True)

            # Label encoding for categorical features
            cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship']
            le = LabelEncoder()
            for col in cat_cols:
                input_df[col] = le.fit_transform(input_df[col])

            # Standard scaling for numerical features
            num_cols = ['age', 'fnlwgt', 'hours-per-week', 'net-capital']
            ss = StandardScaler()
            input_df[num_cols] = ss.fit_transform(input_df[num_cols])

            # Make prediction
            prediction = model.predict(input_df)
            
            # Display result with better formatting
            st.markdown("---")
            if prediction == 1:
                st.success("## Prediction: Salary > $50K")
                st.markdown("This employee is predicted to earn **more than $50,000** annually.")
            else:
                st.info("## Prediction: Salary ≤ $50K")
                st.markdown("This employee is predicted to earn **$50,000 or less** annually.")
            
            # Show input summary
            with st.expander("See input summary"):
                st.write(input_df)
                
        except ValueError as e:
            st.error(f"Invalid input format: {str(e)}. Please enter valid numbers in all fields.")

# Add some footer information
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: small;">
        <p>Employee Salary Classification Model</p>
        <p>Note: Predictions are based on statistical modeling and may not reflect actual salaries.</p>
    </div>
""", unsafe_allow_html=True)