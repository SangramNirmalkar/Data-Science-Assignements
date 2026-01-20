import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set Page Config
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# Function to load data for the explorer
@st.cache_data
def load_test_data():
    try:
        return pd.read_csv('Titanic_test.csv')
    except FileNotFoundError:
        return None

# Load Trained Model and Scaler
@st.cache_resource
def load_model_objects():
    with open('logistic_regression.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

# Sidebar Navigation
st.sidebar.title("App Navigation")
page = st.sidebar.radio("Go to", ["Single Passenger Prediction", "Test Dataset Explorer"])

# Load objects
model, scaler, feature_cols = load_model_objects()

if page == "Single Passenger Prediction":
    st.title("🚢 Titanic Survival Predictor")
    st.markdown("Use this tool to input passenger details and predict their survival probability.")
    
    # Create Two Columns for Input
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox('Sex', ['female', 'male'])
        pclass = st.selectbox('Passenger Class', [1, 2, 3])
        age = st.number_input('Age', min_value=0, max_value=100, value=30)

    with col2:
        fare = st.number_input('Fare', min_value=0.0, value=32.2)
        sibsp = st.number_input('Siblings/Spouses (SibSp)', 0, 10, 0)
        parch = st.number_input('Parents/Children (Parch)', 0, 10, 0)

    # Process Input
    if st.button('Predict Survival'):
        # 1. Create Input Dictionary
        input_dict = {
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Age_missing': 0,
            'Sex_male': 1 if sex == 'male' else 0,
            'Pclass_2': 1 if pclass == 2 else 0,
            'Pclass_3': 1 if pclass == 3 else 0
        }
        
        # 2. Convert to DataFrame and align columns
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_cols]
        
        # 3. Scale numerical values (Age & Fare) exactly as done in training
        input_df[['Age', 'Fare']] = scaler.transform(input_df[['Age', 'Fare']])
        
        # 4. Predict using Probabilities
        prob = model.predict_proba(input_df)[0][1]
        
        # Use the 0.4 threshold from your notebook analysis
        threshold = 0.4
        result = "Survived" if prob >= threshold else "Did Not Survive"
        
        # Display Result
        st.markdown("---")
        if result == "Survived":
            st.success(f"### Prediction: {result}")
        else:
            st.error(f"### Prediction: {result}")
            
        st.write(f"**Survival Probability:** {prob:.2%}")
        st.caption(f"Note: Classification threshold set to {threshold} based on notebook evaluation.")

elif page == "Test Dataset Explorer":
    st.title("📊 Test Dataset Explorer")
    st.write("Below are the entries from the `Titanic_test.csv` file.")
    
    df_test = load_test_data()
    
    if df_test is not None:
        # Search functionality
        search = st.text_input("Filter by Passenger Name")
        if search:
            df_display = df_test[df_test['Name'].str.contains(search, case=False)]
        else:
            df_display = df_test
            
        st.dataframe(df_display, use_container_width=True)
        st.write(f"Showing {len(df_display)} rows.")
    else:
        st.error("Could not find `Titanic_test.csv`. Please ensure it is in the project folder.")