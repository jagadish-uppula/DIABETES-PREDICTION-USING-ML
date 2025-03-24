import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os
from utils.data_loader import load_data
from utils.visualization import (plot_distribution, 
                                plot_correlation_matrix, 
                                plot_feature_importance)

# Set page config
st.set_page_config(
    page_title="AI Diabetes Diagnosis",
    page_icon="🏥",
    layout="wide"
)

# Load custom CSS
def load_css():
    css_file = os.path.join("static", "styles.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model(model_path='models/model.pkl'):
    return joblib.load(model_path)

# Main app function
def main():
    load_css()
    model = load_model()
    data = load_data('data/diabetes.csv')
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose section", 
                               ["Home", "Data Exploration", "Diagnosis"])
    
    # Load images
    logo_img = Image.open(os.path.join("static", "images", "logo.png"))
    diabetes_img = Image.open(os.path.join("static", "images", "diabetes_image.jpg"))
    
    # Home page
    if app_mode == "Home":
        st.image(logo_img, width=200)
        st.title("AI-Powered Diabetes Diagnosis System")
        st.image(diabetes_img, use_column_width=True)
        st.markdown("""
        ## About This System
        This system helps healthcare professionals assess diabetes risk using machine learning.
        
        ### Features:
        - **Data Exploration**: Visualize diabetes dataset
        - **Patient Diagnosis**: Get instant risk assessment
        - **Model Insights**: Understand prediction factors
        """)
    
    # Data Exploration page
    elif app_mode == "Data Exploration":
        st.header("Data Exploration")
        
        if st.checkbox("Show Raw Data"):
            st.dataframe(data)
        
        st.subheader("Data Visualization")
        feature = st.selectbox("Select feature", data.columns[:-1])
        st.pyplot(plot_distribution(data, feature))
        
        st.subheader("Correlation Matrix")
        st.pyplot(plot_correlation_matrix(data))
    
    # Diagnosis page
    elif app_mode == "Diagnosis":
        st.header("Patient Diabetes Risk Assessment")
        
        with st.form("diagnosis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Pregnancies", 0, 20, 1)
                glucose = st.number_input("Glucose (mg/dL)", 0, 300, 100)
                bp = st.number_input("Blood Pressure (mm Hg)", 0, 150, 70)
                skin = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            
            with col2:
                insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
                bmi = st.number_input("BMI", 0.0, 70.0, 25.0, 0.1)
                dpf = st.number_input("Diabetes Pedigree", 0.0, 3.0, 0.5, 0.01)
                age = st.number_input("Age", 0, 120, 30)
            
            submitted = st.form_submit_button("Assess Risk")
            
            if submitted:
                input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                
                if prediction == 1:
                    st.error(f"High Risk ({proba[1]:.1%} probability)")
                    st.warning("Recommend consultation with a healthcare provider")
                else:
                    st.success(f"Low Risk ({proba[0]:.1%} probability)")
                    st.info("Maintain healthy lifestyle with regular checkups")
                
                # Show feature importance
                st.subheader("Key Contributing Factors")
                features = data.columns[:-1]
                importances = model.feature_importances_
                st.pyplot(plot_feature_importance(features, importances))

if __name__ == "__main__":
    main()
