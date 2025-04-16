# AI-Powered Diabetes Diagnosis System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning system for early diabetes prediction using clinical parameters, designed for healthcare professionals.

## Key Features

- **89.5% Accuracy**: Random Forest classifier optimized for medical diagnostics
- **Explainable AI**: SHAP-based feature importance visualization
- **Clinician-Friendly Interface**: Streamlit web app with input validation
- **Deployment Ready**: Docker container for easy clinical integration
- **Multilingual Support**: English/Telugu interface options

## Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup
```bash
git clone https://github.com/yourusername/diabetes-ai.git
cd diabetes-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
