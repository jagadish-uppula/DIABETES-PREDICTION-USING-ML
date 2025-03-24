import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """
    Robust diabetes data loader that handles:
    - Duplicate/multiple header rows
    - Index columns
    - Mixed numeric/string data
    - Missing values
    
    Returns clean pandas DataFrame ready for ML
    """
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    try:
        # First attempt: Try reading as clean CSV
        try:
            data = pd.read_csv(
                data_path,
                header=None,
                names=column_names,
                index_col=False
            )
        except:
            # Fallback: Handle messy files with text headers
            raw = pd.read_csv(data_path, header=None)
            
            # Find first numeric row (skip all header rows)
            for i, row in raw.iterrows():
                if pd.to_numeric(row.iloc[0], errors='coerce') is not np.nan:
                    break
            
            data = raw.iloc[i:].copy()
            data.columns = column_names[:len(data.columns)] # Handle column mismatch
        
        # Convert all columns to numeric
        data = data.apply(pd.to_numeric, errors='coerce')
        
        # Validate data shape
        if len(data.columns) != 9:
            raise ValueError(f"Expected 9 columns, got {len(data.columns)}")
        
        # Check for missing values
        if data.isnull().values.any():
            print(f"Warning: {data.isnull().sum().sum()} missing values detected - dropping them")
            data = data.dropna()
        
        return data
    
    except Exception as e:
        print(f"Fatal error loading data: {str(e)}")
        print("Please verify your CSV file contains:")
        print("- Exactly 9 columns per row")
        print("- Numeric values only")
        print("- No duplicate headers")
        raise

def split_data(data, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    Args:
        data: Clean DataFrame from load_data()
        test_size: Fraction for test set (default 0.2)
        random_state: Random seed (default 42)
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Verify input
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be pandas DataFrame")
    
    if 'Outcome' not in data.columns:
        raise ValueError("DataFrame must contain 'Outcome' column")
    
    # Split data
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y # Preserve class balance
    )

# Example usage:
if __name__ == "__main__":
    try:
        print("Testing data loader...")
        data = load_data("../data/diabetes.csv")
        print("\nFirst 5 rows:")
        print(data.head())
        
        print("\nData types:")
        print(data.dtypes)
        
        print("\nNull values:")
        print(data.isnull().sum())
        
        X_train, X_test, y_train, y_test = split_data(data)
        print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")