import joblib
from sklearn.ensemble import RandomForestClassifier
from utils.data_loader import load_data, split_data

def train_and_save_model(data_path='data/diabetes.csv', model_path='models/model.pkl'):
    """Train and save the Random Forest model"""
    try:
        print("Loading data...")
        data = load_data(data_path)
        print("Data loaded successfully. Sample:")
        print(data.head())
        
        print("Splitting data...")
        X_train, X_test, y_train, y_test = split_data(data)
        
        print("Training model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("Saving model...")
        joblib.dump(model, model_path)
        print(f"Model successfully saved to {model_path}")
        
        return model, X_test, y_test
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    train_and_save_model()