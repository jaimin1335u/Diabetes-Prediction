import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib  # For saving/loading the model
import os      # For checking if the model file exists

# --- Configuration ---
DATA_FILE = "diabetes.csv"
MODEL_FILE = "diabetes_model.joblib"


# Load the dataset
try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found. Please run 'generate_synthetic_data.py' first.")
    data = pd.DataFrame(columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ])

# Handle problematic zero values by replacing them with NaN
if 'Outcome' in data.columns:
    print("Original data shape:", data.shape)
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
    print("Data shape after replacing 0s with NaN:", data.shape)

    # Split into features (X) and target (y)
if 'Outcome' in data.columns and not data.empty:
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    feature_names = X.columns.tolist()

    # Split into training/testing, 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # --- Check for Saved Model ---
    if os.path.exists(MODEL_FILE):
        print(f"Loading saved model from '{MODEL_FILE}'...")
        model_pipeline = joblib.load(MODEL_FILE)
        print("Model loaded successfully.")
    
    else:
        print(f"No saved model found. Training a new model (this may take a few minutes)...")
        # --- Create a preprocessing and modeling pipeline ---
        
        # We use 'model' as the name for the classifier step
        model_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42)) # Base model
        ])
        
        # --- Hyperparameter Tuning with GridSearchCV ---
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }
        
        # Set up the Grid Search
        grid_search = GridSearchCV(
            estimator=model_pipeline, 
            param_grid=param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1
        )
        
        # Train the grid search
        print("Starting hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        
        # Get the best model from the search
        model_pipeline = grid_search.best_estimator_
        
        print("Tuning finished.")
        print(f"Best parameters found: {grid_search.best_params_}")
        
        # --- Save the Model ---
        joblib.dump(model_pipeline, MODEL_FILE)
        print(f"Model saved to '{MODEL_FILE}'")


    # --- Model Performance (runs every time with the loaded or trained model) ---
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probabilities for the positive class
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"New Model Accuracy: {accuracy:.4f}")

else:
    # Handle case where dummy data was created
    model_pipeline = None
    accuracy = 0
    X_test, y_test, y_pred, y_pred_proba = (None, None, None, None)
    feature_names = []


def getModel():
    """Returns the trained model pipeline."""
    return model_pipeline

def getAccuracy():
    """Returns the accuracy score on the test set."""
    return accuracy

def get_test_data():
    """Returns the test features (X) and test target (y)."""
    return X_test, y_test

def get_predictions():
    """Returns the model's predictions (classes) and probabilities on the test set."""
    return y_pred, y_pred_proba

def get_feature_names():
    """Returns the list of feature names."""
    return feature_names

def get_full_data():
    """Returns the complete original dataset."""
    return data