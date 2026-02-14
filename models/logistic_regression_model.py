import joblib
import os
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, output_dir_path, random_state=42):
    """Trains a Logistic Regression model and saves it to the specified path."""
    model = LogisticRegression(random_state=random_state, solver='liblinear')
    model.fit(X_train, y_train)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)
    
    model_path = os.path.join(output_dir_path, 'logistic_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"Logistic Regression model trained and saved to {model_path}")
    return model

def load_model(model_path):
    """Loads a trained Logistic Regression model from the specified path."""
    return joblib.load(model_path)

def predict(model, X_data):
    """Makes predictions (labels and probabilities) using the trained Logistic Regression model."""
    y_pred = model.predict(X_data)
    y_pred_proba = model.predict_proba(X_data)[:, 1]
    return y_pred, y_pred_proba
