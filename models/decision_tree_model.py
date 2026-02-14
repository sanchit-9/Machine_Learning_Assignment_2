import joblib
import os
from sklearn.tree import DecisionTreeClassifier

def train_model(X_train, y_train, output_dir_path, random_state=42):
    """Trains a Decision Tree model and saves it to the specified path."""
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Create directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)

    model_path = os.path.join(output_dir_path, 'decision_tree_model.joblib')
    joblib.dump(model, model_path)
    print(f"Decision Tree model trained and saved to {model_path}")
    return model

def load_model(model_path):
    """Loads a trained Decision Tree model from the specified path."""
    return joblib.load(model_path)

def predict(model, X_data):
    """Makes predictions (labels and probabilities) using the trained Decision Tree model."""
    y_pred = model.predict(X_data)
    y_pred_proba = model.predict_proba(X_data)[:, 1]
    return y_pred, y_pred_proba
