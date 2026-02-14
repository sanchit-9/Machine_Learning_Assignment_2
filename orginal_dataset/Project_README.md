# Machine Learning - Assignment 2 
### Submitted By: Sanchit Bathla (2025AA05922)
### (2025aa05922@wilp.bits-pilani.ac.in)
# Content Virality Prediction Streamlit App

## Overview
This Streamlit application allows users to predict the virality of social media content using various machine learning classification models and compare performance metrics between the models.

## Features
*   **CSV Upload**: Easily upload your social media content data in CSV format for inference.
*   **Training Mode**: Download the dataset, preprocess it, train all models, and then save newly trained models as joblib files.
*   **Data Preprocessing**: Automatic preprocessing of uploaded data, including one-hot encoding for categorical features and standard scaling for numerical features, using pre-trained `OneHotEncoder` and `StandardScaler` objects.
*   **Model Selection**: Choose from a variety of pre-trained classification models, including:
    *   Logistic Regression
    *   Decision Tree
    *   K-Nearest Neighbor
    *   Naive Bayes
    *   Random Forest
    *   XGBoost
*   **Virality Prediction**: Obtain predictions on whether content is likely to go viral.
*   **Performance Metrics**: If the uploaded data includes the target variable (`is_viral`), the application will display comprehensive evaluation metrics for the selected model:
    *   Accuracy
    *   AUC Score
    *   Precision
    *   Recall
    *   F1 Score
    *   Matthews Correlation Coefficient (MCC Score)
*   **Classification Report & Confusion Matrix**: Detailed classification report and confusion matrix to further assess model performance.


## Training Mode
For whenever we need to retrain models:
1. Select 'Training' mode from the sidebar.
2. Enter the password for training, default password is 'admin'.
3. Click the 'Start Training' button. This will download the dataset, preprocess it, train all six models, and save them (along with the preprocessors) into the `models/output_models/` directory.

## Inference Mode
1. Select 'Inference' mode from the sidebar.
2. Upload your CSV file with new content data.
3. Select a model from the dropdown.
4. Click 'Predict and Evaluate' to see predictions and performance metrics.

## Project Structure
*   `streamlit_app.py`: The main Streamlit application script, now supporting both training and inference modes.
*   `requirements.txt`: Lists all Python libraries required to run the application.
*   `original_social_media_viral_content_dataset.csv`: Original Dataset used for the application, Dataset Size is 2000 rows.
*   `test_data_viral.csv`: Test Data of 400 rows obtained after a 80/20 train to test split.
*   `models/`:
    *   `output_models/`: Directory containing all saved preprocessing objects (`OneHotEncoder`, `StandardScaler`) and trained machine learning models in `joblib` format.
    *   `logistic_regression_model.py`: Module for Logistic Regression model training, loading, and prediction.
    *   `decision_tree_model.py`: Module for Decision Tree model training, loading, and prediction.
    *   `knn_model.py`: Module for K-Nearest Neighbor model training, loading, and prediction.
    *   `naive_bayes_model.py`: Module for Naive Bayes model training, loading, and prediction.
    *   `random_forest_model.py`: Module for Random Forest model training, loading, and prediction.
    *   `xgboost_model.py`: Module for XGBoost model training, loading, and prediction.
