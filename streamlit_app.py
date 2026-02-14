import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import kagglehub
import matplotlib.pyplot as plt # Added for visualization
import seaborn as sns # Added for visualization

# Import model modules
from models import logistic_regression_model
from models import decision_tree_model
from models import knn_model
from models import naive_bayes_model
from models import random_forest_model
from models import xgboost_model

# --- Configuration ---
MODEL_DIR = 'models/output_models'
categorical_features = ['platform', 'content_type', 'topic', 'language', 'region']
numerical_features = ['views', 'likes', 'comments', 'shares', 'engagement_rate', 'sentiment_score']

# Default password for training mode
PASSWORD = 'admin'

# --- Streamlit App ---
st.title('Content Virality Prediction App')

st.sidebar.header('Machine Learning - Assignment 2')
st.sidebar.subheader('Submitted By : Sanchit Bathla (2025AA05922)')

# Sidebar for mode selection
app_mode = st.sidebar.radio(
    "Choose an app mode",
    ("Inference", "Training")
)

# --- Training Mode ---
if app_mode == "Training":
    st.header("Training Mode")
    st.write("Enter the password to access training functionalities. (Default Password: admin)")

    entered_password = st.text_input('Enter password to enable training:', type='password')

    if entered_password == PASSWORD:
        st.success("Password correct. You can now train models.")
        st.write("This mode will download the dataset, preprocess it, train all models, and save them along with the preprocessors.")

        if st.button("Start Training"):
            st.write("Starting training process...")

            # 1. Download dataset
            st.info("Downloading dataset from KaggleHub...")
            try:
                path = kagglehub.dataset_download("aliiihussain/social-media-viral-content-and-engagement-metrics")
                file_list = os.listdir(path)
                file_name = file_list[0] # Assuming the first file is the main dataset
                df = pd.read_csv(os.path.join(path, file_name))
                st.success(f"Dataset downloaded and loaded successfully from {os.path.join(path, file_name)}")
            except Exception as e:
                st.error(f"Error downloading or loading dataset: {e}")
                st.stop()

            # 2. Prepare data for training
            st.info("Preprocessing data for training...")
            y = df['is_viral']
            X = df.drop('is_viral', axis=1)

            columns_to_drop = ['post_id', 'post_datetime', 'hashtags']
            X = X.drop(columns=[col for col in columns_to_drop if col in X.columns], axis=1)

            # Split into training and testing sets *before* fitting preprocessors
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 3. Fit and save preprocessors
            os.makedirs(MODEL_DIR, exist_ok=True)

            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            ohe.fit(X_train[categorical_features])
            joblib.dump(ohe, os.path.join(MODEL_DIR, 'onehot_encoder.joblib'))
            st.success("OneHotEncoder fitted and saved to `models/output_models/onehot_encoder.joblib`.")

            scaler = StandardScaler()
            scaler.fit(X_train[numerical_features])
            joblib.dump(scaler, os.path.join(MODEL_DIR, 'standard_scaler.joblib'))
            st.success("StandardScaler fitted and saved to `models/output_models/standard_scaler.joblib`.")

            # Apply preprocessing to training data for model training
            X_categorical_encoded_train = ohe.transform(X_train[categorical_features])
            X_categorical_encoded_df_train = pd.DataFrame(X_categorical_encoded_train, columns=ohe.get_feature_names_out(categorical_features), index=X_train.index)

            X_numerical_scaled_train = scaler.transform(X_train[numerical_features])
            X_numerical_scaled_df_train = pd.DataFrame(X_numerical_scaled_train, columns=numerical_features, index=X_train.index)

            X_processed_train = pd.concat([X_categorical_encoded_df_train, X_numerical_scaled_df_train], axis=1)

            # 4. Train and save models
            st.info("Training and saving machine learning models...")
            model_modules = {
                'Logistic Regression': logistic_regression_model,
                'Decision Tree': decision_tree_model,
                'K-Nearest Neighbor': knn_model,
                'Naive Bayes': naive_bayes_model,
                'Random Forest': random_forest_model,
                'XGBoost': xgboost_model
            }

            for name, module in model_modules.items():
                st.write(f"Training {name}...")
                # train_model function needs X_train, y_train, and output_dir_path
                # Note: X_train and y_train here are the preprocessed X_processed_train and y_train
                try:
                    module.train_model(X_processed_train, y_train, MODEL_DIR) # Pass preprocessed X_train
                    st.success(f"{name} trained and saved to `models/output_models/{module.__name__.split('.')[-1]}.joblib`.")
                except Exception as e:
                    st.error(f"Error training {name}: {e}")
            st.success("All models trained and saved successfully!")
    else:
        st.warning("Incorrect password.")

# --- Inference Mode (Existing functionality) ---
else: # app_mode == "Inference"
    st.header("Inference Mode")
    st.write(
        "Upload your new social media content data (CSV format) to predict virality."
        "The app will preprocess your data, make predictions using your chosen model,"
        "and display evaluation metrics."
    )

    # --- Load Preprocessing Objects ---
    try:
        ohe = joblib.load(os.path.join(MODEL_DIR, 'onehot_encoder.joblib'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'standard_scaler.joblib'))
        st.sidebar.success("Preprocessing objects loaded.")
    except FileNotFoundError:
        st.sidebar.error("Preprocessing objects not found. Please run the 'Training' mode first.")
        st.stop()

    # --- Load Models ---
    models = {}
    model_configs = {
        'Logistic Regression': {'module': logistic_regression_model, 'file': 'logistic_regression_model.joblib'},
        'Decision Tree': {'module': decision_tree_model, 'file': 'decision_tree_model.joblib'},
        'K-Nearest Neighbor': {'module': knn_model, 'file': 'knn_model.joblib'},
        'Naive Bayes': {'module': naive_bayes_model, 'file': 'naive_bayes_model.joblib'},
        'Random Forest': {'module': random_forest_model, 'file': 'random_forest_model.joblib'},
        'XGBoost': {'module': xgboost_model, 'file': 'xgboost_model.joblib'}
    }

    for name, config in model_configs.items():
        try:
            model_path = os.path.join(MODEL_DIR, config['file'])
            loaded_model = config['module'].load_model(model_path)
            models[name] = {'model': loaded_model, 'predict_func': config['module'].predict}
        except FileNotFoundError:
            st.sidebar.error(f"Model '{name}' not found. Please ensure '{config['file']}' is in the '{MODEL_DIR}/' directory or run the 'Training' mode.")
            st.stop()
    st.sidebar.success("Models loaded successfully.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview:")
        st.write(df_uploaded.head())

        # Check for target variable in uploaded data
        if 'is_viral' in df_uploaded.columns:
            y_uploaded = df_uploaded['is_viral']
            X_uploaded = df_uploaded.drop('is_viral', axis=1)
            st.success("Target variable 'is_viral' found in uploaded data. Performance metrics will be calculated.")
        else:
            y_uploaded = None
            X_uploaded = df_uploaded.copy()
            st.warning("Target variable 'is_viral' not found in uploaded data. Only predictions will be made.")

        # --- Preprocessing the uploaded data ---
        # Drop irrelevant columns
        columns_to_drop = ['post_id', 'post_datetime', 'hashtags']
        X_uploaded = X_uploaded.drop(columns=[col for col in columns_to_drop if col in X_uploaded.columns], axis=1)

        # Separate categorical and numerical features
        uploaded_categorical_data = X_uploaded[categorical_features]
        uploaded_numerical_data = X_uploaded[numerical_features]

        # Apply OneHotEncoder
        X_categorical_encoded = ohe.transform(uploaded_categorical_data)
        X_categorical_encoded_df = pd.DataFrame(X_categorical_encoded, columns=ohe.get_feature_names_out(categorical_features), index=X_uploaded.index)

        # Apply StandardScaler
        X_numerical_scaled = scaler.transform(uploaded_numerical_data)
        X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=X_uploaded.index)

        # Concatenate processed features
        X_uploaded_processed = pd.concat([X_categorical_encoded_df, X_numerical_scaled_df], axis=1)

        # Ensure column order matches training data
        # Construct the full list of expected columns from the training phase
        expected_columns = list(ohe.get_feature_names_out(categorical_features)) + numerical_features
        X_uploaded_processed = X_uploaded_processed.reindex(columns=expected_columns, fill_value=0)

        st.subheader("Processed Data Preview (features for prediction):")
        st.write(X_uploaded_processed.head())

        # --- Model Selection ---
        selected_model_name = st.selectbox('Select a Model for Prediction:', list(models.keys()))
        selected_model_info = models[selected_model_name]

        # --- Prediction and Evaluation for selected model ---
        if st.button('Predict and Evaluate Selected Model'): # Renamed button for clarity
            if y_uploaded is not None:
                st.subheader(f"Results for {selected_model_name}:")

                # Make predictions using the loaded model's predict function
                y_pred, y_pred_proba = selected_model_info['predict_func'](selected_model_info['model'], X_uploaded_processed)

                # Calculate and display evaluation metrics
                accuracy = accuracy_score(y_uploaded, y_pred)
                auc = roc_auc_score(y_uploaded, y_pred_proba)
                precision = precision_score(y_uploaded, y_pred)
                recall = recall_score(y_uploaded, y_pred)
                f1 = f1_score(y_uploaded, y_pred)
                mcc = matthews_corrcoef(y_uploaded, y_pred)

                st.write(f"**Accuracy:** {accuracy:.4f}")
                st.write(f"**AUC Score:** {auc:.4f}")
                st.write(f"**Precision:** {precision:.4f}")
                st.write(f"**Recall:** {recall:.4f}")
                st.write(f"**F1 Score:** {f1:.4f}")
                st.write(f"**MCC Score:** {mcc:.4f}")

                st.subheader("Classification Report:")
                report_str = classification_report(y_uploaded, y_pred)
                st.text(report_str)

                st.subheader("Confusion Matrix:")
                cm = confusion_matrix(y_uploaded, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Not Viral', 'Viral'],
                            yticklabels=['Not Viral', 'Viral'])
                ax_cm.set_xlabel('Predicted Label')
                ax_cm.set_ylabel('True Label')
                ax_cm.set_title(f'Confusion Matrix for {selected_model_name}')
                st.pyplot(fig_cm)
                plt.close(fig_cm) # Close the figure to prevent warning
            else:
                st.info("Since the target variable 'is_viral' was not present in the uploaded data, only predictions are available for the selected model.")
                predictions_df = pd.DataFrame({'Predicted Virality': selected_model_info['predict_func'](selected_model_info['model'], X_uploaded_processed)[0]}, index=df_uploaded.index)
                st.subheader("Predictions:")
                st.write(predictions_df.head())

        # --- Compare All Models Performance ---
        if st.button('Compare All Models Performance'):
            if y_uploaded is not None:
                st.subheader('Comparative Model Performance:')
                all_models_metrics = {}

                for model_name, model_info in models.items():
                    model = model_info['model']
                    predict_func = model_info['predict_func']

                    y_pred, y_pred_proba = predict_func(model, X_uploaded_processed)

                    # Calculate evaluation metrics
                    accuracy = accuracy_score(y_uploaded, y_pred)
                    auc = roc_auc_score(y_uploaded, y_pred_proba)
                    precision = precision_score(y_uploaded, y_pred)
                    recall = recall_score(y_uploaded, y_pred)
                    f1 = f1_score(y_uploaded, y_pred)
                    mcc = matthews_corrcoef(y_uploaded, y_pred)

                    all_models_metrics[model_name] = {
                        'Accuracy': accuracy,
                        'AUC': auc,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'MCC': mcc
                    }

                metrics_df = pd.DataFrame(all_models_metrics).T
                st.dataframe(metrics_df)

                # Plotting
                st.subheader('Visual Comparison of Model Performance:')
                sns.set_style('whitegrid')
                fig, ax = plt.subplots(figsize=(15, 8))
                metrics_df.plot(kind='bar', ax=ax, colormap='viridis')
                plt.title('Comparative Model Performance Metrics', fontsize=16)
                plt.xlabel('Model', fontsize=12)
                plt.ylabel('Score', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Close the figure to prevent warning
            else:
                st.warning("Cannot compare all models performance as the target variable 'is_viral' was not found in the uploaded data.")

    else:
        st.info("Please upload a CSV file to get started.")
