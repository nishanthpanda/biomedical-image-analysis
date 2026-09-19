import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

DATA_PATH = 'data/synthetic_clinical_dataset.csv'

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Preprocessing
    print("Preprocessing data...")
    # Drop patient_id as it's not a predictive feature
    if 'patient_id' in df.columns:
        df = df.drop('patient_id', axis=1)
        
    # Handle categorical variables (e.g., 'sex', 'diagnosis')
    # For this example, let's predict 'mortality' (binary classification)
    target = 'mortality'
    
    # We need to encode 'sex' and 'diagnosis' if we are using them as features
    label_encoders = {}
    categorical_cols = ['sex', 'diagnosis']
    
    for col in categorical_cols:
        if col in df.columns and col != target:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            
    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    # We apply scaling to all features here (LabelEncoded categorical features will also be scaled, 
    # which is generally fine for trees, but for other models OneHotEncoding is preferred)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Model Training
    print("Training XGBoost Classifier on GPU...")
    # Using 'hist' tree method and specifying 'cuda' device for GPU acceleration
    model = XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        tree_method='hist', 
        device='cuda'
    )
    model.fit(X_train_scaled, y_train)
    
    # 3. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head())
    
    # 4. Save Model and Scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'clinical_rf_model.joblib')
    joblib.dump(scaler, 'clinical_scaler.joblib')
    # Save encoders for future inference
    joblib.dump(label_encoders, 'clinical_label_encoders.joblib')
    print("Saved to 'clinical_rf_model.joblib' and 'clinical_scaler.joblib'.")

if __name__ == "__main__":
    main()
