import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def load_and_prepare_data(filepath='data/sample_customers.csv'):
    """Load and prepare customer data"""
    df = pd.read_csv(filepath)
    
    # Features and target
    X = df.drop(['customer_id', 'churned'], axis=1)
    y = df['churned']
    
    return df, X, y

def train_models(X, y):
    """Train multiple ML models"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Random Forest ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")
    print(f"Random Forest Accuracy: {rf_model.score(X_test_scaled, y_test):.4f}")
    
    # Train Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_proba = gb_model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"Gradient Boosting ROC-AUC: {roc_auc_score(y_test, gb_proba):.4f}")
    print(f"Gradient Boosting Accuracy: {gb_model.score(X_test_scaled, y_test):.4f}")
    
    # Save models and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(gb_model, 'models/gb_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
        
    # Print feature importance
    print("\nFeature Importance (Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.to_string(index=False))
    
    return rf_model, gb_model, scaler, X.columns

if __name__ == "__main__":
    print("=" * 50)
    print("CHURN PREDICTION MODEL TRAINING")
    print("=" * 50)
    
    # Load and prepare data
    df, X, y = load_and_prepare_data()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Churn rate: {y.mean():.2%}")
    print(f"\nFeatures ({len(X.columns)}):")
    print(X.columns.tolist())
    
    # Train models
    train_models(X, y)
    
    print("\n" + "=" * 50)
    print("Training complete! Ready to run Streamlit app.")
    print("=" * 50)