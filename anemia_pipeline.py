import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
import pickle
import warnings

warnings.filterwarnings('ignore')  

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")

def run_pipeline(data_path):
    # ====================================
    # STEP 1 — DATA UNDERSTANDING
    # ====================================
    print("--- STEP 1: DATA UNDERSTANDING ---")
    df = pd.read_csv(data_path)
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify numerical and categorical features
    # Based on content: Gender and Result are categorical (already encoded)
    # Others are numerical
    numerical_features = ['Hemoglobin', 'MCH', 'MCHC', 'MCV']
    categorical_features = ['Gender']
    target = 'Result'
    
    print("\nNumerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)
    
    print("\nClass Distribution:")
    print(df[target].value_counts(normalize=True))
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # ====================================
    # STEP 2 — DATA PREPROCESSING
    # ====================================
    print("\n--- STEP 2: DATA PREPROCESSING ---")
    
    # 1. Handle missing values (if any)
    if df.isnull().values.any():
        df = df.dropna() # Simple approach as requested "appropriately"
        print("Dropped rows with missing values.")
    
    # 2. Encode categorical features (Gender is already numeric, but let's ensure)
    # If Gender was 'Male'/'Female', we'd use OneHot or LabelEncoding.
    # Here it's 0/1.
    
    # 3. Detect and handle outliers (IQR method)
    def handle_outliers(data, columns):
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Capping instead of dropping to keep data size
            data[col] = np.clip(data[col], lower_bound, upper_bound)
        return data

    df = handle_outliers(df, numerical_features)
    print("Outliers handled using IQR capping.")

    # ====================================
    # STEP 3 — FEATURE ENGINEERING
    # ====================================
    print("\n--- STEP 3: FEATURE ENGINEERING ---")
    
    # Medically meaningful derived features
    # Hematocrit (HCT) is roughly 3 times Hemoglobin
    df['Hematocrit'] = df['Hemoglobin'] * 3 / 100 # Normally as percentage
    
    # Red Cell Distribution Width (RDW) - skip as we don't have enough info
    # Mean Corpuscular Volume (MCV) = (Hematocrit / RBC) * 10
    # Since we have MCV and Hematocrit, we can derive an estimated RBC count
    # RBC = (Hematocrit * 10) / MCV
    df['Estimated_RBC'] = (df['Hematocrit'] * 10) / df['MCV']
    
    print("Derived features 'Hematocrit' and 'Estimated_RBC' created.")
    
    new_numerical = numerical_features + ['Hematocrit', 'Estimated_RBC']
    
    # ====================================
    # STEP 4 — TRAIN-TEST SPLIT
    # ====================================
    print("\n--- STEP 4: TRAIN-TEST SPLIT ---")
    X = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # We should scale all features since they are all numeric now
    columns_to_scale = X_train.columns
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    # 5. Handle class imbalance using SMOTE if necessary
    if y_train.value_counts(normalize=True).min() < 0.4:
        print("Handling class imbalance using SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        print(f"Resampled training set: {X_train_res.shape}")
    else:
        X_train_res, y_train_res = X_train_scaled, y_train
        print("No significant class imbalance detected.")

    # ====================================
    # STEP 5 — MODEL TRAINING
    # ====================================
    print("\n--- STEP 5: MODEL TRAINING ---")
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    results = {}
    best_model = None
    best_auc = 0
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Hyperparameter tuning using GridSearchCV (simplified for speed but robust)
        param_grid = {}
        if name == "Logistic Regression":
            param_grid = {'C': [0.1, 1, 10]}
        elif name == "Random Forest":
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        elif name == "Gradient Boosting":
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
        elif name == "XGBoost":
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
        elif name == "Support Vector Machine":
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
            
        grid = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_res, y_train_res)
        
        best_clf = grid.best_estimator_
        y_pred = best_clf.predict(X_test_scaled)
        y_prob = best_clf.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Model': best_clf,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC': auc
        }
        
        print(f"  AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model = best_clf

    # ====================================
    # STEP 6 — MODEL SELECTION
    # ====================================
    print("\n--- STEP 6: MODEL SELECTION ---")
    results_df = pd.DataFrame(results).T.drop('Model', axis=1)
    print(results_df)
    
    print(f"\nBest Model: {best_model_name}")

    # ====================================
    # STEP 7 — FINAL EVALUATION
    # ====================================
    print("\n--- STEP 7: FINAL EVALUATION ---")
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance Visualization (if applicable)
    feature_names = X_train.columns
    if hasattr(best_model, 'feature_importances_'):
        feat_importances = pd.Series(best_model.feature_importances_, index=feature_names)
        plt.figure(figsize=(10,6))
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title(f'Feature Importances - {best_model_name}')
        plt.savefig('feature_importance.png')
        print("Feature importance saved as 'feature_importance.png'.")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {best_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    print("ROC curve saved as 'roc_curve.png'.")

    # ====================================
    # STEP 8 — MODEL OPTIMIZATION
    # ====================================
    print("\n--- STEP 8: MODEL OPTIMIZATION ---")
    if results[best_model_name]['Accuracy'] < 0.90:
        print(f"Accuracy ({results[best_model_name]['Accuracy']:.4f}) is below 90%. Trying an Ensemble (Voting Classifier)...")
        from sklearn.ensemble import VotingClassifier
        
        # Select top 3 models for voting based on AUC
        sorted_model_names = sorted(results.keys(), key=lambda x: results[x]['AUC'], reverse=True)
        top_models = [(name, results[name]['Model']) for name in sorted_model_names[:3]]
        
        voting_clf = VotingClassifier(estimators=top_models, voting='soft')
        voting_clf.fit(X_train_res, y_train_res)
        
        v_pred = voting_clf.predict(X_test_scaled)
        v_prob = voting_clf.predict_proba(X_test_scaled)[:, 1]
        v_acc = accuracy_score(y_test, v_pred)
        v_auc = roc_auc_score(y_test, v_prob)
        
        print(f"  Ensemble Accuracy: {v_acc:.4f}, AUC: {v_auc:.4f}")
        
        if v_acc > results[best_model_name]['Accuracy']:
            print("  Ensemble performed better. Updating best model.")
            best_model = voting_clf
            best_model_name = "Voting Ensemble"
            results[best_model_name] = {
                'Model': voting_clf,
                'Accuracy': v_acc,
                'AUC': v_auc
            }
    else:
        print("Accuracy is already above 90%. No additional ensemble optimization needed.")

    # ====================================
    # STEP 9 — SAVE MODEL
    # ====================================
    print("\n--- STEP 9: SAVE MODEL ---")
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'features': feature_names.tolist(),
        'best_model_name': best_model_name,
        'metrics': results[best_model_name]
    }
    with open('anemia_prediction_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model ({best_model_name}) saved as 'anemia_prediction_model.pkl'.")
    
    return model_data

def predict_anemia(hemoglobin, rbc, age, gender, mcv, mch, mchc, hematocrit=None):
    """
    Step 10: Prediction function.
    """
    try:
        with open('anemia_prediction_model.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return "Model file not found. Please run the pipeline first."
    
    model = data['model']
    scaler = data['scaler']
    features = data['features']
    
    # Pre-calculate derived features as done in training
    if hematocrit is None:
        hematocrit = float(hemoglobin) * 3 / 100
    
    est_rbc = (float(hematocrit) * 10) / float(mcv)
    
    # Prepare input dictionary
    input_data = {
        'Gender': int(gender),
        'Hemoglobin': float(hemoglobin),
        'MCH': float(mch),
        'MCHC': float(mchc),
        'MCV': float(mcv),
        'Hematocrit': float(hematocrit),
        'Estimated_RBC': est_rbc
    }
    
    # Ensure order matches features list
    input_df = pd.DataFrame([input_data])[features]
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prob = model.predict_proba(input_scaled)[0, 1]
    pred = model.predict(input_scaled)[0]
    
    return {
        'Anemia': 'Yes' if pred == 1 else 'No',
        'Probability': f"{prob:.2%}"
    }

if __name__ == "__main__":
    result_data = run_pipeline('c:/Users/Admin/OneDrive/Desktop/AJ projects/anemia.csv')
    
    print("\n" + "="*40)
    print("FINAL MODEL RESULTS")
    print("="*40)
    print(f"Best Model Name: {result_data['best_model_name']}")
    print(f"Final Model Accuracy: {result_data['metrics']['Accuracy']:.4f}")
    if 'AUC' in result_data['metrics']:
        print(f"ROC-AUC Score: {result_data['metrics']['AUC']:.4f}")
    
    # Example Prediction
    print("\n--- Example Prediction ---")
    # Using typical values from dataset: Gender=1, Hemoglobin=12.7, MCH=19.5, MCHC=28.9, MCV=82.9
    example = predict_anemia(
        hemoglobin=12.7, rbc=4.5, age=30, gender=1, 
        mcv=82.9, mch=19.5, mchc=28.9, hematocrit=0.381
    )
    print(f"Input: Hemoglobin=12.7, RBC=4.5, Age=30, Gender=Male, MCV=82.9, MCH=19.5, MCHC=28.9")
    print(f"Result: {example['Anemia']} (Probability: {example['Probability']})")
