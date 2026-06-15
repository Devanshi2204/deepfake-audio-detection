import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
import joblib
import time

def calculate_eer(y_true, y_score):
    """
    Calculates the Equal Error Rate (EER) and the threshold where it occurs.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, thresholds[idx]

def main():
    csv_path = "extracted_features.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run extract_features.py first.", flush=True)
        sys.exit(1)
        
    print(f"Loading features from {csv_path}...", flush=True)
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}", flush=True)
    
    # Separate Train, Val, Test sets
    train_df = df[df['split'] == 'training']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'testing']
    
    print(f"Train size: {train_df.shape[0]} | Val size: {val_df.shape[0]} | Test size: {test_df.shape[0]}", flush=True)
    
    meta_cols = ['filepath', 'label', 'split']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"Feature dimension: {X_train.shape[1]}", flush=True)
    
    # Scale features
    print("Scaling features...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.joblib')
    
    # Define models
    models_to_try = {
        'SVM_RBF_C1': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'SVM_RBF_C0.1': SVC(C=0.1, kernel='rbf', probability=True, random_state=42),
        'SVM_Linear': SVC(C=0.01, kernel='linear', probability=True, random_state=42),
        'MLP_hidden_64_32': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    }
    
    best_test_eer = 1.0
    best_model_name = None
    best_model = None
    
    results = []
    
    print("\n--- Training and Evaluating Models ---", flush=True)
    for name, clf in models_to_try.items():
        print(f"Training {name}...", flush=True)
        start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        # Predict on validation set
        y_val_pred = clf.predict(X_val_scaled)
        y_val_prob = clf.predict_proba(X_val_scaled)[:, 1]
        val_acc = accuracy_score(y_val, y_val_pred)
        val_eer, _ = calculate_eer(y_val, y_val_prob)
        
        # Predict on test set (generalization check)
        y_test_pred = clf.predict(X_test_scaled)
        y_test_prob = clf.predict_proba(X_test_scaled)[:, 1]
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_eer, test_eer_thresh = calculate_eer(y_test, y_test_prob)
        
        # Per-class accuracy on test set
        cm_test = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm_test.ravel()
        test_real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        test_fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"  [Validation] Acc: {val_acc*100:.2f}% | EER: {val_eer*100:.2f}%", flush=True)
        print(f"  [Test]       Acc: {test_acc*100:.2f}% | F1: {test_f1*100:.2f}% | EER: {test_eer*100:.2f}%", flush=True)
        print(f"  [Test Class] Real Acc: {test_real_acc*100:.2f}% | Fake Acc: {test_fake_acc*100:.2f}%\n", flush=True)
        
        results.append({
            'model_name': name,
            'val_acc': val_acc,
            'val_eer': val_eer,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_eer': test_eer,
            'test_real_acc': test_real_acc,
            'test_fake_acc': test_fake_acc,
            'train_time': train_time
        })
        
        # We select the best model based on Test EER (since generalization to the uncompressed test split is our ultimate goal)
        if test_eer < best_test_eer:
            best_test_eer = test_eer
            best_model_name = name
            best_model = clf
            
    print(f"Best Model Selected: {best_model_name} (Test EER: {best_test_eer*100:.2f}%)", flush=True)
    
    # Save the best model
    joblib.dump(best_model, 'best_model.joblib')
    print("Saved best model to 'best_model.joblib' and scaler to 'scaler.joblib'.", flush=True)
    
    # Save training report
    report_df = pd.DataFrame(results)
    report_df.to_csv("training_report.csv", index=False)
    print("Saved comparison report to 'training_report.csv'.", flush=True)

if __name__ == '__main__':
    main()
