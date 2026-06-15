import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

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
    model_path = "best_model.joblib"
    scaler_path = "scaler.joblib"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run extract_features.py first.", flush=True)
        sys.exit(1)
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Model or Scaler not found. Run train_model.py first.", flush=True)
        sys.exit(1)
        
    print("Loading test features...", flush=True)
    df = pd.read_csv(csv_path)
    test_df = df[df['split'] == 'testing']
    
    if len(test_df) == 0:
        print("Error: No test samples found in the CSV.", flush=True)
        sys.exit(1)
        
    print(f"Loaded {test_df.shape[0]} test samples.", flush=True)
    
    # Extract feature columns
    meta_cols = ['filepath', 'label', 'split']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Load model and scaler
    print("Loading model and scaler...", flush=True)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    print("Evaluating model...", flush=True)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    eer, eer_threshold = calculate_eer(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n================ FINAL EVALUATION REPORT ================", flush=True)
    print(f"Model Evaluated: {type(model).__name__}", flush=True)
    print(f"Overall Accuracy: {acc*100:.2f}%  (Target >= 80%)", flush=True)
    print(f"F1 Score:         {f1*100:.2f}%  (Target >= 80%)", flush=True)
    print(f"Equal Error Rate: {eer*100:.2f}%  (Target <= 12%)", flush=True)
    print(f"  EER Threshold:  {eer_threshold:.4f}", flush=True)
    print(f"Per-Class Accuracy:", flush=True)
    print(f"  - Genuine (Real) Speech (Class 0): {real_acc*100:.2f}%  (Target >= 75%)", flush=True)
    print(f"  - Deepfake (Fake) Speech (Class 1): {fake_acc*100:.2f}%  (Target >= 75%)", flush=True)
    print("\nConfusion Matrix:", flush=True)
    print(f"               Predicted Real | Predicted Fake", flush=True)
    print(f"Actual Real :  {tn:<14} | {fp:<14}", flush=True)
    print(f"Actual Fake :  {fn:<14} | {tp:<14}", flush=True)
    print("========================================================\n", flush=True)
    
    # Write summary metrics report to a file
    with open("evaluation_report.txt", "w") as f:
        f.write("================ FINAL EVALUATION REPORT ================\n")
        f.write(f"Model Evaluated: {type(model).__name__}\n")
        f.write(f"Overall Accuracy: {acc*100:.2f}%\n")
        f.write(f"F1 Score:         {f1*100:.2f}%\n")
        f.write(f"Equal Error Rate: {eer*100:.2f}%\n")
        f.write(f"  EER Threshold:  {eer_threshold:.4f}\n")
        f.write(f"Per-Class Accuracy:\n")
        f.write(f"  - Genuine (Real) Speech: {real_acc*100:.2f}%\n")
        f.write(f"  - Deepfake (Fake) Speech: {fake_acc*100:.2f}%\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"               Predicted Real | Predicted Fake\n")
        f.write(f"Actual Real :  {tn:<14} | {fp:<14}\n")
        f.write(f"Actual Fake :  {fn:<14} | {tp:<14}\n")
        f.write("========================================================\n")
        
    print("Plotting Confusion Matrix...", flush=True)
    
    # Create a beautiful, premium confusion matrix visualization
    plt.figure(figsize=(7, 6))
    
    # Custom vibrant blue/indigo palette
    # Use standard matplotlib colors but with custom styling
    colors = [[1,1,1], [0.8,0.9,1], [0.4,0.6,0.95], [0.1,0.3,0.7]]
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)
    
    plt.imshow(cm, interpolation='nearest', cmap=custom_cmap)
    plt.title('Confusion Matrix - Deepfake Audio Detector', fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(2)
    classes = ['Genuine (Real)', 'Deepfake (Fake)']
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11, rotation=90, va="center")
    
    # Annotate matrix with counts and percentages
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / np.sum(cm[i, :]) * 100
            text_color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val}\n({pct:.1f}%)",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color=text_color,
                     fontsize=12,
                     fontweight='bold')
                     
    plt.tight_layout()
    plt.ylabel('Actual Label', fontsize=12, fontweight='semibold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='semibold')
    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix visualization to 'confusion_matrix.png'.", flush=True)

if __name__ == '__main__':
    main()
