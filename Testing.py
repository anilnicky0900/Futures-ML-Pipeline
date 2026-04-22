"""
MODULE 3 — TESTING / PREDICTION
=================================
LLM Prompt used to generate this module:
  "Use a trained sklearn classifier to predict on unseen test data.
   Apply the same StandardScaler that was fitted during training.
   Return:
     y_true  — binary ground-truth labels (+1 / -1) derived from y_test
     y_pred  — predicted binary labels
     y_proba — probability of the positive class (for ROC curve)"

INPUT : model (fitted classifier), scaler (fitted StandardScaler),
        X_test (ndarray), y_test (ndarray of continuous % changes)
OUTPUT: y_true (ndarray), y_pred (ndarray), y_proba (ndarray)
"""

import numpy as np


def run_testing(model, scaler,
                X_test: np.ndarray,
                y_test: np.ndarray):
    """
    Scales test features with training scaler, predicts labels + probabilities.
    """
    y_true   = np.where(y_test > 0, 1, -1)
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        pos_idx = classes.index(1) if 1 in classes else 0
        y_proba = model.predict_proba(X_scaled)[:, pos_idx]
    else:
        y_proba = y_pred.astype(float)

    return y_true, y_pred, y_proba
