"""
MODULE 2 — MODEL TRAINING
==========================
LLM Prompt used to generate this module:
  "Train a binary classifier on financial time-series features.
   Convert continuous target y (% price change) to binary labels:
     +1 if price went up (y > 0), -1 if price went down (y <= 0).
   Support two model types selectable by a string argument:
     'svm'            — SVC with RBF kernel, probability=True
     'gradient_boost' — GradientBoostingClassifier
   Scale features with StandardScaler before training.
   Return the trained model and the fitted scaler."

INPUT : X_train (ndarray), y_train (ndarray), model_type (str)
OUTPUT: model (fitted classifier), scaler (fitted StandardScaler)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                model_type: str = "svm"):
    """
    Converts continuous y to binary labels then trains chosen model.
    """
    labels = np.where(y_train > 0, 1, -1)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if model_type == "svm":
        model = SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, random_state=42)

    elif model_type == "gradient_boost":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose 'svm' or 'gradient_boost'.")

    model.fit(X_scaled, labels)
    return model, scaler
