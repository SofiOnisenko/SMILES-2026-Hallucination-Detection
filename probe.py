from __future__ import annotations

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class HallucinationProbe:
    """
    Ensemble probe for hallucination detection based on hidden states
    """

    def __init__(self) -> None:

        self.scaler = StandardScaler()

        self.lr = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )

        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
        )

        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-2,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            random_state=42,
        )

        self.threshold = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray):

        X_scaled = self.scaler.fit_transform(X)

        self.lr.fit(X_scaled, y)
        self.rf.fit(X_scaled, y)
        self.mlp.fit(X_scaled, y)

        return self

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):

        probs = self.predict_proba(X_val)[:, 1]

        thresholds = np.linspace(0.2, 0.8, 61)

        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in thresholds:

            preds = (probs >= threshold).astype(int)

            score = f1_score(
                y_val,
                preds,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = threshold

        self.threshold = best_threshold

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:

        probs = self.predict_proba(X)[:, 1]

        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        X_scaled = self.scaler.transform(X)

        lr_prob = self.lr.predict_proba(X_scaled)[:, 1]
        rf_prob = self.rf.predict_proba(X_scaled)[:, 1]
        mlp_prob = self.mlp.predict_proba(X_scaled)[:, 1]

        prob_pos = (
            0.4 * lr_prob +
            0.3 * rf_prob +
            0.3 * mlp_prob
        )

        return np.stack(
            [1.0 - prob_pos, prob_pos],
            axis=1,
        )