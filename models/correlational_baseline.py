"""
Correlational Baseline: Random Forest with naive fine-tuning

This model learns correlations between features and fraud labels,
without explicit causal structure. It uses naive continual learning
(fine-tuning) which is expected to cause catastrophic forgetting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle


class CorrelationalBaseline:
    """
    Random Forest baseline for fraud detection.

    This model:
    - Learns correlations between features and fraud
    - Uses standard supervised learning
    - Continual learning via naive fine-tuning (overwrites weights)
    - Expected to suffer from catastrophic forgetting
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize correlational baseline.

        Args:
            n_estimators: Number of trees in random forest
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',  # Handle imbalance
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X, y, scale=True):
        """
        Train the model.

        Args:
            X: Features (DataFrame or numpy array)
            y: Labels
            scale: Whether to scale features

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        if scale:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict_proba(self, X, scale=True):
        """
        Predict fraud probabilities.

        Args:
            X: Features
            scale: Whether to scale features

        Returns:
            Array of fraud probabilities (class 1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if scale:
            X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Probability of fraud (class 1)

    def predict(self, X, threshold=0.5, scale=True):
        """
        Predict fraud labels.

        Args:
            X: Features
            threshold: Decision threshold
            scale: Whether to scale features

        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X, scale=scale)
        return (proba >= threshold).astype(int)

    def continual_update(self, X_new, y_new, scale=True):
        """
        Continual learning: Naive fine-tuning.

        This method OVERWRITES the model with new data, which is
        expected to cause catastrophic forgetting of old patterns.

        Args:
            X_new: New features
            y_new: New labels
            scale: Whether to scale features

        Returns:
            self
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first before updating.")

        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values

        if scale:
            # Update scaler with new data (this itself can cause issues)
            X_new = self.scaler.fit_transform(X_new)

        # Naive fine-tuning: retrain on new data only
        # This is the KEY problem: old patterns are forgotten!
        self.model.fit(X_new, y_new)

        print("[WARNING] Naive fine-tuning: Old patterns may be forgotten!")

        return self

    def get_feature_importance(self):
        """Get feature importances"""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        importances = self.model.feature_importances_
        if self.feature_names:
            return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        return importances

    def save(self, filepath):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'feature_names': self.feature_names
            }, f)

    def load(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_fitted = data['is_fitted']
            self.feature_names = data['feature_names']


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('..')

    # Load toy dataset
    transactions = pd.read_pickle("../results/data/toy_transactions.pkl")
    customers = pd.read_pickle("../results/data/toy_customers.pkl")

    from utils import extract_features, evaluate_model, print_metrics

    # Split periods
    period1_mask = transactions['TX_TIME_DAYS'] < 30
    period2_mask = transactions['TX_TIME_DAYS'] >= 30

    period1_data = transactions[period1_mask]
    period2_data = transactions[period2_mask]

    # Extract features
    X_period1 = extract_features(period1_data, customers)
    y_period1 = period1_data['TX_FRAUD'].values

    X_period2 = extract_features(period2_data, customers)
    y_period2 = period2_data['TX_FRAUD'].values

    print("=" * 70)
    print("CORRELATIONAL BASELINE TEST")
    print("=" * 70)

    # Train on Period 1
    print("\n[1/4] Training on Period 1...")
    model = CorrelationalBaseline(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_period1, y_period1)

    # Evaluate on Period 1
    print("\n[2/4] Evaluating on Period 1 (before CL)...")
    y_pred_p1_before = model.predict_proba(X_period1)
    metrics_p1_before = evaluate_model(
        y_period1, y_pred_p1_before,
        customer_ids=period1_data['CUSTOMER_ID'].values,
        day_col=period1_data['TX_TIME_DAYS'].values
    )
    print_metrics(metrics_p1_before, "Period 1 Performance (Initial)")

    # Continual learning: Fine-tune on Period 2
    print("\n[3/4] Continual learning on Period 2...")
    model.continual_update(X_period2, y_period2)

    # Evaluate on Period 2 (Forward Transfer)
    print("\n[4/4] Evaluating on Period 2 (Forward Transfer)...")
    y_pred_p2 = model.predict_proba(X_period2)
    metrics_p2 = evaluate_model(
        y_period2, y_pred_p2,
        customer_ids=period2_data['CUSTOMER_ID'].values,
        day_col=period2_data['TX_TIME_DAYS'].values
    )
    print_metrics(metrics_p2, "Period 2 Performance (After CL)")

    # Evaluate on Period 1 again (Backward Transfer - FORGETTING)
    print("\nEvaluating on Period 1 again (Backward Transfer)...")
    y_pred_p1_after = model.predict_proba(X_period1)
    metrics_p1_after = evaluate_model(
        y_period1, y_pred_p1_after,
        customer_ids=period1_data['CUSTOMER_ID'].values,
        day_col=period1_data['TX_TIME_DAYS'].values
    )
    print_metrics(metrics_p1_after, "Period 1 Performance (After CL)")

    # Compute Backward Transfer
    print("\n" + "=" * 70)
    print("BACKWARD TRANSFER (FORGETTING)")
    print("=" * 70)

    for metric in ['AUC_ROC', 'Average_Precision', 'Card_Precision@100']:
        before = metrics_p1_before.get(metric, np.nan)
        after = metrics_p1_after.get(metric, np.nan)
        bt = after - before

        print(f"\n{metric}:")
        print(f"  Before CL: {before:.4f}")
        print(f"  After CL:  {after:.4f}")
        print(f"  Backward Transfer: {bt:+.4f} {'(FORGETTING)' if bt < -0.05 else '(OK)'}")

    print("\n[SUCCESS] Correlational baseline test complete!")
