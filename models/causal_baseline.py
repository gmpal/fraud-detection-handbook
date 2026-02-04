"""
Causal Baseline: Oracle model with causal features

This model uses knowledge of the TRUE causal structure to create
invariant features that generalize across concept drift.

Instead of learning "TX_AMOUNT > 220 -> fraud", it learns:
"Deviation from customer's normal pattern -> fraud"

This represents an upper bound: what if the model knew the causal mechanism?
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle


class CausalBaseline:
    """
    Causal-aware fraud detection model.

    Key idea: Extract features based on CAUSAL mechanisms, not just correlations.

    Instead of raw features:
    - TX_AMOUNT (Period 1: >500 = fraud, Period 2: ~120 = fraud) <- Changes!

    Use causal features:
    - AMOUNT_DEVIATION = (TX_AMOUNT - customer_mean) / customer_std <- Invariant!
    - FREQUENCY_DEVIATION = (frequency - normal_frequency) <- Invariant!

    These features capture "abnormal behavior" which is the TRUE cause of fraud,
    regardless of whether the abnormality manifests as high amount or high frequency.
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize causal baseline.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

        # Store old model for ensemble (simple continual learning strategy)
        self.old_model = None
        self.ensemble_weight = 0.5  # Weight for old model in predictions

    def extract_causal_features(self, transactions, customers):
        """
        Extract causal features based on known causal structure.

        The key insight: "Deviation from normal" is causal, not absolute values.

        Args:
            transactions: Transaction DataFrame
            customers: Customer profiles DataFrame

        Returns:
            DataFrame with causal features
        """
        features = pd.DataFrame()

        # Merge with customer profiles to get baselines
        tx_with_customers = transactions.merge(
            customers[['CUSTOMER_ID', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day']],
            on='CUSTOMER_ID',
            how='left'
        )

        # CAUSAL FEATURE 1: Amount deviation (normalized by customer's pattern)
        # This is invariant: "unusually high" is fraud, regardless of absolute value
        features['AMOUNT_DEVIATION'] = (
            (tx_with_customers['TX_AMOUNT'] - tx_with_customers['mean_amount']) /
            (tx_with_customers['std_amount'] + 1e-6)
        )

        # CAUSAL FEATURE 2: Absolute amount (for very extreme cases)
        features['TX_AMOUNT_LOG'] = np.log1p(tx_with_customers['TX_AMOUNT'])

        # CAUSAL FEATURE 3: Time-of-day deviation
        hour_of_day = (tx_with_customers['TX_TIME_SECONDS'] % 86400) / 3600
        # Unusual time (very early morning or very late night)
        features['TIME_DEVIATION'] = np.abs(hour_of_day - 12)  # Deviation from noon

        # CAUSAL FEATURE 4: Transaction velocity (requires aggregation)
        # Count transactions per customer per day
        tx_counts = tx_with_customers.groupby(['CUSTOMER_ID', 'TX_TIME_DAYS']).size().reset_index(name='daily_tx_count')
        tx_with_counts = tx_with_customers.merge(tx_counts, on=['CUSTOMER_ID', 'TX_TIME_DAYS'], how='left')

        features['FREQUENCY_DEVIATION'] = (
            tx_with_counts['daily_tx_count'] / (tx_with_counts['mean_nb_tx_per_day'] + 1e-6)
        )

        # Basic features
        features['TX_TIME_DAYS'] = tx_with_customers['TX_TIME_DAYS']

        return features

    def fit(self, transactions, customers, scale=True):
        """
        Train the model on causal features.

        Args:
            transactions: Transaction DataFrame
            customers: Customer profiles DataFrame
            scale: Whether to scale features

        Returns:
            self
        """
        X = self.extract_causal_features(transactions, customers)
        y = transactions['TX_FRAUD'].values

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values

        if scale:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self.is_fitted = True

        print("[INFO] Trained on causal features (deviations from normal behavior)")

        return self

    def predict_proba(self, transactions, customers, scale=True, use_ensemble=False):
        """
        Predict fraud probabilities using causal features.

        Args:
            transactions: Transaction DataFrame
            customers: Customer profiles DataFrame
            scale: Whether to scale
            use_ensemble: Whether to ensemble with old model (for continual learning)

        Returns:
            Array of fraud probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X = self.extract_causal_features(transactions, customers)

        if isinstance(X, pd.DataFrame):
            X = X.values

        if scale:
            X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)[:, 1]

        # If continual learning with ensemble, combine old and new model
        if use_ensemble and self.old_model is not None:
            proba_old = self.old_model.predict_proba(X)[:, 1]
            proba = self.ensemble_weight * proba_old + (1 - self.ensemble_weight) * proba

        return proba

    def predict(self, transactions, customers, threshold=0.5, scale=True, use_ensemble=False):
        """Predict fraud labels"""
        proba = self.predict_proba(transactions, customers, scale=scale, use_ensemble=use_ensemble)
        return (proba >= threshold).astype(int)

    def continual_update(self, transactions_new, customers, scale=True, preserve_old=True):
        """
        Continual learning with causal structure preservation.

        Key difference from correlational model:
        - Old model is PRESERVED and ensembled
        - Causal features are re-computed (adapt to new patterns)
        - Old knowledge is not forgotten (ensemble maintains it)

        Args:
            transactions_new: New transaction data
            customers: Customer profiles
            scale: Whether to scale
            preserve_old: Whether to preserve old model (ensemble)

        Returns:
            self
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # PRESERVE old model
        if preserve_old:
            self.old_model = pickle.loads(pickle.dumps(self.model))  # Deep copy
            print("[INFO] Old model preserved for ensemble")

        # Extract causal features from new data
        X_new = self.extract_causal_features(transactions_new, customers)
        y_new = transactions_new['TX_FRAUD'].values

        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values

        if scale:
            # Note: We update scaler, but this is less problematic because
            # causal features (deviations) are already normalized
            X_new = self.scaler.fit_transform(X_new)

        # Train new model on new data
        self.model.fit(X_new, y_new)

        print("[INFO] Model updated with new causal patterns")
        if preserve_old:
            print(f"[INFO] Ensemble: {self.ensemble_weight:.0%} old + {1-self.ensemble_weight:.0%} new")

        return self

    def get_feature_importance(self):
        """Get feature importances"""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        importances = self.model.feature_importances_
        if self.feature_names:
            return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        return importances


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('..')

    from utils import evaluate_model, print_metrics

    # Load toy dataset
    transactions = pd.read_pickle("../results/data/toy_transactions.pkl")
    customers = pd.read_pickle("../results/data/toy_customers.pkl")

    # Split periods
    period1_data = transactions[transactions['TX_TIME_DAYS'] < 30]
    period2_data = transactions[transactions['TX_TIME_DAYS'] >= 30]

    print("=" * 70)
    print("CAUSAL BASELINE TEST")
    print("=" * 70)

    # Train on Period 1
    print("\n[1/4] Training on Period 1 with causal features...")
    model = CausalBaseline(n_estimators=100, max_depth=10, random_state=42)
    model.fit(period1_data, customers)

    print("\nFeature importances:")
    print(model.get_feature_importance())

    # Evaluate on Period 1
    print("\n[2/4] Evaluating on Period 1 (before CL)...")
    y_pred_p1_before = model.predict_proba(period1_data, customers)
    metrics_p1_before = evaluate_model(
        period1_data['TX_FRAUD'].values,
        y_pred_p1_before,
        customer_ids=period1_data['CUSTOMER_ID'].values,
        day_col=period1_data['TX_TIME_DAYS'].values
    )
    print_metrics(metrics_p1_before, "Period 1 Performance (Initial)")

    # Continual learning: Update with Period 2
    print("\n[3/4] Continual learning on Period 2 (with preservation)...")
    model.continual_update(period2_data, customers, preserve_old=True)

    # Evaluate on Period 2 (Forward Transfer)
    print("\n[4/4] Evaluating on Period 2 (Forward Transfer)...")
    y_pred_p2 = model.predict_proba(period2_data, customers, use_ensemble=True)
    metrics_p2 = evaluate_model(
        period2_data['TX_FRAUD'].values,
        y_pred_p2,
        customer_ids=period2_data['CUSTOMER_ID'].values,
        day_col=period2_data['TX_TIME_DAYS'].values
    )
    print_metrics(metrics_p2, "Period 2 Performance (After CL)")

    # Evaluate on Period 1 again (Backward Transfer)
    print("\nEvaluating on Period 1 again (Backward Transfer)...")
    y_pred_p1_after = model.predict_proba(period1_data, customers, use_ensemble=True)
    metrics_p1_after = evaluate_model(
        period1_data['TX_FRAUD'].values,
        y_pred_p1_after,
        customer_ids=period1_data['CUSTOMER_ID'].values,
        day_col=period1_data['TX_TIME_DAYS'].values
    )
    print_metrics(metrics_p1_after, "Period 1 Performance (After CL)")

    # Compute Backward Transfer
    print("\n" + "=" * 70)
    print("BACKWARD TRANSFER (Should be MINIMAL)")
    print("=" * 70)

    for metric in ['AUC_ROC', 'Average_Precision', 'Card_Precision@100']:
        before = metrics_p1_before.get(metric, np.nan)
        after = metrics_p1_after.get(metric, np.nan)
        bt = after - before

        print(f"\n{metric}:")
        print(f"  Before CL: {before:.4f}")
        print(f"  After CL:  {after:.4f}")
        print(f"  Backward Transfer: {bt:+.4f} {'(GOOD!)' if bt > -0.1 else '(FORGETTING)'}")

    print("\n[SUCCESS] Causal baseline test complete!")
