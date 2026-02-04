"""
Utility functions for model training and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score


def extract_features(transactions_df, customer_profiles=None):
    """
    Extract features for fraud detection from transaction data.

    Features include:
    - Transaction amount
    - Time of day
    - Customer aggregates (if customer profiles provided)

    Args:
        transactions_df: Transaction DataFrame
        customer_profiles: Optional customer profiles for aggregates

    Returns:
        DataFrame with features
    """
    features = pd.DataFrame()

    # Basic transaction features
    features['TX_AMOUNT'] = transactions_df['TX_AMOUNT']
    features['TX_TIME_SECONDS'] = transactions_df['TX_TIME_SECONDS']
    features['TX_TIME_DAYS'] = transactions_df['TX_TIME_DAYS']

    # Time of day (hour)
    features['HOUR_OF_DAY'] = (transactions_df['TX_TIME_SECONDS'] % 86400) / 3600

    # Customer ID as categorical (for now, just numeric)
    features['CUSTOMER_ID'] = transactions_df['CUSTOMER_ID']
    features['TERMINAL_ID'] = transactions_df['TERMINAL_ID']

    # If customer profiles available, add aggregates
    if customer_profiles is not None:
        customer_agg = customer_profiles[['CUSTOMER_ID', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day']]
        features = features.merge(customer_agg, on='CUSTOMER_ID', how='left')

        # Deviation from customer's normal pattern
        features['AMOUNT_DEVIATION'] = (features['TX_AMOUNT'] - features['mean_amount']) / (features['std_amount'] + 1e-6)

    # Drop IDs (not useful for most models)
    features_for_model = features.drop(columns=['CUSTOMER_ID', 'TERMINAL_ID'], errors='ignore')

    return features_for_model


def card_precision_at_k(y_true, y_pred_proba, customer_ids, k=100, per_day=True, day_col=None):
    """
    Compute Card Precision@k metric.

    For each day (if per_day=True), identify top-k most suspicious cards
    and compute precision of detecting compromised cards.

    Args:
        y_true: True labels
        y_pred_proba: Predicted fraud probabilities
        customer_ids: Customer IDs for transactions
        k: Number of top cards to flag per day
        per_day: Whether to compute per day or overall
        day_col: Column with day information (if per_day=True)

    Returns:
        Mean card precision@k
    """
    df = pd.DataFrame({
        'TX_FRAUD': y_true,
        'predictions': y_pred_proba,
        'CUSTOMER_ID': customer_ids
    })

    if per_day and day_col is not None:
        df['TX_TIME_DAYS'] = day_col

        # Group by day and customer, take max fraud prob and max fraud label
        daily_customer = df.groupby(['TX_TIME_DAYS', 'CUSTOMER_ID']).agg({
            'predictions': 'max',
            'TX_FRAUD': 'max'
        }).reset_index()

        precisions = []
        for day in daily_customer['TX_TIME_DAYS'].unique():
            day_data = daily_customer[daily_customer['TX_TIME_DAYS'] == day]
            top_k = day_data.nlargest(min(k, len(day_data)), 'predictions')
            if len(top_k) > 0:
                precision = top_k['TX_FRAUD'].sum() / len(top_k)
                precisions.append(precision)

        return np.mean(precisions) if precisions else 0.0
    else:
        # Overall: group by customer
        customer_agg = df.groupby('CUSTOMER_ID').agg({
            'predictions': 'max',
            'TX_FRAUD': 'max'
        }).reset_index()

        top_k = customer_agg.nlargest(min(k, len(customer_agg)), 'predictions')
        if len(top_k) > 0:
            return top_k['TX_FRAUD'].sum() / len(top_k)
        return 0.0


def evaluate_model(y_true, y_pred_proba, customer_ids=None, day_col=None, k=100):
    """
    Comprehensive model evaluation.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        customer_ids: Customer IDs (for CP@k)
        day_col: Day column (for CP@k)
        k: Top-k for card precision

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Standard metrics
    if len(np.unique(y_true)) > 1:  # Need both classes
        metrics['AUC_ROC'] = roc_auc_score(y_true, y_pred_proba)
        metrics['Average_Precision'] = average_precision_score(y_true, y_pred_proba)
    else:
        metrics['AUC_ROC'] = np.nan
        metrics['Average_Precision'] = np.nan

    # Card Precision@k (if customer IDs provided)
    if customer_ids is not None:
        metrics[f'Card_Precision@{k}'] = card_precision_at_k(
            y_true, y_pred_proba, customer_ids, k=k, per_day=(day_col is not None), day_col=day_col
        )

    # Binary predictions at threshold 0.5
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    if len(np.unique(y_true)) > 1 and y_pred_binary.sum() > 0:
        metrics['Precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
    else:
        metrics['Precision'] = 0.0
        metrics['Recall'] = 0.0

    return metrics


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics dictionary"""
    print(f"\n{title}")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
