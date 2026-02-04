"""
Core simulator functions for generating transaction data

Extracted and refactored from Chapter_3_GettingStarted/SimulatedDataset.ipynb
to provide a modular, reusable interface for the fraud detection simulator.
"""

import os
import numpy as np
import pandas as pd
import datetime
import time
import random
from typing import Tuple, Optional


def generate_customer_profiles_table(n_customers: int, random_state: int = 0) -> pd.DataFrame:
    """
    Generate customer profiles with spending characteristics.

    Each customer is characterized by:
    - Geographic location (x, y coordinates in 100x100 grid)
    - Transaction amount distribution (mean, std)
    - Transaction frequency (mean transactions per day)

    Args:
        n_customers: Number of customer profiles to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns: CUSTOMER_ID, x_customer_id, y_customer_id,
                                mean_amount, std_amount, mean_nb_tx_per_day
    """
    np.random.seed(random_state)

    customer_id_properties = []

    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)

        mean_amount = np.random.uniform(5, 100)
        std_amount = mean_amount / 2

        mean_nb_tx_per_day = np.random.uniform(0, 4)

        customer_id_properties.append([
            customer_id,
            x_customer_id, y_customer_id,
            mean_amount, std_amount,
            mean_nb_tx_per_day
        ])

    customer_profiles_table = pd.DataFrame(
        customer_id_properties,
        columns=[
            'CUSTOMER_ID',
            'x_customer_id', 'y_customer_id',
            'mean_amount', 'std_amount',
            'mean_nb_tx_per_day'
        ]
    )

    return customer_profiles_table


def generate_terminal_profiles_table(n_terminals: int, random_state: int = 0) -> pd.DataFrame:
    """
    Generate terminal profiles with geographic locations.

    Args:
        n_terminals: Number of terminal profiles to generate
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns: TERMINAL_ID, x_terminal_id, y_terminal_id
    """
    np.random.seed(random_state)

    terminal_id_properties = []

    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)

        terminal_id_properties.append([
            terminal_id,
            x_terminal_id, y_terminal_id
        ])

    terminal_profiles_table = pd.DataFrame(
        terminal_id_properties,
        columns=['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id']
    )

    return terminal_profiles_table


def get_list_terminals_within_radius(
    customer_profile: pd.Series,
    x_y_terminals: np.ndarray,
    r: float
) -> list:
    """
    Find terminals within radius r of a customer's location.

    Args:
        customer_profile: Row from customer profiles table
        x_y_terminals: Numpy array of terminal (x, y) coordinates
        r: Radius threshold

    Returns:
        List of terminal IDs within radius r
    """
    # Customer location as numpy array
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)

    # Euclidean distance to all terminals
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))

    # Get terminals within radius
    available_terminals = list(np.where(dist_x_y < r)[0])

    return available_terminals


def generate_transactions_table(
    customer_profile: pd.Series,
    start_date: str = "2018-04-01",
    nb_days: int = 10
) -> pd.DataFrame:
    """
    Generate transactions for a single customer over a period of days.

    Args:
        customer_profile: Row from customer profiles table (must include available_terminals)
        start_date: Starting date for transactions (format: YYYY-MM-DD)
        nb_days: Number of days to generate transactions for

    Returns:
        DataFrame with transactions for this customer
    """
    customer_transactions = []

    # Set random seed based on customer ID for reproducibility
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))

    for day in range(nb_days):
        # Sample number of transactions for this day
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)

        if nb_tx > 0:
            for tx in range(nb_tx):
                # Transaction time: centered around noon with std of ~5.5 hours
                time_tx = int(np.random.normal(86400/2, 20000))

                # Only keep valid transaction times (0-24 hours)
                if 0 < time_tx < 86400:
                    # Transaction amount from customer's distribution
                    amount = np.random.normal(
                        customer_profile.mean_amount,
                        customer_profile.std_amount
                    )

                    # Handle negative amounts
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)

                    amount = np.round(amount, decimals=2)

                    # Select random terminal from available terminals
                    if len(customer_profile.available_terminals) > 0:
                        terminal_id = random.choice(customer_profile.available_terminals)

                        customer_transactions.append([
                            time_tx + day * 86400,  # TX_TIME_SECONDS
                            day,                     # TX_TIME_DAYS
                            customer_profile.CUSTOMER_ID,
                            terminal_id,
                            amount
                        ])

    # Convert to DataFrame
    customer_transactions = pd.DataFrame(
        customer_transactions,
        columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT']
    )

    # Add datetime column
    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(
            customer_transactions["TX_TIME_SECONDS"],
            unit='s',
            origin=start_date
        )
        customer_transactions = customer_transactions[[
            'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID',
            'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS'
        ]]

    return customer_transactions


def generate_dataset(
    n_customers: int = 10000,
    n_terminals: int = 1000,
    nb_days: int = 90,
    start_date: str = "2018-04-01",
    r: float = 5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate complete dataset: customer profiles, terminal profiles, and transactions.

    This is the main entry point for generating synthetic transaction data.

    Args:
        n_customers: Number of customers
        n_terminals: Number of terminals
        nb_days: Number of days of transactions to generate
        start_date: Starting date (format: YYYY-MM-DD)
        r: Radius for associating customers to terminals
        verbose: Print timing information

    Returns:
        Tuple of (customer_profiles, terminal_profiles, transactions)
    """
    # Generate customer profiles
    start_time = time.time()
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state=0)
    if verbose:
        print(f"Time to generate customer profiles table: {time.time()-start_time:.2f}s")

    # Generate terminal profiles
    start_time = time.time()
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state=1)
    if verbose:
        print(f"Time to generate terminal profiles table: {time.time()-start_time:.2f}s")

    # Associate terminals to customers
    start_time = time.time()
    x_y_terminals = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r),
        axis=1
    )
    customer_profiles_table['nb_terminals'] = customer_profiles_table.available_terminals.apply(len)
    if verbose:
        print(f"Time to associate terminals to customers: {time.time()-start_time:.2f}s")

    # Generate transactions
    start_time = time.time()
    transactions_df = customer_profiles_table.groupby('CUSTOMER_ID').apply(
        lambda x: generate_transactions_table(x.iloc[0], start_date=start_date, nb_days=nb_days)
    ).reset_index(drop=True)
    if verbose:
        print(f"Time to generate transactions: {time.time()-start_time:.2f}s")

    # Sort transactions chronologically and add transaction IDs
    transactions_df = transactions_df.sort_values('TX_DATETIME')
    transactions_df.reset_index(inplace=True, drop=True)
    transactions_df.reset_index(inplace=True)
    transactions_df.rename(columns={'index': 'TRANSACTION_ID'}, inplace=True)

    return customer_profiles_table, terminal_profiles_table, transactions_df


# Legacy function for backward compatibility
def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    """
    Add fraudulent transactions using original scenarios from handbook.

    This function is kept for backward compatibility. For causal fraud scenarios,
    use simulator.causal_scenarios module instead.

    Scenarios:
    1. Amount > 220 → fraud
    2. Compromised terminals (28 days, 2 terminals/day)
    3. Compromised customers (14 days, 3 customers/day, 1/3 of txs with 5x amount)
    """
    # By default, all transactions are genuine
    transactions_df['TX_FRAUD'] = 0
    transactions_df['TX_FRAUD_SCENARIO'] = 0

    # Scenario 1: Amount > 220
    transactions_df.loc[transactions_df.TX_AMOUNT > 220, 'TX_FRAUD'] = 1
    transactions_df.loc[transactions_df.TX_AMOUNT > 220, 'TX_FRAUD_SCENARIO'] = 1
    nb_frauds_scenario_1 = transactions_df.TX_FRAUD.sum()
    print(f"Number of frauds from scenario 1: {nb_frauds_scenario_1}")

    # Scenario 2: Compromised terminals
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)

        compromised_transactions = transactions_df[
            (transactions_df.TX_TIME_DAYS >= day) &
            (transactions_df.TX_TIME_DAYS < day + 28) &
            (transactions_df.TERMINAL_ID.isin(compromised_terminals))
        ]

        transactions_df.loc[compromised_transactions.index, 'TX_FRAUD'] = 1
        transactions_df.loc[compromised_transactions.index, 'TX_FRAUD_SCENARIO'] = 2

    nb_frauds_scenario_2 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_1
    print(f"Number of frauds from scenario 2: {nb_frauds_scenario_2}")

    # Scenario 3: Compromised customers
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values

        compromised_transactions = transactions_df[
            (transactions_df.TX_TIME_DAYS >= day) &
            (transactions_df.TX_TIME_DAYS < day + 14) &
            (transactions_df.CUSTOMER_ID.isin(compromised_customers))
        ]

        nb_compromised_transactions = len(compromised_transactions)

        random.seed(day)
        index_frauds = random.sample(
            list(compromised_transactions.index.values),
            k=int(nb_compromised_transactions / 3)
        )

        transactions_df.loc[index_frauds, 'TX_AMOUNT'] = transactions_df.loc[index_frauds, 'TX_AMOUNT'] * 5
        transactions_df.loc[index_frauds, 'TX_FRAUD'] = 1
        transactions_df.loc[index_frauds, 'TX_FRAUD_SCENARIO'] = 3

    nb_frauds_scenario_3 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_2 - nb_frauds_scenario_1
    print(f"Number of frauds from scenario 3: {nb_frauds_scenario_3}")

    return transactions_df
