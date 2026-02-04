"""
Quick test to verify core simulator extraction works correctly
"""

from core import generate_customer_profiles_table, generate_terminal_profiles_table, generate_dataset

# Test 1: Generate small customer and terminal profiles
print("Test 1: Generating small profiles...")
customers = generate_customer_profiles_table(n_customers=5, random_state=0)
terminals = generate_terminal_profiles_table(n_terminals=5, random_state=0)

print(f"[OK] Generated {len(customers)} customers")
print(f"[OK] Generated {len(terminals)} terminals")
print(f"[OK] Customer columns: {list(customers.columns)}")
print(f"[OK] Terminal columns: {list(terminals.columns)}")

# Test 2: Generate tiny dataset
print("\nTest 2: Generating tiny dataset (10 customers, 10 terminals, 5 days)...")
customers, terminals, transactions = generate_dataset(
    n_customers=10,
    n_terminals=10,
    nb_days=5,
    start_date="2024-01-01",
    r=50,
    verbose=True
)

print(f"\n[OK] Generated {len(transactions)} transactions")
print(f"[OK] Transaction columns: {list(transactions.columns)}")
print(f"[OK] Date range: {transactions.TX_DATETIME.min()} to {transactions.TX_DATETIME.max()}")
print(f"[OK] Amount range: ${transactions.TX_AMOUNT.min():.2f} to ${transactions.TX_AMOUNT.max():.2f}")

print("\n[SUCCESS] All tests passed! Core simulator extraction successful.")
