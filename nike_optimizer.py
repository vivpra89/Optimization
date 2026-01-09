"""
Nike B2B Constrained Optimization for Product Purchasing

Problem: Given a list of recommended products with purchase probabilities,
determine optimal quantities to purchase while respecting budget constraints.

Objective: Maximize expected value (probability-weighted quantity selection)
Constraints:
    - Total budget constraint
    - Per-category budget constraints
    - Non-negative integer quantities
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Product:
    id: str
    name: str
    category: str
    price: float
    probability: float  # Recommendation strength (0-1)
    min_order: int = 0
    max_order: int = 1000


def generate_mock_data() -> tuple[list[Product], dict]:
    """Generate realistic Nike B2B product data."""
    
    products = [
        # Footwear
        Product("FW001", "Air Max 90", "Footwear", 120.00, 0.92, min_order=10, max_order=500),
        Product("FW002", "Air Force 1 Low", "Footwear", 110.00, 0.88, min_order=10, max_order=600),
        Product("FW003", "Dunk Low Retro", "Footwear", 115.00, 0.85, min_order=5, max_order=400),
        Product("FW004", "Air Jordan 1 Mid", "Footwear", 135.00, 0.78, min_order=5, max_order=300),
        Product("FW005", "Pegasus 41", "Footwear", 140.00, 0.72, min_order=10, max_order=350),
        Product("FW006", "Vomero 18", "Footwear", 160.00, 0.65, min_order=5, max_order=200),
        
        # Apparel
        Product("AP001", "Tech Fleece Hoodie", "Apparel", 130.00, 0.89, min_order=20, max_order=800),
        Product("AP002", "Dri-FIT Training Tee", "Apparel", 35.00, 0.94, min_order=50, max_order=2000),
        Product("AP003", "Sportswear Club Joggers", "Apparel", 65.00, 0.82, min_order=30, max_order=1000),
        Product("AP004", "Pro Compression Shorts", "Apparel", 40.00, 0.76, min_order=40, max_order=1500),
        Product("AP005", "Windrunner Jacket", "Apparel", 110.00, 0.71, min_order=15, max_order=400),
        Product("AP006", "ACG Storm-FIT Jacket", "Apparel", 250.00, 0.58, min_order=5, max_order=150),
        
        # Accessories
        Product("AC001", "Elite Crew Socks 3-Pack", "Accessories", 22.00, 0.91, min_order=100, max_order=3000),
        Product("AC002", "Heritage Backpack", "Accessories", 45.00, 0.79, min_order=20, max_order=500),
        Product("AC003", "Swoosh Headband", "Accessories", 12.00, 0.84, min_order=50, max_order=2000),
        Product("AC004", "Training Gloves", "Accessories", 30.00, 0.67, min_order=30, max_order=600),
        Product("AC005", "Fuel Jug 64oz", "Accessories", 38.00, 0.62, min_order=25, max_order=400),
        
        # Equipment
        Product("EQ001", "Versa Training Ball", "Equipment", 25.00, 0.73, min_order=20, max_order=500),
        Product("EQ002", "Resistance Bands Set", "Equipment", 35.00, 0.69, min_order=15, max_order=400),
        Product("EQ003", "Yoga Mat", "Equipment", 50.00, 0.64, min_order=10, max_order=300),
        Product("EQ004", "Speed Rope", "Equipment", 20.00, 0.71, min_order=30, max_order=600),
    ]
    
    # Budget constraints
    budgets = {
        "total": 500_000,  # $500K total budget
        "category": {
            "Footwear": 200_000,
            "Apparel": 180_000,
            "Accessories": 80_000,
            "Equipment": 40_000,
        }
    }
    
    return products, budgets


def solve_optimization(products: list[Product], budgets: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Solve the constrained optimization problem.
    
    Objective: Maximize sum of (probability * quantity) - weighted by recommendation strength
    Subject to:
        - Total spend <= total budget
        - Category spend <= category budget (for each category)
        - min_order <= quantity <= max_order for each product
        - Quantities are integers
    """
    n = len(products)
    
    # Extract data as arrays
    prices = np.array([p.price for p in products])
    probabilities = np.array([p.probability for p in products])
    min_orders = np.array([p.min_order for p in products])
    max_orders = np.array([p.max_order for p in products])
    categories = [p.category for p in products]
    unique_categories = list(budgets["category"].keys())
    
    # Decision variable: quantity for each product (continuous, will round later)
    quantities = cp.Variable(n)
    
    # Objective: Maximize probability-weighted quantities
    # This prioritizes high-recommendation products
    objective = cp.Maximize(probabilities @ quantities)
    
    # Constraints
    constraints = []
    
    # 1. Total budget constraint
    constraints.append(prices @ quantities <= budgets["total"])
    
    # 2. Category budget constraints
    for cat in unique_categories:
        cat_mask = np.array([1.0 if c == cat else 0.0 for c in categories])
        cat_spend = (prices * cat_mask) @ quantities
        constraints.append(cat_spend <= budgets["category"][cat])
    
    # 3. Min/max order constraints
    constraints.append(quantities >= min_orders)
    constraints.append(quantities <= max_orders)
    
    # Solve using CLARABEL (bundled with cvxpy)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL, verbose=False)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
    # Round to integers (floor to stay within budget)
    final_quantities = np.floor(quantities.value).astype(int)
    
    # Ensure we respect minimums
    final_quantities = np.maximum(final_quantities, min_orders)
    
    # Build results dataframe
    results = []
    for i, product in enumerate(products):
        qty = final_quantities[i]
        spend = qty * product.price
        results.append({
            "Product ID": product.id,
            "Product Name": product.name,
            "Category": product.category,
            "Unit Price": product.price,
            "Rec. Probability": product.probability,
            "Quantity": qty,
            "Total Spend": spend,
        })
    
    df = pd.DataFrame(results)
    
    # Calculate actual objective value with rounded quantities
    actual_objective = float(probabilities @ final_quantities)
    
    if verbose:
        print_results(df, budgets, actual_objective)
    
    return df


def print_results(df: pd.DataFrame, budgets: dict, objective_value: float):
    """Print formatted optimization results."""
    
    print("=" * 80)
    print("NIKE B2B CONSTRAINED OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"\n{'PRODUCT ALLOCATION':^80}")
    print("-" * 80)
    print(f"{'Product':<30} {'Category':<12} {'Price':>10} {'Prob':>6} {'Qty':>8} {'Spend':>12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['Product Name']:<30} {row['Category']:<12} "
              f"${row['Unit Price']:>8.2f} {row['Rec. Probability']:>6.2f} "
              f"{row['Quantity']:>8,} ${row['Total Spend']:>10,.2f}")
    
    print("-" * 80)
    
    # Summary by category
    print(f"\n{'CATEGORY SUMMARY':^80}")
    print("-" * 80)
    print(f"{'Category':<15} {'Budget':>15} {'Allocated':>15} {'Remaining':>15} {'Utilization':>12}")
    print("-" * 80)
    
    for cat, budget in budgets["category"].items():
        allocated = df[df["Category"] == cat]["Total Spend"].sum()
        remaining = budget - allocated
        utilization = (allocated / budget) * 100
        print(f"{cat:<15} ${budget:>13,.2f} ${allocated:>13,.2f} ${remaining:>13,.2f} {utilization:>10.1f}%")
    
    print("-" * 80)
    
    total_allocated = df["Total Spend"].sum()
    total_remaining = budgets["total"] - total_allocated
    total_utilization = (total_allocated / budgets["total"]) * 100
    
    print(f"{'TOTAL':<15} ${budgets['total']:>13,.2f} ${total_allocated:>13,.2f} "
          f"${total_remaining:>13,.2f} {total_utilization:>10.1f}%")
    
    print("\n" + "=" * 80)
    print(f"Objective Value (Probability-Weighted Quantity): {objective_value:,.2f}")
    print(f"Total Products Ordered: {df['Quantity'].sum():,}")
    print(f"Total Spend: ${total_allocated:,.2f}")
    print("=" * 80)


def main():
    """Main entry point."""
    print("Generating mock Nike B2B data...")
    products, budgets = generate_mock_data()
    
    print(f"Products: {len(products)}")
    print(f"Total Budget: ${budgets['total']:,}")
    print(f"Category Budgets: {budgets['category']}")
    print("\nSolving optimization problem...\n")
    
    results_df = solve_optimization(products, budgets)
    
    # Save results to CSV
    results_df.to_csv("optimization_results.csv", index=False)
    print("\nResults saved to optimization_results.csv")


if __name__ == "__main__":
    main()

