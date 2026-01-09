"""
Nike B2B Optimization UI - Streamlit Application
Interactive interface for testing and validating the constrained optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Page config
st.set_page_config(
    page_title="Nike B2B Optimizer",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #111111;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666666;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stDataFrame {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def get_default_products():
    """Return default Nike product catalog with previous season quantities."""
    return pd.DataFrame([
        # Footwear - prev_quantity represents last season's order
        {"id": "FW001", "name": "Air Max 90", "category": "Footwear", "price": 120.0, "probability": 0.92, "prev_quantity": 450, "min_order": 10, "max_order": 500},
        {"id": "FW002", "name": "Air Force 1 Low", "category": "Footwear", "price": 110.0, "probability": 0.88, "prev_quantity": 520, "min_order": 10, "max_order": 600},
        {"id": "FW003", "name": "Dunk Low Retro", "category": "Footwear", "price": 115.0, "probability": 0.85, "prev_quantity": 350, "min_order": 5, "max_order": 400},
        {"id": "FW004", "name": "Air Jordan 1 Mid", "category": "Footwear", "price": 135.0, "probability": 0.78, "prev_quantity": 200, "min_order": 5, "max_order": 300},
        {"id": "FW005", "name": "Pegasus 41", "category": "Footwear", "price": 140.0, "probability": 0.72, "prev_quantity": 180, "min_order": 10, "max_order": 350},
        {"id": "FW006", "name": "Vomero 18", "category": "Footwear", "price": 160.0, "probability": 0.65, "prev_quantity": 80, "min_order": 5, "max_order": 200},
        # Apparel
        {"id": "AP001", "name": "Tech Fleece Hoodie", "category": "Apparel", "price": 130.0, "probability": 0.89, "prev_quantity": 400, "min_order": 20, "max_order": 800},
        {"id": "AP002", "name": "Dri-FIT Training Tee", "category": "Apparel", "price": 35.0, "probability": 0.94, "prev_quantity": 1800, "min_order": 50, "max_order": 2000},
        {"id": "AP003", "name": "Sportswear Club Joggers", "category": "Apparel", "price": 65.0, "probability": 0.82, "prev_quantity": 600, "min_order": 30, "max_order": 1000},
        {"id": "AP004", "name": "Pro Compression Shorts", "category": "Apparel", "price": 40.0, "probability": 0.76, "prev_quantity": 1200, "min_order": 40, "max_order": 1500},
        {"id": "AP005", "name": "Windrunner Jacket", "category": "Apparel", "price": 110.0, "probability": 0.71, "prev_quantity": 250, "min_order": 15, "max_order": 400},
        {"id": "AP006", "name": "ACG Storm-FIT Jacket", "category": "Apparel", "price": 250.0, "probability": 0.58, "prev_quantity": 60, "min_order": 5, "max_order": 150},
        # Accessories
        {"id": "AC001", "name": "Elite Crew Socks 3-Pack", "category": "Accessories", "price": 22.0, "probability": 0.91, "prev_quantity": 2200, "min_order": 100, "max_order": 3000},
        {"id": "AC002", "name": "Heritage Backpack", "category": "Accessories", "price": 45.0, "probability": 0.79, "prev_quantity": 300, "min_order": 20, "max_order": 500},
        {"id": "AC003", "name": "Swoosh Headband", "category": "Accessories", "price": 12.0, "probability": 0.84, "prev_quantity": 1500, "min_order": 50, "max_order": 2000},
        {"id": "AC004", "name": "Training Gloves", "category": "Accessories", "price": 30.0, "probability": 0.67, "prev_quantity": 350, "min_order": 30, "max_order": 600},
        {"id": "AC005", "name": "Fuel Jug 64oz", "category": "Accessories", "price": 38.0, "probability": 0.62, "prev_quantity": 200, "min_order": 25, "max_order": 400},
        # Equipment
        {"id": "EQ001", "name": "Versa Training Ball", "category": "Equipment", "price": 25.0, "probability": 0.73, "prev_quantity": 350, "min_order": 20, "max_order": 500},
        {"id": "EQ002", "name": "Resistance Bands Set", "category": "Equipment", "price": 35.0, "probability": 0.69, "prev_quantity": 280, "min_order": 15, "max_order": 400},
        {"id": "EQ003", "name": "Yoga Mat", "category": "Equipment", "price": 50.0, "probability": 0.64, "prev_quantity": 150, "min_order": 10, "max_order": 300},
        {"id": "EQ004", "name": "Speed Rope", "category": "Equipment", "price": 20.0, "probability": 0.71, "prev_quantity": 400, "min_order": 30, "max_order": 600},
    ])


def run_optimization(
    products_df: pd.DataFrame, 
    total_budget: float, 
    category_budgets: dict,
    stability_weight: float = 0.0,
    diversity_weight: float = 0.0,
    min_skus_per_category: int = 0
):
    """
    Run the constrained optimization to MAXIMIZE EXPECTED REVENUE.
    
    Objective: Maximize Σ(price × probability × quantity) = Expected Revenue
    
    Where probability = P(partner purchases the product)
    
    Args:
        products_df: Product catalog with prices, probabilities, prev_quantity
        total_budget: Total budget constraint
        category_budgets: Per-category budget constraints
        stability_weight: Penalty for deviating from historical quantities (0 = ignore)
        diversity_weight: Penalty for concentration (encourages spreading across products)
        min_skus_per_category: Minimum number of SKUs to include per category
    """
    n = len(products_df)
    
    prices = products_df["price"].values
    probs = products_df["probability"].values
    mins = products_df["min_order"].values.astype(float)
    maxs = products_df["max_order"].values.astype(float)
    categories = products_df["category"].tolist()
    unique_categories = list(set(categories))
    
    # Previous season quantities (default to 0 if not present)
    if "prev_quantity" in products_df.columns:
        prev_qty = products_df["prev_quantity"].values.astype(float)
    else:
        prev_qty = np.zeros(n)
    
    # Decision variable: quantity for each product
    qty = cp.Variable(n)
    
    # ===========================================
    # OBJECTIVE: Maximize Expected Revenue
    # E[Revenue] = Σ(price_i × probability_i × quantity_i)
    # ===========================================
    expected_revenue = (prices * probs) @ qty
    
    # Penalty terms (subtracted from objective)
    penalties = 0
    
    # 1. Stability penalty: penalize deviation from historical quantities
    if stability_weight > 0 and np.any(prev_qty > 0):
        # Normalize by average historical quantity for intuitive weighting
        scale = np.mean(prev_qty[prev_qty > 0]) if np.any(prev_qty > 0) else 1.0
        # Penalty = λ × Σ((qty - prev_qty)²) / scale
        penalties += (stability_weight * 1000 / scale) * cp.sum_squares(qty - prev_qty)
    
    # 2. Diversity penalty: penalize concentration (Herfindahl-like)
    # Encourages spreading orders across more products
    if diversity_weight > 0:
        # Penalty on squared quantities (penalizes putting too much in one product)
        total_possible = np.sum(maxs)
        penalties += (diversity_weight * 100 / total_possible) * cp.sum_squares(qty)
    
    objective = cp.Maximize(expected_revenue - penalties)
    
    # ===========================================
    # CONSTRAINTS
    # ===========================================
    constraints = [
        qty >= mins,
        qty <= maxs,
    ]
    
    # Budget constraint: total spend <= total budget
    constraints.append(prices @ qty <= total_budget)
    
    # Category budget constraints
    for cat, budget in category_budgets.items():
        mask = np.array([1.0 if c == cat else 0.0 for c in categories])
        constraints.append((prices * mask) @ qty <= budget)
    
    # Minimum SKUs per category constraint
    # Uses a soft approach: if min_order > 0, product is "selected"
    if min_skus_per_category > 0:
        for cat in unique_categories:
            cat_indices = [i for i, c in enumerate(categories) if c == cat]
            if len(cat_indices) >= min_skus_per_category:
                # Ensure at least min_skus products have qty >= min_order
                # This is approximated by ensuring sum of indicators >= min_skus
                # For LP relaxation, we use: Σ(qty_i / max_i) >= min_skus (soft version)
                cat_maxs = np.array([maxs[i] for i in cat_indices])
                cat_qty = cp.hstack([qty[i] for i in cat_indices])
                constraints.append(cp.sum(cat_qty / cat_maxs) >= min_skus_per_category * 0.1)
    
    # ===========================================
    # SOLVE
    # ===========================================
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None, f"Optimization failed: {problem.status}. Try adjusting constraints."
    
    # Build results
    final_qty = np.floor(qty.value).astype(int)
    final_qty = np.maximum(final_qty, mins.astype(int))
    
    results_df = products_df.copy()
    results_df["quantity"] = final_qty
    results_df["spend"] = results_df["quantity"] * results_df["price"]
    results_df["expected_revenue"] = results_df["quantity"] * results_df["price"] * results_df["probability"]
    
    # Calculate change from previous season
    if "prev_quantity" in products_df.columns:
        results_df["change"] = results_df["quantity"] - results_df["prev_quantity"]
        results_df["change_pct"] = np.where(
            results_df["prev_quantity"] > 0,
            ((results_df["quantity"] - results_df["prev_quantity"]) / results_df["prev_quantity"] * 100).round(1),
            np.nan
        )
    
    return results_df, None


def main():
    # Header
    st.markdown('<p class="main-header">👟 Nike B2B Constrained Optimizer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimize product quantities based on recommendations and budget constraints</p>', unsafe_allow_html=True)
    
    # Mathematical formulation expander
    with st.expander("📐 View Optimization Formulation"):
        st.markdown("### Objective Function — Maximize Expected Revenue:")
        st.latex(r"\text{E}[\text{Revenue}] = \sum_{i=1}^{n} \text{price}_i \times \text{probability}_i \times \text{quantity}_i")
        st.caption("Where probability = P(partner purchases the product)")
        
        st.markdown("### Optional Penalties (subtracted from objective):")
        
        st.markdown("**1. Historical Stability Penalty:**")
        st.latex(r"\lambda_1 \sum_{i=1}^{n} (\text{quantity}_i - \text{prev\_quantity}_i)^2")
        st.caption("Penalizes large deviations from previous season orders")
        
        st.markdown("**2. Diversity Penalty:**")
        st.latex(r"\lambda_2 \sum_{i=1}^{n} \text{quantity}_i^2")
        st.caption("Penalizes concentration; encourages spreading across products")
        
        st.markdown("### Subject to Constraints:")
        st.latex(r"\sum_{i=1}^{n} \text{price}_i \times \text{quantity}_i \leq \text{Total Budget}")
        st.caption("Total spend must not exceed budget")
        
        st.latex(r"\sum_{i \in \text{category}_c} \text{price}_i \times \text{quantity}_i \leq \text{Budget}_c \quad \forall \text{ categories } c")
        st.caption("Spend per category must not exceed category budget allocation")
        
        st.latex(r"\text{min\_order}_i \leq \text{quantity}_i \leq \text{max\_order}_i \quad \forall \text{ products } i")
        st.caption("Each product quantity must be within its min/max bounds")
        
        st.markdown("""
        ---
        **Key Variables:**
        - `probability` = P(partner purchases this product) — from your recommendation model
        - `price` = wholesale price per unit
        - `λ₁` = historical stability weight (0 = ignore, higher = prefer consistency)
        - `λ₂` = diversity weight (0 = concentrate on best, higher = spread orders)
        """)
    
    st.divider()
    
    # Initialize session state
    if "products_df" not in st.session_state:
        st.session_state.products_df = get_default_products()
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    
    # Sidebar - Budget Configuration
    with st.sidebar:
        st.header("💰 Budget Configuration")
        
        total_budget = st.number_input(
            "Total Budget ($)",
            min_value=10000,
            max_value=10000000,
            value=500000,
            step=10000,
            format="%d"
        )
        
        st.subheader("Category Budgets (%)")
        st.caption("Percentage of total budget per category")
        categories = st.session_state.products_df["category"].unique().tolist()
        
        # Default percentages
        default_pcts = {
            "Footwear": 40,
            "Apparel": 36,
            "Accessories": 16,
            "Equipment": 8
        }
        
        category_pcts = {}
        for cat in categories:
            default_val = default_pcts.get(cat, int(100 / len(categories)))
            category_pcts[cat] = st.slider(
                f"{cat}",
                min_value=0,
                max_value=100,
                value=default_val,
                step=1,
                format="%d%%",
                key=f"budget_pct_{cat}"
            )
        
        # Validation and compute actual budgets
        pct_sum = sum(category_pcts.values())
        if pct_sum != 100:
            st.warning(f"⚠️ Percentages sum to {pct_sum}% (should be 100%)")
        
        # Convert percentages to dollar amounts
        category_budgets = {cat: total_budget * (pct / 100) for cat, pct in category_pcts.items()}
        
        # Show computed amounts
        with st.expander("💵 View Dollar Amounts"):
            for cat, budget in category_budgets.items():
                st.text(f"{cat}: ${budget:,.0f}")
        
        st.divider()
        
        # Optimization settings
        st.header("⚙️ Optimization Settings")
        
        stability_weight = st.slider(
            "Historical Stability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="0 = Ignore history, 1 = Strongly prefer previous season quantities"
        )
        
        diversity_weight = st.slider(
            "Product Diversity",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="0 = Concentrate on best products, 1 = Spread across more products"
        )
        
        min_skus = st.number_input(
            "Min SKUs per Category",
            min_value=0,
            max_value=10,
            value=0,
            help="Minimum number of products to order from each category"
        )
        
        st.divider()
        
        # Run optimization button
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Maximizing Expected Revenue..."):
                results, error = run_optimization(
                    st.session_state.products_df,
                    total_budget,
                    category_budgets,
                    stability_weight=stability_weight,
                    diversity_weight=diversity_weight,
                    min_skus_per_category=min_skus
                )
                
                if error:
                    st.error(error)
                else:
                    st.session_state.results_df = results
                    st.success("✅ Optimization complete!")
    
    # Main content area - Tabs
    tab1, tab2, tab3 = st.tabs(["📦 Product Catalog", "📊 Results", "📈 Analytics"])
    
    # Tab 1: Product Catalog
    with tab1:
        st.subheader("Product Catalog")
        st.caption("Edit product data directly in the table. Changes will be used in the next optimization run.")
        
        edited_df = st.data_editor(
            st.session_state.products_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "name": st.column_config.TextColumn("Product Name", width="medium"),
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=["Footwear", "Apparel", "Accessories", "Equipment"],
                    width="small"
                ),
                "price": st.column_config.NumberColumn("Price ($)", format="$%.2f", min_value=0),
                "probability": st.column_config.NumberColumn(
                    "P(Buy)",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                    help="Probability that partner will purchase this product (0-1)"
                ),
                "prev_quantity": st.column_config.NumberColumn(
                    "Prev Season Qty",
                    format="%d",
                    min_value=0,
                    help="Quantity ordered in previous season"
                ),
                "min_order": st.column_config.NumberColumn("Min Order", min_value=0),
                "max_order": st.column_config.NumberColumn("Max Order", min_value=1),
            },
            hide_index=True,
        )
        st.session_state.products_df = edited_df
        
        # Upload CSV option
        st.divider()
        uploaded_file = st.file_uploader("Or upload a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                new_df = pd.read_csv(uploaded_file)
                required_cols = ["id", "name", "category", "price", "probability", "prev_quantity", "min_order", "max_order"]
                if all(col in new_df.columns for col in required_cols):
                    st.session_state.products_df = new_df[required_cols]
                    st.success(f"Loaded {len(new_df)} products from CSV")
                    st.rerun()
                else:
                    st.error(f"CSV must contain columns: {required_cols}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    # Tab 2: Results
    with tab2:
        if st.session_state.results_df is None:
            st.info("👈 Configure budgets and click 'Run Optimization' to see results")
        else:
            results = st.session_state.results_df
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            total_spend = results["spend"].sum()
            total_units = results["quantity"].sum()
            expected_rev = results["expected_revenue"].sum() if "expected_revenue" in results.columns else (results["price"] * results["probability"] * results["quantity"]).sum()
            roi_ratio = expected_rev / total_spend if total_spend > 0 else 0
            
            with col1:
                st.metric("Total Spend", f"${total_spend:,.0f}", f"{(total_spend/total_budget)*100:.1f}% of budget")
                st.caption("ℹ️ Total dollar amount allocated")
            with col2:
                st.metric("Expected Revenue", f"${expected_rev:,.0f}")
                st.caption("ℹ️ Σ(price × probability × qty)")
            with col3:
                st.metric("Expected ROI", f"{roi_ratio:.1%}")
                st.caption("ℹ️ Expected Revenue / Spend")
            with col4:
                st.metric("Products Ordered", f"{(results['quantity'] > 0).sum()}")
                st.caption("ℹ️ Number of unique SKUs with orders")
            
            st.divider()
            
            # Results table
            st.subheader("Optimized Quantities")
            
            # Include key columns
            display_cols = ["id", "name", "category", "price", "probability", "quantity", "spend", "expected_revenue"]
            if "change_pct" in results.columns:
                display_cols.append("change_pct")
            
            available_cols = [c for c in display_cols if c in results.columns]
            display_df = results[available_cols].copy()
            display_df = display_df.sort_values("expected_revenue", ascending=False)
            
            col_config = {
                "price": st.column_config.NumberColumn("Price", format="$%.2f", help="Wholesale price per unit"),
                "probability": st.column_config.ProgressColumn("P(Buy)", min_value=0, max_value=1, help="Probability partner purchases this product (from your recommendation model)"),
                "quantity": st.column_config.NumberColumn("Qty", format="%d", help="Optimized quantity to order"),
                "spend": st.column_config.NumberColumn("Spend", format="$%.0f", help="Your cost = Price × Quantity"),
                "expected_revenue": st.column_config.NumberColumn("E[Revenue]", format="$%.0f", help="Expected revenue = Price × P(Buy) × Qty. This is what you expect to earn if partner buys at this probability."),
            }
            if "change_pct" in results.columns:
                col_config["change_pct"] = st.column_config.NumberColumn("Δ%", format="%.1f%%", help="% change from previous season quantity. Negative = ordering less than before.")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config=col_config,
                hide_index=True
            )
            
            # Column explanations
            with st.expander("ℹ️ Column Definitions"):
                st.markdown("""
                | Column | Formula | Meaning |
                |--------|---------|---------|
                | **Qty** | — | Optimized quantity to order for this product |
                | **Spend** | `Price × Qty` | Your cost (what you pay) |
                | **E[Revenue]** | `Price × P(Buy) × Qty` | Expected revenue (what you expect to earn) |
                | **Δ%** | `(Qty - Prev) / Prev × 100` | Change from previous season (negative = ordering less) |
                
                **Example:** If you order 100 units at $50 with 80% purchase probability:
                - **Spend** = $50 × 100 = **$5,000** (your investment)
                - **E[Revenue]** = $50 × 0.80 × 100 = **$4,000** (expected return)
                - If E[Revenue] < Spend, you expect a loss on that product
                """)
            
            # Download results
            csv = results.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "optimization_results.csv",
                "text/csv",
                use_container_width=True
            )
    
    # Tab 3: Analytics
    with tab3:
        if st.session_state.results_df is None:
            st.info("👈 Run optimization to see analytics")
        else:
            results = st.session_state.results_df
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Spend by category
                st.subheader("Budget Utilization by Category")
                cat_summary = results.groupby("category").agg({
                    "spend": "sum",
                    "quantity": "sum"
                }).reset_index()
                cat_summary["budget"] = cat_summary["category"].map(category_budgets)
                cat_summary["utilization"] = (cat_summary["spend"] / cat_summary["budget"] * 100).round(1)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Allocated",
                    x=cat_summary["category"],
                    y=cat_summary["spend"],
                    marker_color="#667eea"
                ))
                fig.add_trace(go.Bar(
                    name="Remaining",
                    x=cat_summary["category"],
                    y=cat_summary["budget"] - cat_summary["spend"],
                    marker_color="#e0e0e0"
                ))
                fig.update_layout(
                    barmode="stack",
                    yaxis_title="Amount ($)",
                    showlegend=True,
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Spend distribution pie
                st.subheader("Spend Distribution")
                fig = px.pie(
                    cat_summary,
                    values="spend",
                    names="category",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hole=0.4
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            # Probability vs Quantity scatter
            st.subheader("Recommendation Score vs Quantity Allocated")
            fig = px.scatter(
                results[results["quantity"] > 0],
                x="probability",
                y="quantity",
                size="spend",
                color="category",
                hover_name="name",
                hover_data=["price", "spend"],
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                xaxis_title="Recommendation Score",
                yaxis_title="Quantity Allocated",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top products by spend
            st.subheader("Top 10 Products by Spend")
            top_products = results.nlargest(10, "spend")
            fig = px.bar(
                top_products,
                x="spend",
                y="name",
                orientation="h",
                color="category",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                xaxis_title="Total Spend ($)",
                yaxis_title="",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Previous vs Optimized Quantities comparison
            if "prev_quantity" in results.columns:
                st.subheader("📅 Previous Season vs Optimized Quantities")
                st.caption("Compare how the optimization deviates from historical purchasing patterns")
                
                # Filter to products with both prev and new quantities
                compare_df = results[results["prev_quantity"] > 0].nlargest(15, "spend").copy()
                
                if len(compare_df) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Previous Season",
                        x=compare_df["name"],
                        y=compare_df["prev_quantity"],
                        marker_color="#94a3b8"
                    ))
                    fig.add_trace(go.Bar(
                        name="Optimized",
                        x=compare_df["name"],
                        y=compare_df["quantity"],
                        marker_color="#667eea"
                    ))
                    fig.update_layout(
                        barmode="group",
                        xaxis_tickangle=-45,
                        yaxis_title="Quantity",
                        height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Change distribution
                    if "change_pct" in results.columns:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Change Distribution")
                            change_data = results[results["change_pct"].notna()]["change_pct"]
                            fig = px.histogram(
                                change_data,
                                nbins=20,
                                labels={"value": "Change %", "count": "Products"},
                                color_discrete_sequence=["#667eea"]
                            )
                            fig.add_vline(x=0, line_dash="dash", line_color="red")
                            fig.update_layout(
                                showlegend=False,
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Change Summary")
                            valid_changes = results[results["change_pct"].notna()]
                            increased = (valid_changes["change_pct"] > 0).sum()
                            decreased = (valid_changes["change_pct"] < 0).sum()
                            unchanged = (valid_changes["change_pct"] == 0).sum()
                            avg_change = valid_changes["change_pct"].mean()
                            
                            st.metric("Avg Change", f"{avg_change:+.1f}%")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("📈 Increased", increased)
                            with col_b:
                                st.metric("📉 Decreased", decreased)
                            with col_c:
                                st.metric("➡️ Unchanged", unchanged)


if __name__ == "__main__":
    main()

