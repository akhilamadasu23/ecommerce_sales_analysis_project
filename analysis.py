
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# E-COMMERCE / RESTAURANT SALES DATA ANALYSIS PROJECT
# ============================================================
# Skills: Python, NumPy, Pandas, SQL, Matplotlib
# Run this file with: python analysis.py
# It creates synthetic data, performs SQL/Pandas analysis,
# and saves charts in the outputs/ folder.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# -----------------------------
# 1. CREATE CUSTOMERS
# -----------------------------
n_customers = 500

customers = pd.DataFrame({
    "customer_id": range(1, n_customers + 1),
    "customer_name": [f"Customer_{i:03d}" for i in range(1, n_customers + 1)],
    "city": np.random.choice(
        ["Hyderabad", "Warangal", "Bengaluru", "Chennai", "Pune", "Mumbai", "Delhi"],
        n_customers
    ),
    "customer_segment": np.random.choice(
        ["Regular", "Premium", "VIP"],
        n_customers,
        p=[0.65, 0.25, 0.10]
    )
})

customers.to_csv(DATA_DIR / "customers.csv", index=False)

# -----------------------------
# 2. CREATE PRODUCTS
# -----------------------------
products_list = [
    ("Laptop", "Electronics", 55000),
    ("Smartphone", "Electronics", 25000),
    ("Headphones", "Electronics", 2500),
    ("Keyboard", "Electronics", 1800),
    ("Mouse", "Electronics", 900),
    ("Backpack", "Accessories", 2200),
    ("Shoes", "Fashion", 3000),
    ("T-Shirt", "Fashion", 900),
    ("Jeans", "Fashion", 1800),
    ("Watch", "Accessories", 4500),
    ("Coffee Maker", "Home", 5000),
    ("Air Fryer", "Home", 6500),
    ("Mixer Grinder", "Home", 4200),
    ("Water Bottle", "Home", 700),
    ("Notebook", "Stationery", 250),
]

products = pd.DataFrame(products_list, columns=["product_name", "category", "base_price"])
products.insert(0, "product_id", range(1, len(products) + 1))

products.to_csv(DATA_DIR / "products.csv", index=False)

# -----------------------------
# 3. CREATE ORDERS
# -----------------------------
n_orders = 5000

order_dates = pd.to_datetime(
    np.random.choice(
        pd.date_range("2025-01-01", "2025-12-31", freq="D"),
        n_orders
    )
)

orders = pd.DataFrame({
    "order_id": range(1, n_orders + 1),
    "customer_id": np.random.randint(1, n_customers + 1, n_orders),
    "order_date": order_dates,
    "payment_method": np.random.choice(
        ["UPI", "Credit Card", "Debit Card", "Cash on Delivery"],
        n_orders,
        p=[0.45, 0.25, 0.20, 0.10]
    ),
    "order_status": np.random.choice(
        ["Delivered", "Cancelled", "Returned"],
        n_orders,
        p=[0.86, 0.09, 0.05]
    )
})

orders.to_csv(DATA_DIR / "orders.csv", index=False)

# -----------------------------
# 4. CREATE ORDER ITEMS
# -----------------------------
items = []

for order_id in orders["order_id"]:
    number_of_items = np.random.randint(1, 5)
    selected_products = np.random.choice(
        products["product_id"],
        size=number_of_items,
        replace=False
    )

    for product_id in selected_products:
        base_price = products.loc[
            products["product_id"] == product_id, "base_price"
        ].iloc[0]

        quantity = np.random.randint(1, 4)
        unit_price = round(base_price * np.random.uniform(0.90, 1.10), 2)

        items.append([
            order_id,
            product_id,
            quantity,
            unit_price
        ])

order_items = pd.DataFrame(
    items,
    columns=["order_id", "product_id", "quantity", "unit_price"]
)

order_items["line_total"] = (
    order_items["quantity"] * order_items["unit_price"]
).round(2)

order_items.to_csv(DATA_DIR / "order_items.csv", index=False)

print("Data generated successfully.")
print(f"Customers: {len(customers)}")
print(f"Orders: {len(orders)}")
print(f"Order items: {len(order_items)}")
print(f"Products: {len(products)}")

# ============================================================
# 5. LOAD DATA WITH PANDAS
# ============================================================

customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
order_items = pd.read_csv(DATA_DIR / "order_items.csv")

print("\n--- CUSTOMERS ---")
print(customers.head())

print("\n--- PRODUCTS ---")
print(products.head())

print("\n--- ORDERS ---")
print(orders.head())

print("\n--- ORDER ITEMS ---")
print(order_items.head())

# ============================================================
# 6. DATA QUALITY CHECK
# ============================================================

print("\n--- MISSING VALUES ---")
print(customers.isnull().sum())
print(products.isnull().sum())
print(orders.isnull().sum())
print(order_items.isnull().sum())

print("\n--- DUPLICATES ---")
print("Customer duplicates:", customers.duplicated().sum())
print("Product duplicates:", products.duplicated().sum())
print("Order duplicates:", orders.duplicated().sum())
print("Order item duplicates:", order_items.duplicated().sum())

# ============================================================
# 7. CREATE MASTER DATASET
# ============================================================

master = (
    order_items
    .merge(orders, on="order_id", how="left")
    .merge(products, on="product_id", how="left")
    .merge(customers, on="customer_id", how="left")
)

# Revenue should normally be calculated from delivered orders.
master["revenue"] = np.where(
    master["order_status"] == "Delivered",
    master["line_total"],
    0
)

master["month"] = master["order_date"].dt.to_period("M").astype(str)

print("\n--- MASTER DATASET ---")
print(master.head())

# ============================================================
# 8. BUSINESS QUESTIONS USING PANDAS
# ============================================================

# Q1. Total revenue
total_revenue = master["revenue"].sum()
print("\n1. Total Revenue:", round(total_revenue, 2))

# Q2. Total delivered orders
delivered_orders = orders.loc[
    orders["order_status"] == "Delivered", "order_id"
].nunique()

print("2. Delivered Orders:", delivered_orders)

# Q3. Average order value
delivered_order_revenue = (
    master[master["order_status"] == "Delivered"]
    .groupby("order_id")["revenue"]
    .sum()
)

average_order_value = delivered_order_revenue.mean()
print("3. Average Order Value:", round(average_order_value, 2))

# Q4. Top 10 products by revenue
top_products = (
    master.groupby("product_name")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n4. Top 10 Products by Revenue:")
print(top_products)

# Q5. Revenue by category
category_revenue = (
    master.groupby("category")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n5. Revenue by Category:")
print(category_revenue)

# Q6. Monthly revenue
monthly_revenue = (
    master.groupby("month")["revenue"]
    .sum()
)

print("\n6. Monthly Revenue:")
print(monthly_revenue)

# Q7. Top customers
top_customers = (
    master.groupby(["customer_id", "customer_name"])["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n7. Top 10 Customers:")
print(top_customers)

# Q8. Payment method usage
payment_usage = orders["payment_method"].value_counts()

print("\n8. Payment Method Usage:")
print(payment_usage)

# Q9. Order status distribution
order_status = orders["order_status"].value_counts()

print("\n9. Order Status:")
print(order_status)

# Q10. City revenue
city_revenue = (
    master.groupby("city")["revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n10. Revenue by City:")
print(city_revenue)

# ============================================================
# 9. SAVE ANALYSIS RESULTS
# ============================================================

top_products.to_csv(OUTPUT_DIR / "top_products.csv")
category_revenue.to_csv(OUTPUT_DIR / "category_revenue.csv")
monthly_revenue.to_csv(OUTPUT_DIR / "monthly_revenue.csv")
top_customers.to_csv(OUTPUT_DIR / "top_customers.csv")
city_revenue.to_csv(OUTPUT_DIR / "city_revenue.csv")

# ============================================================
# 10. VISUALIZATIONS
# ============================================================

# Monthly revenue
plt.figure(figsize=(12, 5))
plt.plot(monthly_revenue.index, monthly_revenue.values, marker="o")
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_revenue.png", dpi=150)
plt.close()

# Category revenue
plt.figure(figsize=(8, 5))
category_revenue.plot(kind="bar")
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_revenue.png", dpi=150)
plt.close()

# Top products
plt.figure(figsize=(10, 6))
top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Products by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_products.png", dpi=150)
plt.close()

# Payment methods
plt.figure(figsize=(8, 5))
payment_usage.plot(kind="bar")
plt.title("Orders by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "payment_methods.png", dpi=150)
plt.close()

# City revenue
plt.figure(figsize=(9, 5))
city_revenue.plot(kind="bar")
plt.title("Revenue by City")
plt.xlabel("City")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "city_revenue.png", dpi=150)
plt.close()

# ============================================================
# 11. SQL ANALYSIS USING SQLITE
# ============================================================

db_path = OUTPUT_DIR / "ecommerce.db"

connection = sqlite3.connect(db_path)

customers.to_sql("customers", connection, if_exists="replace", index=False)
products.to_sql("products", connection, if_exists="replace", index=False)
orders.to_sql("orders", connection, if_exists="replace", index=False)
order_items.to_sql("order_items", connection, if_exists="replace", index=False)

sql_queries = {
    "top_products": """
        SELECT
            p.product_name,
            ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.order_status = 'Delivered'
        GROUP BY p.product_name
        ORDER BY revenue DESC
        LIMIT 10;
    """,

    "revenue_by_category": """
        SELECT
            p.category,
            ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.order_status = 'Delivered'
        GROUP BY p.category
        ORDER BY revenue DESC;
    """,

    "revenue_by_city": """
        SELECT
            c.city,
            ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status = 'Delivered'
        GROUP BY c.city
        ORDER BY revenue DESC;
    """,

    "monthly_revenue": """
        SELECT
            strftime('%Y-%m', o.order_date) AS month,
            ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.order_status = 'Delivered'
        GROUP BY month
        ORDER BY month;
    """,

    "top_customers": """
        SELECT
            c.customer_id,
            c.customer_name,
            ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_status = 'Delivered'
        GROUP BY c.customer_id, c.customer_name
        ORDER BY revenue DESC
        LIMIT 10;
    """
}

print("\n================ SQL RESULTS ================")

for name, query in sql_queries.items():
    print(f"\n--- {name.upper()} ---")
    result = pd.read_sql_query(query, connection)
    print(result)

    result.to_csv(OUTPUT_DIR / f"sql_{name}.csv", index=False)

connection.close()

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n================ PROJECT SUMMARY ================")
print(f"Total Revenue: ₹{total_revenue:,.2f}")
print(f"Delivered Orders: {delivered_orders}")
print(f"Average Order Value: ₹{average_order_value:,.2f}")
print(f"Best Category: {category_revenue.index[0]}")
print(f"Best Product: {top_products.index[0]}")
print(f"Best City: {city_revenue.index[0]}")
print("\nCharts and SQL results are available in the outputs/ folder.")
print("Project completed successfully.")
