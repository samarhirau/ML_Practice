import pandas as pd

data = {
    "City": ["Mumbai", "Mumbai", "Delhi", "Delhi", "Delhi", "Bangalore", "Bangalore"],
    "Product": ["Laptop", "Phone", "Laptop", "Phone", "Tablet", "Laptop", "Phone"],
    "Sales": [1000, 500, 1200, 600, 300, 800, 400]
}

df = pd.DataFrame(data)

print("--- Initial Dataset ---")
print(df)
print("\n" + "="*40 + "\n")

# 2. Use GroupBy to find total sales per city
total_sales_per_city = df.groupby("City")["Sales"].sum()
print("--- Total Sales per City ---")
print(total_sales_per_city)
print("\n" + "="*40 + "\n")

# 3. Create a multi-level grouping by both City & Product to calculate average sales
avg_sales_city_product = df.groupby(["City", "Product"])["Sales"].mean()
print("--- Average Sales by City & Product ---")
print(avg_sales_city_product)
print("\n" + "="*40 + "\n")

# 4. Apply a custom function that normalizes sales by the maximum value in each city
def normalize_by_city_max(city_sales):
    return city_sales / city_sales.max()

# Using transform to keep original DataFrame index and structure
df["Normalized_Sales"] = df.groupby("City")["Sales"].transform(normalize_by_city_max)

print("--- Final Dataset with Normalized Sales ---")
print(df)