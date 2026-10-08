"""
Pandas for ML - Practice Exercises
Learn by doing. Modify, experiment, break things!
"""

import pandas as pd
import numpy as np

# ============================================================================
# EXERCISE 1: Create and Explore DataFrames
# ============================================================================
print("EXERCISE 1: Create and Explore Data")
print("-" * 50)

# Create a simple dataset: Customer purchases
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'age': [25, 45, 35, 50, 28],
    'purchase_amount': [100, 500, 250, 1000, 150],
    'region': ['North', 'South', 'North', 'East', 'South']
})

# TODO: Your task
# 1. Print the first 3 rows
# 2. Print the shape
# 3. Print data types
# 4. Print summary statistics

print(f"DataFrame:\n{customers}\n")
print(f"Shape: {customers.shape}")
print(f"Data types:\n{customers.dtypes}")
print(f"Summary:\n{customers.describe()}\n")

# ============================================================================
# EXERCISE 2: Select and Filter Data
# ============================================================================
print("\nEXERCISE 2: Select and Filter Data")
print("-" * 50)

# TODO: Your task
# 1. Select only the 'age' column
# 2. Select customers with age > 40
# 3. Select customers from 'North' region
# 4. Select customers with purchase > 300

print(f"Ages only:\n{customers['age']}\n")

over_40 = customers[customers['age'] > 40]
print(f"Customers over 40:\n{over_40}\n")

north = customers[customers['region'] == 'North']
print(f"North region customers:\n{north}\n")

high_spend = customers[customers['purchase_amount'] > 300]
print(f"High spenders (>300):\n{high_spend}\n")

# ============================================================================
# EXERCISE 3: Add New Columns & Transform
# ============================================================================
print("\nEXERCISE 3: Add and Transform Columns")
print("-" * 50)

# TODO: Your task
# 1. Add a column: "age_group" (Young: <30, Middle: 30-45, Senior: >45)
# 2. Add a column: "high_value" (1 if purchase > 300, else 0)
# 3. Create a new column: "purchase_per_age" = purchase_amount / age

customers_transformed = customers.copy()

# Age group (using conditions)
customers_transformed['age_group'] = pd.cut(
    customers_transformed['age'],
    bins=[0, 30, 45, 100],
    labels=['Young', 'Middle', 'Senior']
)

# High value flag
customers_transformed['high_value'] = (customers_transformed['purchase_amount'] > 300).astype(int)

# Purchase per age
customers_transformed['purchase_per_age'] = customers_transformed['purchase_amount'] / customers_transformed['age']

print(f"Transformed data:\n{customers_transformed}\n")

# ============================================================================
# EXERCISE 4: Handling Missing Data
# ============================================================================
print("\nEXERCISE 4: Handle Missing Data")
print("-" * 50)

# Create data with missing values
data_missing = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'score': [85, np.nan, 92, 88, np.nan],
    'status': ['active', 'inactive', np.nan, 'active', 'inactive']
})

print(f"Data with missing:\n{data_missing}\n")

# TODO: Your task
# 1. Check for missing values
# 2. Fill score with mean
# 3. Fill status with 'unknown'
# 4. Show cleaned data

missing_count = data_missing.isnull().sum()
print(f"Missing count:\n{missing_count}\n")

data_clean = data_missing.copy()
data_clean['score'] = data_clean['score'].fillna(data_clean['score'].mean())
data_clean['status'] = data_clean['status'].fillna('unknown')

print(f"Cleaned data:\n{data_clean}\n")

# ============================================================================
# EXERCISE 5: Group By & Aggregation
# ============================================================================
print("\nEXERCISE 5: GroupBy & Aggregation")
print("-" * 50)

# TODO: Your task
# 1. Average purchase by region
# 2. Count of customers per region
# 3. Min, max, mean purchase by region
# 4. Total purchase per region

avg_by_region = customers.groupby('region')['purchase_amount'].mean()
print(f"Average purchase by region:\n{avg_by_region}\n")

count_by_region = customers.groupby('region').size()
print(f"Customer count by region:\n{count_by_region}\n")

stats_by_region = customers.groupby('region')['purchase_amount'].agg(['min', 'max', 'mean'])
print(f"Purchase stats by region:\n{stats_by_region}\n")

total_by_region = customers.groupby('region')['purchase_amount'].sum()
print(f"Total purchase by region:\n{total_by_region}\n")

# ============================================================================
# EXERCISE 6: Remove Duplicates
# ============================================================================
print("\nEXERCISE 6: Remove Duplicates")
print("-" * 50)

data_dup = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'score': [90, 85, 90, 88, 85]
})

print(f"Data with duplicates:\n{data_dup}\n")

# Check duplicates
print(f"Duplicate rows:\n{data_dup[data_dup.duplicated()]}\n")

# Remove duplicates
data_unique = data_dup.drop_duplicates()
print(f"After removing duplicates:\n{data_unique}\n")

# ============================================================================
# EXERCISE 7: Sorting & Ranking
# ============================================================================
print("\nEXERCISE 7: Sort and Rank")
print("-" * 50)

# TODO: Your task
# 1. Sort by purchase_amount (descending)
# 2. Sort by age (ascending)
# 3. Get top 3 customers by spending

sorted_by_spend = customers.sort_values('purchase_amount', ascending=False)
print(f"Sorted by spending (high to low):\n{sorted_by_spend[['customer_id', 'purchase_amount']]}\n")

sorted_by_age = customers.sort_values('age')
print(f"Sorted by age (low to high):\n{sorted_by_age[['customer_id', 'age']]}\n")

top_3 = customers.nlargest(3, 'purchase_amount')
print(f"Top 3 spenders:\n{top_3[['customer_id', 'purchase_amount']]}\n")

# ============================================================================
# EXERCISE 8: Data Type Conversion
# ============================================================================
print("\nEXERCISE 8: Convert Data Types")
print("-" * 50)

data_types = pd.DataFrame({
    'id': ['1', '2', '3', '4'],
    'score': ['85.5', '90.2', '78.9', '92.1'],
    'passed': ['yes', 'yes', 'no', 'yes']
})

print(f"Original types:\n{data_types.dtypes}\n")

# Convert
data_types['id'] = data_types['id'].astype(int)
data_types['score'] = data_types['score'].astype(float)
data_types['passed'] = data_types['passed'].map({'yes': 1, 'no': 0})

print(f"Converted types:\n{data_types.dtypes}\n")
print(f"Converted data:\n{data_types}\n")

# ============================================================================
# EXERCISE 9: Real ML Preparation Workflow
# ============================================================================
print("\nEXERCISE 9: Prepare Data for ML")
print("-" * 50)

# Simulated student data
students = pd.DataFrame({
    'math': [85, 92, 75, 88, 65, 95, 72, 85],
    'english': [90, 88, 70, 85, 60, 92, 68, 87],
    'science': [88, 95, 65, 90, 55, 98, 70, 89],
    'grade': ['A', 'A', 'C', 'B', 'F', 'A', 'C', 'B']
})

print(f"Raw data:\n{students}\n")

# Step 1: Check data quality
print(f"Missing values: {students.isnull().sum().sum()}")
print(f"Data types:\n{students.dtypes}\n")

# Step 2: Separate features (X) and target (y)
X = students[['math', 'english', 'science']]
y = students['grade']

print(f"Features (X):\n{X.head()}\n")
print(f"Target (y):\n{y.head()}\n")

# Step 3: Normalize features (0-1)
X_normalized = (X - X.min()) / (X.max() - X.min())
print(f"Normalized features (first row):\n{X_normalized.iloc[0]}\n")

# Step 4: Check correlations
print(f"Feature correlations:\n{X.corr()}\n")

# ============================================================================
# EXERCISE 10: Challenge - Real Dataset Workflow
# ============================================================================
print("\nEXERCISE 10: Complete ML Workflow")
print("-" * 50)

# Create a "sales" dataset
sales_data = pd.DataFrame({
    'product': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
    'quantity': [10, 20, 15, 8, 25, 12, 9, 18],
    'price': [100, 50, 100, 200, 50, 100, 200, 50],
    'customer_type': ['new', 'returning', 'new', 'returning', 'new', 'returning', 'new', 'returning']
})

print(f"Sales data:\n{sales_data}\n")

# TODO: Complete this workflow
# 1. Add a column: "total_sale" = quantity * price
# 2. Calculate average sale by product
# 3. Count sales by customer type
# 4. Find which customer type spends more on average

sales_data['total_sale'] = sales_data['quantity'] * sales_data['price']
print(f"With total_sale:\n{sales_data}\n")

avg_by_product = sales_data.groupby('product')['total_sale'].mean()
print(f"Avg sale by product:\n{avg_by_product}\n")

count_by_type = sales_data.groupby('customer_type').size()
print(f"Count by customer type:\n{count_by_type}\n")

avg_by_type = sales_data.groupby('customer_type')['total_sale'].mean()
print(f"Avg sale by customer type:\n{avg_by_type}\n")

print("\n" + "=" * 50)
print("💡 KEY TAKEAWAYS")
print("=" * 50)
print("""
1. Load data with read_csv()
2. Explore with head(), info(), describe()
3. Clean: dropna(), fillna(), drop_duplicates()
4. Filter: df[condition]
5. Select: df['column'] or df[['col1', 'col2']]
6. Add columns: df['new'] = calculation
7. GroupBy: df.groupby('col').agg(function)
8. Normalize/Standardize before ML
9. Separate X (features) and y (target)
10. Always check data quality FIRST!
""")
