"""
Pandas for Machine Learning - Complete Tutorial
Focus: Data loading, cleaning, exploration, and preparation
"""

import pandas as pd
import numpy as np

# ============================================================================
# 1. CREATING DATAFRAMES (Series & DataFrames)
# ============================================================================
print("=" * 70)
print("1. CREATING DATAFRAMES")
print("=" * 70)

# Series = Single column (1D)
ages = pd.Series([25, 30, 35, 40, 28])
print(f"Series:\n{ages}\n")

# DataFrame = Multiple columns (2D) - like a table
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 28],
    'salary': [50000, 60000, 75000, 80000, 55000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
}
df = pd.DataFrame(data)
print(f"DataFrame:\n{df}\n")

# Basic info
print(f"Shape: {df.shape}")  # (5 rows, 4 columns)
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}\n")

# ============================================================================
# 2. LOADING DATA (Most common in ML)
# ============================================================================
print("=" * 70)
print("2. LOADING DATA FROM FILES")
print("=" * 70)

# Create a sample CSV file for practice
sample_csv = """student_id,math,english,science,passed
1,85,90,88,1
2,92,88,95,1
3,75,70,65,0
4,88,85,90,1
5,65,60,55,0
6,95,92,98,1
7,72,68,70,0
8,85,87,89,1"""

with open('sample_data.csv', 'w') as f:
    f.write(sample_csv)

# Load CSV
df_csv = pd.read_csv('sample_data.csv')
print(f"Loaded CSV:\n{df_csv}\n")

# ============================================================================
# 3. EXPLORING DATA (Critical for ML)
# ============================================================================
print("=" * 70)
print("3. EXPLORING DATA")
print("=" * 70)

# View first/last rows
print(f"First 3 rows:\n{df_csv.head(3)}\n")
print(f"Last 2 rows:\n{df_csv.tail(2)}\n")

# Info about data
print(f"Data info:")
print(df_csv.info())
print()

# Statistics
print(f"Summary statistics:\n{df_csv.describe()}\n")

# Check for missing values
print(f"Missing values:\n{df_csv.isnull().sum()}\n")

# ============================================================================
# 4. SELECTING DATA (Indexing & Filtering)
# ============================================================================
print("=" * 70)
print("4. SELECTING DATA")
print("=" * 70)

# Select single column
print(f"Math scores (Series):\n{df_csv['math']}\n")

# Select multiple columns
print(f"Math and English:\n{df_csv[['math', 'english']]}\n")

# Select by position (iloc - integer location)
print(f"First row: {df_csv.iloc[0].to_dict()}\n")
print(f"Rows 1-3:\n{df_csv.iloc[1:3]}\n")

# Select by condition (Boolean indexing)
passed = df_csv['passed'] == 1
print(f"Students who passed:\n{df_csv[passed]}\n")

# Multiple conditions
high_math = (df_csv['math'] > 85) & (df_csv['english'] > 85)
print(f"High achievers (math>85 AND english>85):\n{df_csv[high_math]}\n")

# ============================================================================
# 5. HANDLING MISSING DATA
# ============================================================================
print("=" * 70)
print("5. HANDLING MISSING DATA")
print("=" * 70)

# Create data with missing values
data_missing = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, np.nan, 35, 40, 28],
    'salary': [50000, 60000, np.nan, 80000, 55000]
}
df_miss = pd.DataFrame(data_missing)
print(f"Data with missing values:\n{df_miss}\n")

# Check missing
print(f"Missing count:\n{df_miss.isnull().sum()}\n")

# Remove rows with any missing
df_clean1 = df_miss.dropna()
print(f"After dropna():\n{df_clean1}\n")

# Fill missing with value
df_clean2 = df_miss.fillna(0)
print(f"After fillna(0):\n{df_clean2}\n")

# Fill with mean
df_clean3 = df_miss.copy()
df_clean3['age'] = df_clean3['age'].fillna(df_clean3['age'].mean())
print(f"Age filled with mean:\n{df_clean3}\n")

# ============================================================================
# 6. DATA MANIPULATION (Transform, Group, Aggregate)
# ============================================================================
print("=" * 70)
print("6. DATA MANIPULATION")
print("=" * 70)

# Add new column
df_csv['total_score'] = df_csv['math'] + df_csv['english'] + df_csv['science']
df_csv['average'] = df_csv[['math', 'english', 'science']].mean(axis=1)
print(f"Added columns:\n{df_csv[['student_id', 'math', 'average', 'total_score']]}\n")

# Rename columns
df_csv_renamed = df_csv.rename(columns={'math': 'math_score', 'english': 'english_score'})
print(f"Renamed columns: {df_csv_renamed.columns.tolist()}\n")

# Sort data
df_sorted = df_csv.sort_values('average', ascending=False)
print(f"Sorted by average (descending):\n{df_sorted[['student_id', 'average']]}\n")

# ============================================================================
# 7. GROUPBY & AGGREGATION (Essential for EDA)
# ============================================================================
print("=" * 70)
print("7. GROUPBY & AGGREGATION")
print("=" * 70)

# Simple aggregation on original df
df_agg = df.copy()
print(f"Average salary by department:\n{df_agg.groupby('department')['salary'].mean()}\n")

# Multiple aggregations
print(f"Multiple stats by department:")
print(df_agg.groupby('department')['salary'].agg(['mean', 'min', 'max']))
print()

# Group by passed/failed
print(f"Average score by pass/fail:")
print(df_csv.groupby('passed')[['math', 'english', 'science']].mean())
print()

# ============================================================================
# 8. HANDLING DUPLICATES
# ============================================================================
print("=" * 70)
print("8. HANDLING DUPLICATES")
print("=" * 70)

# Create data with duplicates
data_dup = pd.DataFrame({
    'id': [1, 2, 2, 3, 3, 3],
    'value': [10, 20, 20, 30, 30, 30]
})
print(f"Data with duplicates:\n{data_dup}\n")

# Check duplicates
print(f"Duplicate rows:\n{data_dup[data_dup.duplicated()]}\n")

# Remove duplicates
df_unique = data_dup.drop_duplicates()
print(f"After drop_duplicates():\n{df_unique}\n")

# ============================================================================
# 9. DATA TYPES & CONVERSION
# ============================================================================
print("=" * 70)
print("9. DATA TYPES & CONVERSION")
print("=" * 70)

# Check types
print(f"Current types:\n{df_csv.dtypes}\n")

# Convert types
df_csv['passed'] = df_csv['passed'].astype(bool)  # int to bool
print(f"Converted 'passed' to bool: {df_csv['passed'].dtype}\n")

# String to numeric
data_str = pd.DataFrame({'value': ['10', '20', '30', '40']})
data_str['value'] = pd.to_numeric(data_str['value'])
print(f"Converted strings to numeric: {data_str['value'].dtype}\n")

# ============================================================================
# 10. STATISTICS & CORRELATION (For ML feature understanding)
# ============================================================================
print("=" * 70)
print("10. STATISTICS & CORRELATION")
print("=" * 70)

# Correlation matrix (see which features are related)
print(f"Correlation matrix:\n{df_csv[['math', 'english', 'science']].corr()}\n")

# Individual statistics
print(f"Math statistics:")
print(f"  Mean: {df_csv['math'].mean():.2f}")
print(f"  Std:  {df_csv['math'].std():.2f}")
print(f"  Min:  {df_csv['math'].min()}")
print(f"  Max:  {df_csv['math'].max()}")
print(f"  Median: {df_csv['math'].median()}\n")

# ============================================================================
# 11. PREPARING DATA FOR ML (Common workflow)
# ============================================================================
print("=" * 70)
print("11. PREPARING DATA FOR ML")
print("=" * 70)

# Separate features (X) and target (y)
X = df_csv[['math', 'english', 'science']]
y = df_csv['passed']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}\n")

# Normalize features (0-1)
X_normalized = (X - X.min()) / (X.max() - X.min())
print(f"Normalized features (first row):\n{X_normalized.iloc[0]}\n")

# Standardize features (mean=0, std=1)
X_standardized = (X - X.mean()) / X.std()
print(f"Standardized features (first row):\n{X_standardized.iloc[0]}\n")

# Train/test split (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

# ============================================================================
# 12. COMMON PANDAS OPERATIONS (Cheat Sheet)
# ============================================================================
print("=" * 70)
print("12. COMMON OPERATIONS SUMMARY")
print("=" * 70)

print("""
Common Operations:
- df.head() / df.tail()         → View first/last rows
- df.shape                      → Get dimensions
- df.info()                     → Column types and nulls
- df.describe()                 → Statistics
- df['column']                  → Select column
- df[['col1', 'col2']]         → Select multiple
- df[df['age'] > 30]           → Filter rows
- df.fillna(value)             → Fill missing
- df.dropna()                  → Remove missing
- df.drop_duplicates()         → Remove duplicates
- df.groupby('col').mean()     → Group and aggregate
- df.sort_values('col')        → Sort data
- df.rename(columns={...})     → Rename columns
- df.astype(type)              → Convert type
- df.corr()                    → Correlation matrix
- df.isnull().sum()            → Count missing
""")

print("=" * 70)
print("💡 KEY TAKEAWAY FOR ML")
print("=" * 70)
print("""
Pandas Workflow:
1. Load data with read_csv()
2. Check data: info(), describe(), isnull()
3. Handle missing: dropna(), fillna()
4. Remove duplicates: drop_duplicates()
5. Filter/select features
6. Transform features (normalize, standardize)
7. Separate X (features) and y (target)
8. Split train/test
9. Feed to ML model

Most ML issues come from bad data, not bad models.
Use Pandas to fix data first!
""")
