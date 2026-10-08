# Pandas for ML - Quick Reference

## Create DataFrames

```python
# From dictionary
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

# From list of lists
df = pd.DataFrame([[1, 'a'], [2, 'b']], columns=['col1', 'col2'])

# From NumPy array
df = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])

# From CSV file (most common)
df = pd.read_csv('file.csv')

# Create Series (single column)
s = pd.Series([1, 2, 3, 4])
```

## Load & Save Data

```python
# Read
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
df = pd.read_json('file.json')

# Write
df.to_csv('file.csv', index=False)
df.to_excel('file.xlsx', index=False)
```

## Explore Data

```python
df.head()                    # First 5 rows
df.tail()                    # Last 5 rows
df.shape                     # (rows, columns)
df.info()                    # Column types, nulls
df.describe()                # Statistics (mean, std, min, max)
df.columns                   # Column names
df.dtypes                    # Data types
df.isnull().sum()           # Count missing values
```

## Select Data

```python
# Single column (returns Series)
df['column_name']
df.column_name

# Multiple columns (returns DataFrame)
df[['col1', 'col2']]

# By position (iloc - integer location)
df.iloc[0]                   # First row
df.iloc[0:3]                 # First 3 rows
df.iloc[0, 1]                # Row 0, Column 1
df.iloc[:, 0]                # All rows, first column

# By label (loc - label based)
df.loc[0]                    # Row with index 0
df.loc[0, 'column']          # Row 0, specific column
```

## Filter Data (Boolean Indexing)

```python
# Single condition
df[df['age'] > 30]           # Rows where age > 30
df[df['name'] == 'Alice']    # Rows where name is 'Alice'

# Multiple conditions (use & for AND, | for OR)
df[(df['age'] > 30) & (df['salary'] > 50000)]
df[(df['region'] == 'North') | (df['region'] == 'South')]

# NOT condition
df[~(df['age'] > 30)]        # Rows where age NOT > 30

# Using isin()
df[df['region'].isin(['North', 'South'])]
```

## Add & Modify Columns

```python
# Add new column
df['new_column'] = 100
df['sum'] = df['col1'] + df['col2']

# Rename columns
df.rename(columns={'old': 'new'})

# Drop columns
df.drop('column', axis=1)
df.drop(['col1', 'col2'], axis=1)

# Modify existing column
df['age'] = df['age'] * 2
```

## Handle Missing Data

```python
# Check for missing
df.isnull()                  # Boolean mask
df.isnull().sum()            # Count missing per column
df.isnull().any()            # Any missing in each column?

# Remove missing
df.dropna()                  # Drop rows with any NaN
df.dropna(subset=['col1'])   # Drop if NaN in specific column

# Fill missing
df.fillna(0)                 # Fill all NaN with 0
df.fillna({'col1': 0, 'col2': 'unknown'})  # Different values per column
df.fillna(df.mean())         # Fill with mean
df['col'].fillna(df['col'].median())  # Fill with median
```

## Data Types

```python
# Check types
df.dtypes
df['col'].dtype

# Convert types
df['col'] = df['col'].astype(int)
df['col'] = df['col'].astype(float)
df['col'] = df['col'].astype(str)
df['col'] = df['col'].astype(bool)

# Convert strings to numeric (handles errors)
pd.to_numeric(df['col'], errors='coerce')

# Map values
df['col'] = df['col'].map({'yes': 1, 'no': 0})
```

## Handle Duplicates

```python
# Check duplicates
df.duplicated()              # Boolean mask
df.duplicated().sum()        # Count duplicates

# Remove duplicates
df.drop_duplicates()         # Remove all duplicate rows
df.drop_duplicates(subset=['col1'])  # Remove based on one column
df.drop_duplicates(keep='first')  # Keep first occurrence (default)
```

## Sort & Rank

```python
# Sort
df.sort_values('age')        # Sort ascending
df.sort_values('age', ascending=False)  # Sort descending
df.sort_values(['col1', 'col2'])  # Sort by multiple columns

# Get top/bottom N
df.nlargest(5, 'salary')     # Top 5 by salary
df.nsmallest(5, 'age')       # Bottom 5 by age

# Rank
df['rank'] = df['salary'].rank()
```

## GroupBy & Aggregation (Critical for ML!)

```python
# Group by single column
df.groupby('department')['salary'].mean()

# Multiple aggregations
df.groupby('department')['salary'].agg(['mean', 'min', 'max', 'sum'])

# Group by multiple columns
df.groupby(['department', 'region'])['salary'].mean()

# Custom aggregation
df.groupby('department').agg({
    'salary': 'mean',
    'age': 'max',
    'employee': 'count'
})

# Get group size
df.groupby('department').size()
```

## Statistics & Correlation

```python
# Single column stats
df['col'].mean()
df['col'].median()
df['col'].std()              # Standard deviation
df['col'].min(), df['col'].max()
df['col'].quantile(0.25)     # 25th percentile

# Correlation (see which features are related)
df.corr()                    # Correlation matrix
df[['col1', 'col2']].corr()  # Correlation between two columns

# Covariance
df.cov()
```

## Reshape Data

```python
# Pivot table (reorganize data)
pd.pivot_table(df, values='salary', index='department', aggfunc='mean')

# Melt (convert wide to long)
pd.melt(df, id_vars=['name'], value_vars=['math', 'english'])

# Concatenate DataFrames
pd.concat([df1, df2])        # Combine rows
pd.concat([df1, df2], axis=1)  # Combine columns

# Merge (join) DataFrames
pd.merge(df1, df2, on='id')  # Inner join (default)
pd.merge(df1, df2, how='left')  # Left join
```

## Apply Functions

```python
# Apply function to column
df['col'].apply(lambda x: x * 2)
df['col'].apply(str.upper)

# Apply to DataFrame
df.apply(lambda x: x.sum())

# Apply custom function
def classify(age):
    if age < 30:
        return 'Young'
    else:
        return 'Old'

df['age_group'] = df['age'].apply(classify)
```

## Common ML Preparation Workflow

```python
# 1. Load data
df = pd.read_csv('data.csv')

# 2. Explore
df.head()
df.info()
df.describe()
df.isnull().sum()

# 3. Clean
df = df.dropna()
df = df.drop_duplicates()

# 4. Transform
df['new_feature'] = df['col1'] + df['col2']

# 5. Separate features and target
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# 6. Normalize/Standardize (for neural networks)
X_normalized = (X - X.min()) / (X.max() - X.min())
X_standardized = (X - X.mean()) / X.std()

# 7. Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 8. Feed to ML model
# model.fit(X_train, y_train)
```

## Quick Comparison: Pandas vs NumPy

| Task | Pandas | NumPy |
|---|---|---|
| Load CSV | `pd.read_csv()` | Manual loop needed |
| Select column | `df['col']` | `array[:, 0]` |
| Handle missing | `dropna()`, `fillna()` | Manual |
| GroupBy | `groupby()` | Manual |
| Label data | Automatic column names | Need to track manually |
| Handle mixed types | Built-in | Harder |
| For ML prep | ✓ Use this! | Use after Pandas |

## Performance Tips

```python
# ✓ Vectorized (fast)
df['col'] > 100

# ❌ Loop (slow)
for i in range(len(df)):
    if df.iloc[i]['col'] > 100:
        pass

# ✓ GroupBy (fast)
df.groupby('col1')['col2'].mean()

# ❌ Manual loop (slow)
for group in df['col1'].unique():
    df[df['col1'] == group]['col2'].mean()
```

## Common Errors & Solutions

| Error | Cause | Solution |
|---|---|---|
| `KeyError: 'column'` | Column doesn't exist | Check spelling, use `df.columns` |
| `TypeError: unsupported operand` | Wrong data type | Convert with `astype()` |
| `NaN` in calculations | Missing data | Use `dropna()` or `fillna()` |
| Memory error | DataFrame too large | Use `chunks` parameter in `read_csv()` |

## When to Use Pandas vs NumPy

**Use Pandas for:**
- Loading data (CSV, Excel, JSON)
- Handling missing data
- Grouping & aggregation
- Data exploration
- Mixed data types

**Use NumPy for:**
- Math operations after data prep
- Neural networks
- Matrix calculations
- Large numerical arrays

---

**Rule of thumb:** Pandas prepares data, NumPy does math, ML models learn patterns.
