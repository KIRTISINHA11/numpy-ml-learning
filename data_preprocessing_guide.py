import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DATA PREPROCESSING COMPLETE GUIDE WITH COUNTRIES.CSV
# ============================================================================

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================
print("=" * 70)
print("STEP 1: DATA LOADING")
print("=" * 70)

df = pd.read_csv('Countries.csv')
print(f"Dataset shape: {df.shape}")  # (196, 65) - 196 countries, 65 features
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# Check data types
print("\nData Types:")
print(df.dtypes)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# ============================================================================
# STEP 3: HANDLING MISSING VALUES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: HANDLING MISSING VALUES")
print("=" * 70)

# Theory: Missing values are gaps in data that can bias analysis
# Types: MCAR, MAR, MNAR

# Check missing values
print("\nMissing values in each column:")
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_summary[missing_summary['Missing_Count'] > 0])

# Strategy 1: Drop rows with missing values (if < 5% missing)
df_clean = df.dropna()  # Drops rows with ANY missing values
print(f"\nAfter dropping NaN rows: {df_clean.shape}")

# Strategy 2: Drop specific columns with too many missing values
missing_threshold = 50  # Drop columns with >50% missing
cols_to_drop = missing_summary[missing_summary['Missing_Percentage'] > missing_threshold]['Column'].tolist()
print(f"\nColumns with >50% missing: {cols_to_drop}")

# Strategy 3: Fill missing values
df_filled = df.copy()

# Fill numeric columns with median (robust to outliers)
numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = df_filled[col].median()
    df_filled[col].fillna(median_val, inplace=True)

# Fill categorical columns with mode (most frequent value)
categorical_cols = df_filled.select_dtypes(include=['object']).columns
for col in categorical_cols:
    mode_val = df_filled[col].mode()[0] if len(df_filled[col].mode()) > 0 else 'Unknown'
    df_filled[col].fillna(mode_val, inplace=True)

print(f"\nAfter filling missing values: {df_filled.isnull().sum().sum()} total missing")

# Strategy 4: Forward fill / Backward fill (for time series)
# df_filled = df.fillna(method='ffill')  # Forward fill
# df_filled = df.fillna(method='bfill')  # Backward fill

# ============================================================================
# STEP 4: DETECTING & HANDLING DUPLICATES
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: DETECTING & HANDLING DUPLICATES")
print("=" * 70)

# Check for duplicate rows
print(f"\nDuplicate rows: {df_filled.duplicated().sum()}")
print(f"Duplicate countries: {df_filled.duplicated(subset=['country']).sum()}")

# Remove duplicates (keep first occurrence)
df_filled = df_filled.drop_duplicates(subset=['country'], keep='first')
print(f"Dataset shape after removing duplicates: {df_filled.shape}")

# ============================================================================
# STEP 5: OUTLIER DETECTION & HANDLING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: OUTLIER DETECTION & HANDLING")
print("=" * 70)

# Theory: Outliers are unusual data points that deviate significantly from others

# Method 1: IQR (Interquartile Range)
print("\nOutlier Detection - IQR Method (GDP):")
Q1 = df_filled['gdp'].quantile(0.25)
Q3 = df_filled['gdp'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df_filled[(df_filled['gdp'] < lower_bound) | (df_filled['gdp'] > upper_bound)]
print(f"Outliers found (IQR): {len(outliers_iqr)}")
print(f"Bounds: [{lower_bound:.2e}, {upper_bound:.2e}]")

# Method 2: Z-Score (values > 3 std from mean are outliers)
print("\nOutlier Detection - Z-Score Method:")
from scipy import stats
z_scores = np.abs(stats.zscore(df_filled['gdp'].dropna()))
outliers_zscore = z_scores > 3
print(f"Outliers found (Z-Score): {outliers_zscore.sum()}")

# Handle outliers: Cap them (winsorization)
df_filled['gdp_capped'] = df_filled['gdp'].clip(lower=lower_bound, upper=upper_bound)
print(f"\nGDP before capping - Min: {df_filled['gdp'].min():.2e}, Max: {df_filled['gdp'].max():.2e}")
print(f"GDP after capping - Min: {df_filled['gdp_capped'].min():.2e}, Max: {df_filled['gdp_capped'].max():.2e}")

# ============================================================================
# STEP 6: DATA TRANSFORMATION
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: DATA TRANSFORMATION")
print("=" * 70)

# Theory: Transform skewed data to make it more normally distributed

# Check skewness
print("\nSkewness of GDP (right-skewed):")
print(f"Original GDP skewness: {df_filled['gdp'].skew():.4f}")

# Log transformation (for right-skewed data)
df_filled['gdp_log'] = np.log1p(df_filled['gdp'])
print(f"Log-transformed GDP skewness: {df_filled['gdp_log'].skew():.4f}")

# Square root transformation
df_filled['gdp_sqrt'] = np.sqrt(df_filled['gdp'])
print(f"Sqrt-transformed GDP skewness: {df_filled['gdp_sqrt'].skew():.4f}")

# ============================================================================
# STEP 7: NORMALIZATION & SCALING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: NORMALIZATION & SCALING")
print("=" * 70)

# Theory: Scale features to same range (important for ML algorithms)

# Select numeric columns for scaling
numeric_features = ['population', 'gdp', 'life_expectancy', 'inflation']
numeric_features = [col for col in numeric_features if col in df_filled.columns]

print(f"\nBefore scaling - GDP: Min={df_filled['gdp'].min():.2e}, Max={df_filled['gdp'].max():.2e}")

# Method 1: Standardization (Z-score normalization) - mean=0, std=1
scaler_standard = StandardScaler()
df_scaled_standard = df_filled.copy()
df_scaled_standard[numeric_features] = scaler_standard.fit_transform(df_filled[numeric_features].fillna(0))
print(f"\nAfter Standardization - GDP Mean: {df_scaled_standard['gdp'].mean():.4f}, Std: {df_scaled_standard['gdp'].std():.4f}")

# Method 2: Min-Max Scaling (0-1 range)
scaler_minmax = MinMaxScaler()
df_scaled_minmax = df_filled.copy()
df_scaled_minmax[numeric_features] = scaler_minmax.fit_transform(df_filled[numeric_features].fillna(0))
print(f"After Min-Max Scaling - GDP Min: {df_scaled_minmax['gdp'].min():.4f}, Max: {df_scaled_minmax['gdp'].max():.4f}")

# ============================================================================
# STEP 8: CATEGORICAL ENCODING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: CATEGORICAL ENCODING")
print("=" * 70)

# Theory: Convert categorical variables to numeric for ML algorithms

# Method 1: Label Encoding (for ordinal data)
print("\nLabel Encoding - Democracy Type:")
le = LabelEncoder()
df_filled['democracy_type_encoded'] = le.fit_transform(df_filled['democracy_type'].astype(str))
print(df_filled[['democracy_type', 'democracy_type_encoded']].drop_duplicates().head())

# Method 2: One-Hot Encoding (for nominal data)
print("\nOne-Hot Encoding - Continent (first 5 countries):")
df_onehot = pd.get_dummies(df_filled[['continent']], prefix='continent')
print(df_onehot.head())

# ============================================================================
# STEP 9: FEATURE ENGINEERING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: FEATURE ENGINEERING")
print("=" * 70)

# Theory: Create new features from existing ones to improve model performance

# Feature 1: GDP per capita
df_filled['gdp_per_capita'] = df_filled['gdp'] / df_filled['population']
print(f"\nGDP per capita created")

# Feature 2: Urban population ratio
df_filled['urban_ratio'] = df_filled['urban_population'] / df_filled['population']
print(f"Urban ratio created")

# Feature 3: Human development indicator (combination)
df_filled['development_index'] = (df_filled['life_expectancy'] + df_filled['internet_pct']) / 2
print(f"Development index created")

# Feature 4: Binning continuous variables
df_filled['gdp_category'] = pd.cut(df_filled['gdp'],
                                    bins=[0, 1e9, 1e10, 1e11, 1e12, np.inf],
                                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
print("\nGDP Categories:")
print(df_filled['gdp_category'].value_counts())

# ============================================================================
# STEP 10: DATA SPLITTING
# ============================================================================
print("\n" + "=" * 70)
print("STEP 10: DATA SPLITTING")
print("=" * 70)

from sklearn.model_selection import train_test_split

# Prepare data for splitting
X = df_filled.select_dtypes(include=[np.number]).fillna(0)
y = df_filled['life_expectancy']  # Target variable

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Train-Test ratio: {len(X_train)}/{len(X_test)}")

# ============================================================================
# STEP 11: HANDLING CLASS IMBALANCE (if needed)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 11: CLASS DISTRIBUTION CHECK")
print("=" * 70)

# For classification problems
if 'democracy_type' in df_filled.columns:
    print("\nDemocracy Type Distribution:")
    print(df_filled['democracy_type'].value_counts())

# ============================================================================
# SUMMARY & BEST PRACTICES
# ============================================================================
print("\n" + "=" * 70)
print("DATA PREPROCESSING SUMMARY")
print("=" * 70)

summary = f"""
1. DATA LOADING: Load data using pd.read_csv() → {df.shape}

2. EDA: Explore structure, types, statistics

3. MISSING VALUES:
   - Drop rows/columns
   - Fill with mean/median (numeric)
   - Fill with mode (categorical)

4. DUPLICATES: Remove duplicate rows

5. OUTLIERS: Detect with IQR/Z-score → Cap or remove

6. TRANSFORMATION: Log/sqrt for skewed data

7. SCALING:
   - StandardScaler: (x - mean) / std → Mean=0, Std=1
   - MinMaxScaler: (x - min) / (max - min) → Range [0,1]

8. ENCODING:
   - Label Encoding: Ordinal data
   - One-Hot Encoding: Nominal data

9. FEATURE ENGINEERING: Create new meaningful features

10. DATA SPLITTING: Train (80%), Test (20%), Validation (optional)

FINAL CLEANED DATASET: {df_filled.shape}
"""

print(summary)

print("\nPreprocessing complete! Dataset ready for machine learning.")
