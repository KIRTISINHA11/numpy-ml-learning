"""
NumPy for Machine Learning - Practical Guide
Focus: Operations used in actual ML workflows
"""

import numpy as np

# ============================================================================
# 1. BASICS: Arrays & Shapes
# ============================================================================
print("=" * 70)
print("1. ARRAYS & SHAPES")
print("=" * 70)

# Create arrays (you'll do this constantly)
a = np.array([1, 2, 3])  # 1D array
print(f"1D array: {a}, shape: {a.shape}")

b = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array (matrix)
print(f"2D array shape: {b.shape}")  # (2, 3) = 2 rows, 3 columns

# Why shape matters: A dataset is (n_samples, n_features)
# 100 images, each 28x28 pixels → (100, 784) when flattened
# 1000 customers, 5 features each → (1000, 5)

print(f"\nDataset example: 1000 rows × 20 features")
dataset = np.random.randn(1000, 20)
print(f"Shape: {dataset.shape}")
print(f"First sample (row 0): {dataset[0, :5]}")  # First 5 features of sample 0

# ============================================================================
# 2. RESHAPING DATA (Critical for ML)
# ============================================================================
print("\n" + "=" * 70)
print("2. RESHAPING DATA")
print("=" * 70)

# Flatten a matrix (often needed before feeding to model)
img = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3 image
flat = img.reshape(-1)  # -1 means "figure out this dimension"
print(f"Original shape: {img.shape}, Flattened shape: {flat.shape}")

# Reshape back
reshaped = flat.reshape(2, 3)
print(f"Reshaped back: {reshaped.shape}")

# Transpose (flip rows/columns)
data = np.array([[1, 2], [3, 4], [5, 6]])  # 3×2
print(f"\nOriginal (3×2):\n{data}")
print(f"Transposed (2×3):\n{data.T}")

# ============================================================================
# 3. INDEXING & SLICING (You'll do this ALL the time)
# ============================================================================
print("\n" + "=" * 70)
print("3. INDEXING & SLICING")
print("=" * 70)

data = np.arange(20).reshape(4, 5)  # 4×5 matrix
print(f"Data:\n{data}\n")

print(f"Single element [0, 2]: {data[0, 2]}")
print(f"First row: {data[0, :]}")
print(f"First column: {data[:, 0]}")
print(f"Rows 1-3, columns 0-2:\n{data[1:3, 0:2]}")

# CRITICAL: Train/test split uses slicing
train_size = 3
X_train = data[:train_size, :]  # First 3 rows
X_test = data[train_size:, :]   # Remaining rows
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ============================================================================
# 4. ELEMENT-WISE OPERATIONS
# ============================================================================
print("\n" + "=" * 70)
print("4. ELEMENT-WISE OPERATIONS")
print("=" * 70)

x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])

print(f"x: {x}, y: {y}")
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")
print(f"x ** 2 = {x ** 2}")
print(f"np.sqrt(x) = {np.sqrt(x)}")

# This is how neural networks compute activations
z = x * 2 + 1  # Linear transformation
a = np.maximum(0, z)  # ReLU activation
print(f"\nLinear: z = 2x + 1 → {z}")
print(f"ReLU: max(0, z) → {a}")

# ============================================================================
# 5. MATRIX MULTIPLICATION (Core to ML)
# ============================================================================
print("\n" + "=" * 70)
print("5. MATRIX MULTIPLICATION")
print("=" * 70)

# This is literally how neural networks work: output = input @ weights + bias
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
W = np.array([[0.5, 0.1, 0.2], [0.3, 0.4, 0.6]])  # 2 features → 3 neurons

result = X @ W  # Matrix multiply
print(f"X shape: {X.shape}")
print(f"W shape: {W.shape}")
print(f"Result (X @ W) shape: {result.shape}")
print(f"Result:\n{result}")

# ============================================================================
# 6. BROADCASTING (NumPy's superpower)
# ============================================================================
print("\n" + "=" * 70)
print("6. BROADCASTING")
print("=" * 70)

# Problem: Normalize each column independently
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])
mean = np.array([4, 5, 6])  # Mean of each column

# This works even though shapes don't match!
normalized = data - mean
print(f"Data:\n{data}")
print(f"Mean: {mean}")
print(f"Data - mean:\n{normalized}")

# It broadcasts: (3,3) - (3,) → treats (3,) as (1,3) → applies to each row

# Normalize features (standardization in ML)
features = np.array([[10, 100],
                     [20, 200],
                     [30, 300]])
mean = features.mean(axis=0)  # Mean of each column
std = features.std(axis=0)    # Std dev of each column
normalized = (features - mean) / std
print(f"\nFeatures:\n{features}")
print(f"Normalized:\n{normalized}")

# ============================================================================
# 7. STATISTICAL OPERATIONS
# ============================================================================
print("\n" + "=" * 70)
print("7. STATISTICS")
print("=" * 70)

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Data: {data}")
print(f"Mean: {np.mean(data)}")
print(f"Std dev: {np.std(data)}")
print(f"Min: {np.min(data)}, Max: {np.max(data)}")
print(f"Sum: {np.sum(data)}")

# For 2D data (multiple samples)
data_2d = np.random.randn(100, 5)  # 100 samples, 5 features
print(f"\nData shape: {data_2d.shape}")
print(f"Mean per feature: {np.mean(data_2d, axis=0)}")  # axis=0 = along samples
print(f"Std per feature: {np.std(data_2d, axis=0)}")

# ============================================================================
# 8. AGGREGATION ALONG AXES
# ============================================================================
print("\n" + "=" * 70)
print("8. AGGREGATION (axis parameter)")
print("=" * 70)

data = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(f"Data shape: {data.shape}")
print(f"Data:\n{data}")
print(f"Sum all: {np.sum(data)}")
print(f"Sum along axis=0 (down columns): {np.sum(data, axis=0)}")  # → [5, 7, 9]
print(f"Sum along axis=1 (across rows): {np.sum(data, axis=1)}")    # → [6, 15]

# This is used for:
# - Computing loss: sum all errors
# - Computing gradients: sum errors per sample

# ============================================================================
# 9. RANDOM NUMBERS (For initialization & data generation)
# ============================================================================
print("\n" + "=" * 70)
print("9. RANDOM NUMBERS")
print("=" * 70)

# Uniform random [0, 1)
uniform = np.random.rand(5)
print(f"Uniform [0,1): {uniform}")

# Normal/Gaussian distribution (most common for weight init)
normal = np.random.randn(5)
print(f"Normal (mean=0, std=1): {normal}")

# Initialize neural network weights
weights = np.random.randn(10, 5) * 0.01  # Small random values
print(f"NN weights shape: {weights.shape}")

# Random integers
random_ints = np.random.randint(0, 10, size=5)
print(f"Random ints [0,10): {random_ints}")

# ============================================================================
# 10. PRACTICAL ML WORKFLOW EXAMPLE
# ============================================================================
print("\n" + "=" * 70)
print("10. PRACTICAL ML WORKFLOW")
print("=" * 70)

# 1. Load/create synthetic dataset
np.random.seed(42)  # For reproducibility
X = np.random.randn(100, 3)  # 100 samples, 3 features
y = np.array([1 if x.sum() > 0 else 0 for x in X])  # Binary labels

print(f"Dataset: X shape {X.shape}, y shape {y.shape}")

# 2. Split train/test
split_idx = 80
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 3. Normalize features
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std
print(f"Normalized train mean: {X_train_norm.mean(axis=0)}")

# 4. Initialize model weights
n_features = X_train.shape[1]
n_neurons = 5
W = np.random.randn(n_features, n_neurons) * 0.01
b = np.zeros(n_neurons)
print(f"Weights shape: {W.shape}, Bias shape: {b.shape}")

# 5. Forward pass (what happens inside a neural network)
z = X_train_norm @ W + b  # Linear: (100,3) @ (3,5) = (100,5)
a = np.maximum(0, z)      # ReLU activation
print(f"Forward pass output shape: {a.shape}")

# 6. Simple prediction
print(f"First 3 predictions:\n{a[:3]}")

# ============================================================================
# 11. COMMON PITFALLS & SOLUTIONS
# ============================================================================
print("\n" + "=" * 70)
print("11. COMMON PITFALLS")
print("=" * 70)

print("\n❌ Pitfall 1: Shape mismatch")
A = np.array([[1, 2], [3, 4]])  # (2,2)
B = np.array([10, 20])           # (2,)
try:
    # This fails:
    C = A @ B
except ValueError as e:
    print(f"Error: {e}")
print(f"✓ Solution: Reshape B to (2,1) or use dot product correctly")
print(f"A @ B (correct): {A @ B}")  # NumPy is smart enough here

print("\n❌ Pitfall 2: Forgetting axis parameter")
data = np.array([[1, 2], [3, 4]])
print(f"data.sum(): {data.sum()} (sums everything)")
print(f"data.sum(axis=0): {data.sum(axis=0)} (sum each column)")

print("\n❌ Pitfall 3: Modifying original data")
original = np.array([1, 2, 3])
copy = original.copy()  # Must use .copy()!
copy[0] = 999
print(f"Original: {original} (unchanged)")
print(f"Copy: {copy} (modified)")

print("\n" + "=" * 70)
print("SUMMARY: Key operations for ML")
print("=" * 70)
print("""
1. Indexing: data[0], data[:, 0]
2. Reshaping: reshape(), T (transpose)
3. Math: +, -, *, /, ** (element-wise)
4. Matrix multiply: @ operator
5. Aggregation: sum(), mean(), std() with axis=
6. Broadcasting: automatic shape alignment
7. Random: randn() for weights
8. Slicing: train/test splits

Practice: Do these operations 10 times on random data until they feel natural.
""")
