# NumPy for ML - Quick Reference

## Create Arrays
```python
np.array([1, 2, 3])           # From list
np.zeros((3, 4))              # All zeros
np.ones((3, 4))               # All ones
np.random.randn(3, 4)         # Normal distribution
np.random.rand(3, 4)          # Uniform [0,1)
np.arange(10)                 # Like range()
np.linspace(0, 1, 10)         # 10 points from 0 to 1
```

## Shapes & Dimensions
```python
a.shape                        # Get shape: (rows, cols)
a.ndim                         # Number of dimensions
a.size                         # Total elements
len(a)                         # First dimension size

a.reshape(10, 5)              # Change shape (must fit)
a.reshape(-1)                 # Flatten to 1D
a.reshape(-1, 1)              # Column vector
a.T                           # Transpose
```

## Indexing & Slicing
```python
# For array with shape (10, 5):
a[0]                          # First row
a[:, 0]                       # First column
a[0, 2]                       # Element at row 0, col 2
a[1:3, 0:2]                   # Rows 1-2, cols 0-1
a[::2, :]                      # Every 2nd row, all cols

# Boolean indexing (filtering)
mask = a > 5
a[mask]                        # Elements where a > 5
indices = np.where(a > 5)      # Get indices where condition true
```

## Element-wise Operations
```python
a + b, a - b, a * b, a / b    # Basic math
a ** 2                         # Power
np.sqrt(a)                     # Square root
np.exp(a)                      # e^a
np.log(a)                      # Natural log
np.abs(a)                      # Absolute value
```

## Matrix Operations (Critical for ML)
```python
# Matrix multiply - shapes must match!
# (m, n) @ (n, p) → (m, p)
a @ b                          # Preferred
np.dot(a, b)                   # Alternative
a.T @ a                        # Transpose then multiply

# Element-wise multiply (different!)
a * b                          # Not matrix multiply!

# Common operations
np.sum(a)                      # Sum all elements
np.mean(a)                     # Average
np.std(a)                      # Standard deviation
np.max(a), np.min(a)          # Max/min
```

## Aggregation with Axis
```python
# 2D array shape (3, 4):
a.sum()                        # Sum all: returns scalar
a.sum(axis=0)                  # Sum each column: returns (4,)
a.sum(axis=1)                  # Sum each row: returns (3,)

# Same for other operations:
a.mean(axis=0)                 # Mean per column
a.std(axis=0)                  # Std dev per column
a.max(axis=1)                  # Max per row
```

## Broadcasting (NumPy's Superpower)
```python
# Shapes don't need to match exactly - they broadcast!
a = np.array([[1, 2], [3, 4], [5, 6]])    # (3, 2)
b = np.array([10, 20])                     # (2,)
a - b                          # Broadcasts to (3, 2)

# Normalize data (subtract mean from each column)
mean = a.mean(axis=0)          # Shape (2,)
normalized = a - mean          # Broadcasts to (3, 2)
```

## Common ML Operations
```python
# Train/test split
split = int(0.8 * len(data))
X_train = data[:split]         # First 80%
X_test = data[split:]          # Last 20%

# Normalize features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_norm = (X - mean) / std

# Forward pass (linear layer)
z = X @ W + b                  # (n_samples, n_in) @ (n_in, n_out) → (n_samples, n_out)

# ReLU activation
a = np.maximum(0, z)           # Max of 0 or z

# Softmax (for classification)
exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for stability
softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Cross-entropy loss
-np.mean(y_true * np.log(y_pred + 1e-8))

# Mean Squared Error
np.mean((y_true - y_pred) ** 2)

# Accuracy
(np.argmax(y_pred, axis=1) == y_true).mean()

# Gradient computation
dW = (X.T @ error) / batch_size
db = np.mean(error, axis=0)

# Gradient descent step
W = W - learning_rate * dW
b = b - learning_rate * db
```

## Random Numbers (for initialization & data generation)
```python
np.random.seed(42)             # Set seed for reproducibility

np.random.randn(5, 3)          # Normal dist: mean=0, std=1
np.random.randn(5, 3) * 0.01   # Small random (weight init)
np.random.rand(5, 3)           # Uniform [0, 1)
np.random.randint(0, 10, size=5)  # Random ints [0, 10)
np.random.shuffle(a)           # Shuffle in-place
np.random.choice(a, size=10)   # Random sample
```

## Important Parameters
```python
keepdims=True                  # Keep dimensions after aggregation
                               # np.sum(a, axis=1, keepdims=True)
                               # (3,4) @ axis=1 → (3,1) not (3,)

dtype=np.float32               # Data type
axis=0                         # Which dimension (0=rows, 1=cols)
```

## Common Mistakes & Fixes

### ❌ Shape Mismatch
```python
# WRONG: (3,) + (3,3)
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a + b  # Broadcasting handles this: applies a to each row

# EXPLICIT: reshape for clarity
a_reshaped = a.reshape(1, -1)  # Now (1, 3)
a_reshaped + b                  # (1, 3) + (3, 3)
```

### ❌ Matrix Multiply vs Element-wise
```python
# NOT the same!
a @ b          # Matrix multiply
a * b          # Element-wise multiply (same shape required)
```

### ❌ Modifying Original Data
```python
# WRONG: This modifies the original!
copy = a
copy[0] = 999
print(a)  # a[0] is now 999!

# CORRECT:
copy = a.copy()
copy[0] = 999
print(a)  # a[0] unchanged
```

### ❌ Axis Confusion
```python
# For (n_samples, n_features):
a.mean(axis=0)  # Mean of each FEATURE (across samples)
a.mean(axis=1)  # Mean of each SAMPLE (across features)
# axis=0 = DOWN (combine rows)
# axis=1 = ACROSS (combine columns)
```

### ❌ Broadcasting Fails
```python
# (3, 4) can broadcast with (4,) - OK
# (3, 4) can broadcast with (1, 4) - OK
# (3, 4) with (3, 1) - OK
# (3, 4) with (3, 5) - ERROR!

# Rule: Dimensions must be equal or one is 1
```

## When to Use What

| Task | NumPy |
|------|-------|
| Load image | `np.array(Image.open())` |
| Normalize data | `(x - mean) / std` |
| Split data | `data[:80%], data[80%:]` |
| Forward pass | `X @ W + b` |
| Compute loss | `np.mean((y_true - y_pred)**2)` |
| Compute gradients | `X.T @ error / batch_size` |
| Update weights | `W = W - lr * dW` |
| Batch processing | `data[batch_idx]` |
| Shuffle data | `np.random.shuffle()` |
| Random init | `np.random.randn(n, m) * 0.01` |

## Performance Tips

```python
# ✓ Vectorized (fast)
result = X @ W

# ❌ Loop (slow)
for i in range(len(X)):
    result[i] = X[i] @ W

# ✓ Use broadcasting
data - mean

# ❌ Manual loop
for i in range(len(data)):
    data[i] -= mean[i % len(mean)]
```

## Debug Commands
```python
print(a.shape)          # Check dimensions
print(a.dtype)          # Check data type
print(a.min(), a.max()) # Check value ranges
np.isnan(a).any()       # Check for NaNs
a[0]                    # Inspect first element
a[:5]                   # Inspect first 5
```

---

**Rule of thumb**: If you're writing a loop over samples or features, you probably should use NumPy's vectorization instead. NumPy is fast; loops are slow.
