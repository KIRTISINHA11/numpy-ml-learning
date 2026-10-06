"""
NumPy for ML - Practice Exercises
Run these, modify them, break them. Learn by doing.
"""

import numpy as np

# ============================================================================
# EXERCISE 1: Load & Explore Data
# ============================================================================
print("EXERCISE 1: Load & Explore Data")
print("-" * 50)

# Create a fake dataset: 50 customers, 4 features (age, income, credit_score, num_purchases)
np.random.seed(42)
data = np.random.randn(50, 4) * [15, 30000, 100, 5] + [35, 50000, 700, 20]

# TODO: Your task
# 1. Print the shape of data
# 2. Print the mean of each feature (hint: use axis=0)
# 3. Print the first 3 rows

print(f"Shape: {data.shape}")
print(f"Mean per feature: {data.mean(axis=0)}")
print(f"First 3 rows:\n{data[:3]}")

# ============================================================================
# EXERCISE 2: Normalize Features
# ============================================================================
print("\n\nEXERCISE 2: Normalize Features (Standardization)")
print("-" * 50)

# Before ML, we normalize: (x - mean) / std
# This puts all features on same scale

# Calculate mean and std for each feature
mean = data.mean(axis=0)
std = data.std(axis=0)
data_normalized = (data - mean) / std

# TODO: Verify this worked
# 1. Print mean of normalized data (should be ~0)
# 2. Print std of normalized data (should be ~1)

print(f"Normalized mean: {data_normalized.mean(axis=0)}")
print(f"Normalized std: {data_normalized.std(axis=0)}")

# ============================================================================
# EXERCISE 3: Train/Test Split
# ============================================================================
print("\n\nEXERCISE 3: Train/Test Split")
print("-" * 50)

# Common: 80% train, 20% test
train_size = int(0.8 * len(data))
print(f"Total samples: {len(data)}, Train: {train_size}, Test: {len(data) - train_size}")

X_train = data_normalized[:train_size]
X_test = data_normalized[train_size:]

# TODO: Print shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================================
# EXERCISE 4: Matrix Multiplication (Forward Pass)
# ============================================================================
print("\n\nEXERCISE 4: Matrix Multiplication")
print("-" * 50)

# Simulate a neural network layer
# Input: (40, 4) → Output: (40, 3)
# We need weights (4, 3) to go from 4 inputs to 3 outputs

X = X_train
n_inputs = X.shape[1]  # 4
n_outputs = 3

# Initialize random weights
W = np.random.randn(n_inputs, n_outputs) * 0.1
b = np.zeros(n_outputs)

# Forward pass: output = input @ weights + bias
z = X @ W + b  # Linear transformation
a = np.maximum(0, z)  # ReLU activation

# TODO: Print shapes and check output
print(f"X shape: {X.shape}")
print(f"W shape: {W.shape}")
print(f"b shape: {b.shape}")
print(f"z (linear) shape: {z.shape}")
print(f"a (after ReLU) shape: {a.shape}")
print(f"First sample output: {a[0]}")

# ============================================================================
# EXERCISE 5: Compute Loss (Simple MSE)
# ============================================================================
print("\n\nEXERCISE 5: Compute Loss")
print("-" * 50)

# Fake target (what we're trying to predict)
y_true = np.random.randint(0, 2, size=len(X_train))
y_pred = np.random.rand(len(X_train))  # Model outputs between 0-1

# Mean Squared Error: mean((y_true - y_pred)^2)
error = y_true - y_pred
mse = np.mean(error ** 2)

print(f"y_true shape: {y_true.shape}")
print(f"y_pred shape: {y_pred.shape}")
print(f"MSE Loss: {mse:.4f}")

# TODO: Try this with different data
print(f"Min error: {np.min(error):.4f}, Max error: {np.max(error):.4f}")

# ============================================================================
# EXERCISE 6: Batch Processing
# ============================================================================
print("\n\nEXERCISE 6: Batch Processing")
print("-" * 50)

# Mini-batch gradient descent: Process data in chunks
batch_size = 8

# TODO: Process first batch
batch_indices = np.arange(0, batch_size)
X_batch = X_train[batch_indices]
y_batch = y_true[batch_indices]

print(f"Batch indices: {batch_indices}")
print(f"X_batch shape: {X_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")

# TODO: Do the same for second batch (rows 8-16)
batch_indices_2 = np.arange(batch_size, 2 * batch_size)
X_batch_2 = X_train[batch_indices_2]
print(f"Second batch shape: {X_batch_2.shape}")

# ============================================================================
# EXERCISE 7: Accuracy Computation
# ============================================================================
print("\n\nEXERCISE 7: Accuracy for Classification")
print("-" * 50)

# Convert continuous predictions to binary (threshold at 0.5)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Count correct: where prediction == true
correct = (y_pred_binary == y_true)
accuracy = np.mean(correct)  # Percentage of correct predictions

print(f"Predictions: {y_pred_binary[:10]}")
print(f"True labels: {y_true[:10]}")
print(f"Accuracy: {accuracy * 100:.1f}%")

# ============================================================================
# EXERCISE 8: Confusion Matrix Elements
# ============================================================================
print("\n\nEXERCISE 8: Confusion Matrix Elements")
print("-" * 50)

# True Positives: pred=1, true=1
tp = np.sum((y_pred_binary == 1) & (y_true == 1))
# True Negatives: pred=0, true=0
tn = np.sum((y_pred_binary == 0) & (y_true == 0))
# False Positives: pred=1, true=0
fp = np.sum((y_pred_binary == 1) & (y_true == 0))
# False Negatives: pred=0, true=1
fn = np.sum((y_pred_binary == 0) & (y_true == 1))

print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

# Calculate precision and recall
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")

# ============================================================================
# CHALLENGE: Implement Mini Gradient Step (Optional)
# ============================================================================
print("\n\nCHALLENGE: Gradient Descent Step")
print("-" * 50)

# Simplified: Just update bias to reduce error
X_sample = X_train[0:5]  # 5 samples
y_sample = y_true[0:5]

# Forward pass
z = X_sample @ W + b
a = np.maximum(0, z)  # ReLU

# Compute loss
loss_before = np.mean((a - y_sample.reshape(-1, 1)) ** 2)
print(f"Loss before: {loss_before:.4f}")

# Update bias (negative gradient direction)
learning_rate = 0.01
error_signal = np.mean(y_sample.reshape(-1, 1) - a, axis=0)  # Average error
b = b + learning_rate * error_signal

# Forward pass again with new bias
z = X_sample @ W + b
a = np.maximum(0, z)
loss_after = np.mean((a - y_sample.reshape(-1, 1)) ** 2)
print(f"Loss after: {loss_after:.4f}")
print(f"Improved: {loss_before > loss_after}")

print("\n" + "=" * 50)
print("💡 KEY TAKEAWAYS")
print("=" * 50)
print("""
- Shapes: Data is (n_samples, n_features)
- Normalize: (x - mean) / std puts features on same scale
- Matrix multiply: Input @ Weights = Output
- Slicing: Use [start:end] for train/test splits
- Broadcasting: Automatic alignment of shapes
- Axis parameter: axis=0 (down), axis=1 (across)
- Indexing: Use boolean masks for filtering
""")
