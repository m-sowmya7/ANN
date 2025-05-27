# ANN

```
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --- Activation Functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- Load and Preprocess Data ---
iris = load_iris()
x = iris.data                            # Features
y = iris.target.reshape(-1, 1)           # Labels reshaped to column vector

# One-hot encode the class labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# --- Neural Network Parameters ---
input_neurons = x_train.shape[1]         # 4 input features
hidden_neurons = 5
output_neurons = y_encoded.shape[1]      # 3 output classes
epochs = 1000
learning_rate = 0.1
np.random.seed(1)                        # For reproducibility

# Initialize weights and biases
weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
bias_hidden = np.zeros((1, hidden_neurons))
weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)
bias_output = np.zeros((1, output_neurons))

# --- Training Loop ---
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(x_train, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Calculate error and perform backpropagation
    error = y_train - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = np.dot(d_output, weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases using gradient descent
    weights_hidden_output += np.dot(hidden_output.T, d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += np.dot(x_train.T, d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Print loss every 100 epochs
    if epoch % 500 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- Testing ---
hidden_test = sigmoid(np.dot(x_test, weights_input_hidden) + bias_hidden)
final_test = sigmoid(np.dot(hidden_test, weights_hidden_output) + bias_output)

# Predict class with highest probability
predictions = np.argmax(final_test, axis=1)
actual = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predictions == actual) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")

```
