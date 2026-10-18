NumPy Multi-Layer Perceptron (MLP) from ScratchA lightweight, purely NumPy-based implementation of a Multi-Layer Perceptron (MLP) built from first principles. This repository demonstrates forward propagation, backpropagation via the chain rule, and Stochastic Gradient Descent (SGD) optimization without relying on machine learning frameworks like PyTorch or TensorFlow.📌 FeaturesPure NumPy Implementation: All matrix operations, activation functions, and gradient updates are implemented manually.Activation Functions:ReLU for hidden layer activations.Sigmoid for binary output probabilities.Stable Loss Computation: Binary Cross-Entropy loss with numerical clipping to prevent overflow/underflow.Demonstration Tasks:XOR Logic Gate: Solves non-linearly separable logic functions.Two Moons Dataset: Classifies non-linear, highly intertwined 2D crescent patterns.Visualization: Integrated decision region boundary plotting with matplotlib.📐 Mathematical Overview1. Forward PassFor input matrix $X$:Hidden Layer Linear Transformation:
$$Z^{(1)} = X W^{(1)} + b^{(1)}$$Hidden Layer Activation:
$$A^{(1)} = \text{ReLU}(Z^{(1)}) = \max(0, Z^{(1)})$$Output Layer Linear Transformation:
$$Z^{(2)} = A^{(1)} W^{(2)} + b^{(2)}$$Output Layer Activation (Prediction):
$$\hat{Y} = A^{(2)} = \sigma(Z^{(2)}) = \frac{1}{1 + e^{-Z^{(2)}}}$$2. Loss Function (Binary Cross-Entropy)$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ Y_i \log(\hat{Y}_i) + (1 - Y_i) \log(1 - \hat{Y}_i) \right]$$3. Backpropagation EquationsUsing the chain rule to calculate derivatives backward through the architecture:Output Error Derivative:
$$\delta^{(2)} = \frac{\partial L}{\partial Z^{(2)}} = A^{(2)} - Y$$Output Weight & Bias Gradients:
$$\frac{\partial L}{\partial W^{(2)}} = \frac{1}{N} (A^{(1)})^T \delta^{(2)}, \quad \frac{\partial L}{\partial b^{(2)}} = \frac{1}{N} \sum \delta^{(2)}$$Hidden Layer Error Derivative:
$$\delta^{(1)} = (\delta^{(2)} (W^{(2)})^T) \odot \text{ReLU}'(Z^{(1)})$$Hidden Layer Weight & Bias Gradients:
$$\frac{\partial L}{\partial W^{(1)}} = \frac{1}{N} X^T \delta^{(1)}, \quad \frac{\partial L}{\partial b^{(1)}} = \frac{1}{N} \sum \delta^{(1)}$$🛠️ RequirementsPython 3.8+numpymatplotlibscikit-learn (only for generating the Two Moons dataset)Install dependencies using pip:pip install numpy matplotlib scikit-learn
🚀 Quickstart & UsageRun the main Python script to train the model on both XOR and Two Moons, and display the decision boundaries:import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# --- Activation Functions ---
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

# --- Neural Network Class ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # He Initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Xavier/Glorot Initialization for Sigmoid Output
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, Y, learning_rate):
        m = Y.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = np.dot(self.X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Parameter Update
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, Y, epochs, learning_rate):
        epsilon = 1e-15
        for epoch in range(epochs):
            predictions = self.forward(X)
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))
            self.backward(Y, learning_rate)
            
            if epoch % (epochs // 5) == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
📊 Plotting Decision RegionsTo render decision boundaries across 2D feature spaces:def plot_decision_boundary(model, X, Y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.forward(grid).reshape(xx.shape)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, preds, levels=50, cmap=plt.cm.Spectral, alpha=0.8)
    plt.colorbar(label='Probability Output')
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap=plt.cm.coolwarm, edgecolor='k', s=60)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
📄 LicenseThis project is open-source under the MIT License.