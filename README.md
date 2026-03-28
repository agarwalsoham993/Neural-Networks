# Neural Network from Scratch in C++

Implementation and building of Neural Network from scratch in C++, with both forward and Backward Propagation to keep a mathematics first perspective about the central core architecture of modern AI.

This repository by means of NN we have the foundational implementation of a Multi-Layer Perceptron (MLP) Neural Network written entirely in C++.
Built without the use of external machine learning libraries,and thus this project serves as an educational framework to understand the core mathematics and architecture behind deep learning,
including Forward Propagation, Backpropagation, and Gradient Descent.

<img width="944" height="790" alt="Screenshot from 2025-12-06 17-34-06" src="https://github.com/user-attachments/assets/bd27906f-8cdb-4946-8349-5e10dbd341a8" />

## Parts

* **Dynamic Topology**: The network supports dynamic layer generation (e.g., `Input -> Hidden -> Output`) using a simple integer vector representation.
* **parsers**: Includes a raw CSV parser and Min-Max data normalizer.
* **Maths **: Sigmoid activation, derivatives, and Mean Squared Error (MSE) loss natively.

---

## Step-by-Step Architecture Guide

The neural network operates sequentially,lets learn providing code snippets to illustrate the logic used.

### Data Ingestion and Normalization


Before a network can learn, the data must be formatted and scaled.
The `loadCSV` function parses comma-separated values, separating features from target labels. 
The inputs are scaled using Min-Max Normalization.

Why:
- To prevent exploding gradients.
- ensure smooth convergence.

```cpp
void normalizeDataset(Dataset& data) {
    // ... finds minVal and maxVal for each feature ...
    for (auto& row : data.inputs) {
        for (size_t i = 0; i < numInputs; ++i) {
            if (maxVal[i] - minVal[i] != 0) {
                // Scales values strictly between 0.0 and 1.0
                row[i] = (row[i] - minVal[i]) / (maxVal[i] - minVal[i]);
            }
        }
    }
}
```

### Network creation

The network's structure is defined by an array representing the number of neurons in each layer 

(e.g., `{2, 3, 1}` for 2 inputs, 3 hidden neurons, 1 output).

Weights and biases are initialized randomly between `-1.0` and `1.0`.

The weights are stored in a 3D Vector: `[layer_index][current_neuron][previous_neuron]`.

```cpp
// Constructor Initialization
for (size_t i = 0; i < topology.size(); ++i) {
    int numNeurons = topology[i];
    // ... initialize neuron and bias arrays ...
    if (i > 0) { 
        int prevNeurons = topology[i - 1];
        for (int j = 0; j < numNeurons; ++j) {
            layerBiases[j] = randomWeight(); // Random bias initialization
            vector<double> neuronWeights;
            for (int k = 0; k < prevNeurons; ++k) {
                neuronWeights.push_back(randomWeight()); // Random weight initialization
            }
            layerWeights.push_back(neuronWeights);
        }
    }
}
```

### Forward Propagation

During forward propagation, the network processes the input data layer by layer.

For each neuron,
- it calculates the weighted sum of inputs from the previous layer,
- adds the bias,
- and passes the result through a Sigmoid activation function to introduce non-linearity.

```cpp
void feedForward(const vector<double>& input) {
    neurons[0] = input; // Set input layer

    for (size_t i = 1; i < topology.size(); ++i) { 
        for (int j = 0; j < topology[i]; ++j) { 
            double activation = 0.0;
            // Calculate: (Weight * Input)
            for (int k = 0; k < topology[i - 1]; ++k) {
                activation += neurons[i-1][k] * weights[i][j][k];
            }
            activation += biases[i][j]; // Add Bias
            neurons[i][j] = sigmoid(activation); // Squash via Sigmoid
        }
    }
}
```

### Backpropagation, The core learning mechanism.

The network calculates the error at the output layer by comparing predictions to actual targets.

It then pushes this error backward through the hidden layers using the chain rule (represented by the Sigmoid derivative).

Finally, weights and biases are updated using the defined learning rate.

```cpp
void backPropagate(const vector<double>& target) {
    // 1. Calculate Output Layer Error: (Target - Output) * Derivative(Output)
    double output = neurons[lastLayerIdx][i];
    errors[lastLayerIdx][i] = (target[i] - output) * sigmoidDerivative(output);

    // 2. Calculate Hidden Layer Errors
    for (int i = lastLayerIdx - 1; i > 0; --i) {
        for (int j = 0; j < topology[i]; ++j) {
            double errorSum = 0.0;
            for (int k = 0; k < nextLayerSize; ++k) {
                errorSum += errors[i+1][k] * weights[i+1][k][j];
            }
            errors[i][j] = errorSum * sigmoidDerivative(neurons[i][j]);
        }
    }

    // 3. Update Weights and Biases (Gradient Descent)
    for (size_t i = 1; i < topology.size(); ++i) {
        for (int j = 0; j < topology[i]; ++j) {
            biases[i][j] += learningRate * errors[i][j];
            for (int k = 0; k < topology[i-1]; ++k) {
                weights[i][j][k] += learningRate * errors[i][j] * neurons[i-1][k];
            }
        }
    }
}
```

### The Training Loop

The `main` function ties it all together by feeding the entire dataset through the network multiple times (epochs).

It monitors the Mean Squared Error (MSE) to ensure the model is learning and converging properly.

```cpp
NeuralNetwork nn({2, 3, 1}, 0.5); // Topology: 2-3-1, Learning Rate: 0.5

for (int epoch = 0; epoch < epochs; ++epoch) {
    double totalError = 0.0;

    for (size_t i = 0; i < data.inputs.size(); ++i) {
        nn.feedForward(data.inputs[i]);
        nn.backPropagate(data.targets[i]);
        
        // Accumulate MSE Loss
        vector<double> out = nn.getOutput();
        double err = data.targets[i][0] - out[0];
        totalError += err * err;
    }
    // Periodic logging of network improvement...
}
```

## How to Run

1. **Prerequisites**: A standard C++ compiler (such as `g++`).
2. **Data**: Ensure `data.csv` is in the same directory as the executable.
3. **Compile**:
   ```bash
   g++ nn.cpp -o neural_network
   ```
4. **Execute**:
   ```bash
   ./neural_network
   ```

*The console will output the initialized structure, the training loss per 1000 epochs, and a final table comparing target outputs to the network's final predictions.*
