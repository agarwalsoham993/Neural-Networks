#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>
using namespace std;


struct Dataset {                    // datastructure
    vector<vector<double>> inputs;
    vector<vector<double>> targets;
};

vector<double> parseCSVLine(string line) {
    vector<double> values;
    stringstream ss(line);
    string val;
    while (getline(ss, val, ',')) {
        try {
            values.push_back(stod(val));        // stod is used to convert a string of floating-point number into a double type
        } catch (...) {
            values.push_back(0.0);
        }
    }
    return values;
}

Dataset loadCSV(const string& filename, int targetSize) {
    Dataset data;
    ifstream file(filename);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    while (getline(file, line)) {
        if (line.empty()) continue;
        vector<double> row = parseCSVLine(line);
        
        if (row.size() <= targetSize) continue; // Skip invalid lines

        vector<double> input;
        vector<double> target;

        size_t inputSize = row.size() - targetSize;
        for (size_t i = 0; i < inputSize; ++i) input.push_back(row[i]);
        for (size_t i = inputSize; i < row.size(); ++i) target.push_back(row[i]);

        data.inputs.push_back(input);
        data.targets.push_back(target);
    }
    return data;
}

void normalizeDataset(Dataset& data) {                  //min-max normalization
    if (data.inputs.empty()) return;

    size_t numInputs = data.inputs[0].size();
    vector<double> minVal(numInputs, 1e9);
    vector<double> maxVal(numInputs, -1e9);

    for (const auto& row : data.inputs) {               // calculate min and max
        for (size_t i = 0; i < numInputs; ++i) {
            if (row[i] < minVal[i]) minVal[i] = row[i];
            if (row[i] > maxVal[i]) maxVal[i] = row[i];
        }
    }

    for (auto& row : data.inputs) {
        for (size_t i = 0; i < numInputs; ++i) {
            if (maxVal[i] - minVal[i] != 0) {            // Apply normalization
                row[i] = (row[i] - minVal[i]) / (maxVal[i] - minVal[i]);
            } else {
                row[i] = 0.0;
            }
        }
    }
    cout << "Data normalized." << endl;
}


inline double sigmoid(double x) {                       // for activation function
    return 1.0 / (1.0 + exp(-x));
}

inline double sigmoidDerivative(double x) {
    return x * (1.0 - x); 
}

inline double randomWeight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

class NeuralNetwork {
private:
    vector<int> topology;
    
    vector<vector<double>> neurons;
    vector<vector<double>> biases;
    
    vector<vector<vector<double>>> weights;                 // [no of layers,[prev_neur]ith,[curr_neur]ith]

    double learningRate;

public:
    // taken structure (e.g., {2, 3, 1} = 2 input, 3 in the hidden, 1 output)
    NeuralNetwork(const vector<int>& topology, double learningRate = 0.1) 
        : topology(topology), learningRate(learningRate) 
    {
        srand(time(NULL)); // Seed random no

        for (size_t i = 0; i < topology.size(); ++i) {
            int numNeurons = topology[i];
            vector<double> layerNeurons(numNeurons, 0.0);
            vector<double> layerBiases(numNeurons, 0.0);
            vector<vector<double>> layerWeights;                    // weights matrix [prev_neurons,curr_neurons]

            if (i > 0) {                                            // weights and biases for hidden and output layer
                int prevNeurons = topology[i - 1];
                for (int j = 0; j < numNeurons; ++j) {
                    layerBiases[j] = randomWeight();
                    
                    vector<double> neuronWeights;
                    for (int k = 0; k < prevNeurons; ++k) {
                        neuronWeights.push_back(randomWeight());
                    }
                    layerWeights.push_back(neuronWeights);
                }
            }

            neurons.push_back(layerNeurons);                        // [[2],[3],[1]]
            biases.push_back(layerBiases);                          // [[3],[1]]
            weights.push_back(layerWeights);                        // [[[2][3]],[[3][1]]]
        }
        cout << "Network Initialized structure: ";
        for(int n : topology) cout << n << " ";
    }

    void feedForward(const vector<double>& input) {
        if (input.size() != topology[0]) {
            cerr << "Error: Input size mismatch." << endl;
            return;
        }

        neurons[0] = input;

        for (size_t i = 1; i < topology.size(); ++i) { // Loop through layers
            int prevLayerSize = topology[i - 1];
            
            for (int j = 0; j < topology[i]; ++j) { // Loop through neurons in that layer
                double activation = 0.0;
                
                for (int k = 0; k < prevLayerSize; ++k) {
                    activation += neurons[i-1][k] * weights[i][j][k];    // a
                }
                activation += biases[i][j];
                neurons[i][j] = sigmoid(activation);                     // z
            }
        }
    }

    // Backpropagation
    void backPropagate(const vector<double>& target) {
        if (target.size() != topology.back()) {
            cerr << "Error: Target size mismatch." << endl;
            return;
        }
        
        vector<vector<double>> errors(topology.size());   //seperatre data structure for errors

        // Error = (Target - Output) * Derivative(Output)       // Output Layer Error
        int lastLayerIdx = topology.size() - 1;
        errors[lastLayerIdx].resize(topology[lastLayerIdx]);
        
        for (int i = 0; i < topology[lastLayerIdx]; ++i) {
            double output = neurons[lastLayerIdx][i];
            double error = (target[i] - output); 
            errors[lastLayerIdx][i] = error * sigmoidDerivative(output);
        }

        for (int i = lastLayerIdx - 1; i > 0; --i) {            // Hidden Layer Errors
            errors[i].resize(topology[i]);
            int nextLayerSize = topology[i+1];
            
            for (int j = 0; j < topology[i]; ++j) {
                double errorSum = 0.0;
                for (int k = 0; k < nextLayerSize; ++k) {
                    errorSum += errors[i+1][k] * weights[i+1][k][j];
                }
                errors[i][j] = errorSum * sigmoidDerivative(neurons[i][j]);         // Sum of (Error_next * Weight_connecting)
            }
        }

        for (size_t i = 1; i < topology.size(); ++i) {
            for (int j = 0; j < topology[i]; ++j) {
                biases[i][j] += learningRate * errors[i][j];
                
                for (int k = 0; k < topology[i-1]; ++k) {
                    weights[i][j][k] += learningRate * errors[i][j] * neurons[i-1][k];      // weights updated
                }
            }
        }
    }

    vector<double> getOutput() {
        return neurons.back();
    }
};

int main() {
    string csvFile = "data.csv";
    ifstream f(csvFile);

    
    vector<int> topology = {2, 3, 1}; 
    int numTargetColumns = 1; 

    Dataset data = loadCSV(csvFile, numTargetColumns);
    
    if (data.inputs.empty()) {
        cerr << "No data loaded." << endl;
        return 1;
    }

    normalizeDataset(data);
    if (data.inputs[0].size() != topology[0]) {
        cerr << "Error: CSV input columns (" << data.inputs[0].size() 
                  << ") do not match input layer size (" << topology[0] << ")." << endl;
        return 1;
    }

    int epochs = 50000;               // epochs
    NeuralNetwork nn(topology, 0.5); // 0.5 Learning Rate

    cout << "Training for " << epochs << " epochs..." << endl;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        for (size_t i = 0; i < data.inputs.size(); ++i) {
            nn.feedForward(data.inputs[i]);
            nn.backPropagate(data.targets[i]);
            
            vector<double> out = nn.getOutput();                    // MSE loss
            double err = data.targets[i][0] - out[0];
            totalError += err * err;
        }

        if ((epoch + 1) % 1000 == 0) {
            cout << "Epoch " << epoch + 1 << " | Average Error: " << totalError / data.inputs.size() << endl;
        }
    }

    cout << "\nFinal Predictions:" << endl;
    cout << "Input\t\tTarget\tPredicted" << endl;
    cout << "------------------------------------" << endl;

    for (size_t i = 0; i < data.inputs.size(); ++i) {
        nn.feedForward(data.inputs[i]);
        vector<double> result = nn.getOutput();
        
        cout << "[";
        for(double val : data.inputs[i]) cout << fixed << setprecision(0) << val << " ";
        cout << "]\t" 
                  << fixed << setprecision(0) << data.targets[i][0] << "\t" 
                  << fixed << setprecision(4) << result[0] << endl;
    }

    return 0;
}