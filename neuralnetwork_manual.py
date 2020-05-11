import random
import math
    
# RandomMatrix
# Creates a matrix (row x column) with random values (min, max)
def RandomMatrix(row, column, min, max, fill=0.0):
    matrix = []
    for _ in range(row):
        matrix.append([fill]*column)    
    for i in range(row):
        for j in range(column):
            matrix[i][j] = (max-min) * random.random() + min
    return matrix

class NeuralNetwork(object):
    def __init__(self, number_of_inputs, number_of_hidden_layers, number_of_outputs, verbose = False):
        self.Xn = number_of_inputs + 1          # + 1 for bias
        self.Hn = number_of_hidden_layers + 1   # + 1 for bias
        self.Yn = number_of_outputs
        self.verbose = verbose
        
        # 1. Weights in the NN 
        self.layer1 = RandomMatrix(self.Xn, self.Hn, -1.0, 1.0)
        self.layer2 = RandomMatrix(self.Hn, self.Yn, -1.0, 1.0)
        # 2. Nodes in the NN
        self.Xnodes = [1.0]*self.Xn
        self.Hnodes = [1.0]*self.Hn
        self.Ynodes = [1.0]*self.Yn

        if (verbose):
            print("Initialized neural network\n", self.Xn - 1, " input neurons (+1 bias)\n", self.Hn - 1, " hidden neurons (+1 bias)\n", self.Yn, " output neurons\n")
            print("First Matrix (input x hidden) layer:\n", self.layer1)
            print("")
            print("Second Matrix (hidden x output) layer:\n", self.layer2)

    # FeedForward
    # Finds the output value of the neural network
    def FeedForward(self, X):
        # Set the values of the Input nodes with given data
        for i in range(self.Xn - 1):
            self.Xnodes[i] = X[i]

        # Dot product of Input nodes and Layer 1
        for i in range(self.Hn):
            dotproduct = 0.0
            for j in range(self.Xn):
                dotproduct += self.Xnodes[j] * self.layer1[j][i]
            self.Hnodes[i] = self.sigmoid(dotproduct)
        
        # Dot product of Hidden nodes and Layer 2
        for i in range(self.Yn):
            dotproduct = 0.0
            for j in range(self.Hn):
                dotproduct += self.Hnodes[j] * self.layer2[j][i]
            self.Ynodes[i] = self.sigmoid(dotproduct)        
        return self.Ynodes
    
    # Back Propagation
    # 1. Calculate Error in NN's prediction
    # 2. Calculate Error in Hidden Neuron's values
    # 3. Update layer 2 weights
    # 4. Update layer 1 weights
    # 5. (optional) Cost Function (MSE)
    def BackPropagation(self, Y, learning_rate):
        # Calculate output error
        Ydelta = [0.0] * self.Yn
        for i in range(self.Yn):
            error = Y[i] - self.Ynodes[i]
            Ydelta[i] = self.sigmoid_derived(self.Ynodes[i]) * error
        
        # Calculate hidden error
        Wdelta = [0.0] * self.Hn
        for i in range(self.Hn):
            error = 0.0
            for j in range(self.Yn):
                error += Ydelta[j] * self.layer2[i][j]
            Wdelta[i] = self.sigmoid_derived(self.Hnodes[i]) * error
        
        # Update layer 2 (hidden -> output) weights
        for i in range(self.Hn):
            for j in range(self.Yn):
                change = Ydelta[j] * self.Hnodes[i]
                self.layer2[i][j] += (learning_rate * change)

        # Update layer 1 (input -> hidden) weights
        for i in range(self.Xn):
            for j in range(self.Hn):
                change = Wdelta[j] * self.Xnodes[i]
                self.layer1[i][j] += (learning_rate * change)
        
        # Calculate cost
        cost = 0.0
        for i in range(len(Y)):
            cost += ((Y[i]-self.Ynodes[i]) ** 2) / 2.0  # MSE
        return cost

    def Train(self, trainingdata, labels, epochs, learning_rate):
        if (self.verbose):
            print("\n\nTraining the network with ", epochs, " epoch(s).")
        for epoch in range(epochs):
            cost = 0.0
            for data, label in zip(trainingdata, labels):
                self.FeedForward(data)
                cost = cost + self.BackPropagation(label, learning_rate)
            if (self.verbose and epoch % 250 == 0):
                print("Cost @ Epoch #", epoch, ": ", cost)

    def Predict(self, X):
        print(X, self.FeedForward(X))

    def sigmoid(self, x):
        return 1.0/(1.0 + math.exp(-x))
    
    def sigmoid_derived(self, x):
        return x * (1 - x)
