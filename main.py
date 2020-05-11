from neuralnetwork_manual import NeuralNetwork
import numpy

# Neural Network
network = NeuralNetwork(2, 4, 1, True)

# Data
trainingdata = [[1,0], [1,0], [0,1], [0,0]]
labels       = [[1], [1], [1], [0]]

# Train
network.Train(trainingdata, labels, 10000, 0.1)

# Test
network.Predict([1,1])
network.Predict([1,0])
network.Predict([0,1])
network.Predict([0,0])
