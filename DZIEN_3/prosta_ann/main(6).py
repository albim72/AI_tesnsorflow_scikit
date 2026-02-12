import numpy as np
from simplenn import SimpleNeuralNetwork

network = SimpleNeuralNetwork()
print(network)
# print(network.weights)

#dane wejściowe
train_inputs = np.array([[1,1,0],[1,1,1],[1,1,0],[1,0,0],[0,1,1],[0,1,0]])
train_outputs = np.array([[1,0,1,1,0,1]]).T
train_iterators = 50_000

#trening sieci neuronowej
network.train(train_inputs,train_outputs,train_iterators)
print(f"\nwagi po treningu:\n{network.weights}")

#dane testowe
testdata = np.array([[1,1,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]])

print("predykcja z użyciem modelu")
for data in testdata:
    print(f"wynik dla {data} wynosi: {network.propagation(data)}")
