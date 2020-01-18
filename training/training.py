import numpy as np
# https://en.wikipedia.org/wiki/Sigmoid_function
#get result between 0 - 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

train_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

out = np.array([[0, 1, 1, 0]]).T
np.random.seed(1)

syn_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights: ")
print (syn_weights)
#Transfer function calculates the net weight.

for interation in range(1):
     input_layer = train_inputs
     outputs = sigmoid(np.dot(input_layer, syn_weights))
     #Calculation of sum (x1, x2, ..xn) * (w1, ...)

print("Outputs: ")
print(outputs)
#Neural outputs results.
#this course helped me alot for starting and understanding neural network for machine learning.