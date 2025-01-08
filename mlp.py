import numpy as np
import layers
import activations
import initializers
import losses
from model import Model

input = np.array([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8]
                ])

model = Model([
    layers.Dense((2, 3), initializers.random),
    activations.ReLU(),
    layers.Dense((3, 2), initializers.random),
    activations.Softmax()
])

output = model.forward(input)

y_true = np.array([0, 1, 0, 1])

loss = losses.cce(output, y_true)
print(loss)
