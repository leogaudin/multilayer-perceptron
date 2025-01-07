import numpy as np
from layers import Dense
from activations import relu, softmax
from initializers import random
from loss import cce

layer = Dense(
    shape=(2, 3),
    activation=relu,
    initializer=random
)

input = np.array([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8]
                ])

output = layer.forward(input)

output_layer = Dense(
    shape=(3, 2),
    activation=softmax,
    initializer=random
)

output = output_layer.forward(output)
print(output)

y_true = np.array([0, 1, 0, 1])

loss = cce(output, y_true)
print(loss)
