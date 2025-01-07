import numpy as np
import layers
import activations
import initializers
import losses

layer = layers.Dense(
    shape=(2, 3),
    activation=activations.relu,
    initializer=initializers.random
)

input = np.array([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8]
                ])

output = layer.forward(input)

output_layer = layers.Dense(
    shape=(3, 2),
    activation=activations.softmax,
    initializer=initializers.random
)

output = output_layer.forward(output)
print(output)

y_true = np.array([0, 1, 0, 1])

loss = losses.cce(output, y_true)
print(loss)
