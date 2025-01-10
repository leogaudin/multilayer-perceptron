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

model = Model(
    layers=[
        layers.Dense((2, 4), initializers.random),
        activations.ReLU(),
        layers.Dense((4, 6), initializers.random),
        activations.ReLU(),
        layers.Dense((6, 2), initializers.random),
        activations.Softmax()
    ],
    loss=losses.CCE()
)

y_true = np.array([
                    [1, 0],
                    [0, 1],
                    [1, 0],
                    [0, 1]
                ])

cce = losses.CCE()

model.fit(
    x_train=input,
    y_train=y_true,
    epochs=100,
    learning_rate=0.01
)
