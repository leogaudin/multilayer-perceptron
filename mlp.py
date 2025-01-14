# import numpy as np
import layers
import activations
import initializers
import losses
from model import Model
from preprocessing import load_data, to_categorical

X_train, y_train, X_test, y_test = load_data(
    train_path="data_train.csv",
    test_path="data_test.csv"
)

# # Test with XOR instead

# X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_train = np.array([0, 1, 1, 0])
# X_test = X_train
# y_test = y_train

model = Model(
    layers=[
        layers.Dense((10, 7), initializers.random),
        activations.ReLU(),
        layers.Dense((7, 7), initializers.random),
        activations.ReLU(),
        layers.Dense((7, 2), initializers.random),
        activations.Softmax()
    ],
)

model.fit(
    x_train=X_train,
    y_train=to_categorical(y_train),
    x_test=X_test,
    y_test=to_categorical(y_test),
    epochs=100,
    learning_rate=0.001,
    loss_function=losses.CCE(),
)
