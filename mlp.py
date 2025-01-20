import layers
import activations
import initializers
import losses
import optimizers
from model import Model
from preprocessing import load_data, to_categorical
from scaler import StandardScaler
from stats import plot_multiple_losses


def main():
    model_losses = []

    X_train, y_train, X_test, y_test = load_data(
        train_path="data_train.csv",
        test_path="data_test.csv"
    )
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    scaler = StandardScaler()
    scaler.fit(X_train)

    model_adam = Model(
        layers=[
            layers.Dense((30, 800), initializers.he),
            activations.ReLU(),
            layers.Dense((800, 400), initializers.he),
            activations.ReLU(),
            layers.Dense((400, 2), initializers.he),
            activations.Softmax()
        ],
        scaler=scaler,
        loss=losses.CCE(),
        optimizer=optimizers.Adam(),
    )

    (
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies
    ) = model_adam.fit(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        epochs=42,
        batch_size=32,
    )

    model_adam.save()

    model_losses.append(train_losses)

    model_sgd = Model(
        layers=[
            layers.Dense((30, 800), initializers.he),
            activations.ReLU(),
            layers.Dense((800, 400), initializers.he),
            activations.ReLU(),
            layers.Dense((400, 2), initializers.he),
            activations.Softmax()
        ],
        scaler=scaler,
        loss=losses.CCE(),
        optimizer=optimizers.SGD(),
    )

    (
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies
    ) = model_sgd.fit(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        epochs=42,
        batch_size=32,
    )

    model_sgd.save()

    model_losses.append(train_losses)

    model_rms = Model(
        layers=[
            layers.Dense((30, 800), initializers.he),
            activations.ReLU(),
            layers.Dense((800, 400), initializers.he),
            activations.ReLU(),
            layers.Dense((400, 2), initializers.he),
            activations.Softmax()
        ],
        scaler=scaler,
        loss=losses.CCE(),
        optimizer=optimizers.RMSprop(),
    )

    (
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies
    ) = model_rms.fit(
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        epochs=42,
        batch_size=32,
    )

    model_rms.save()

    model_losses.append(train_losses)

    plot_multiple_losses(
        losses=model_losses,
        labels=["Adam", "SGD", "RMSprop"],
        title="Losses",
        xlabel="Epochs",
        ylabel="Loss",
    )


if __name__ == "__main__":
    main()
