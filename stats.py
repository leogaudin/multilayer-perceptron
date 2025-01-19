import matplotlib.pyplot as plt


def plot_loss_and_accuracy(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
):
    fig, ax = plt.subplots(2)

    ax[0].plot(train_losses, label="Train Loss")
    ax[0].plot(test_losses, label="Test Loss")
    ax[0].legend()

    ax[1].plot(train_accuracies, label="Train Accuracy")
    ax[1].plot(test_accuracies, label="Test Accuracy")
    ax[1].legend()

    plt.show()


def plot_multiple_losses(
    losses,
    labels,
    title="Losses",
    xlabel="Epochs",
    ylabel="Loss",
):
    for i, loss in enumerate(losses):
        plt.plot(loss, label=labels[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
