import matplotlib.pyplot as plt

def plot_learning_curves(train_losses, val_losses):
    """
    Plot the learning curves for training and validation loss across epochs.

    Args:
        train_losses (list): List of training loss values for each epoch.
        val_losses (list): List of validation loss values for each epoch.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='red')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./plots/learning_curves_epoch_loss.png')
    plt.show()
