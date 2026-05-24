import numpy as np

from nn.loss import cross_entropy_loss


def accuracy(y_true, y_pred):

    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    return np.mean(y_true == y_pred)


def train(
    model,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=5,
    batch_size=32
):

    history = []

    n_samples = X_train.shape[0]

    for epoch in range(epochs):

        indices = np.random.permutation(n_samples)

        X_train = X_train[indices]
        y_train = y_train[indices]

        for start in range(0, n_samples, batch_size):

            end = start + batch_size

            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            predictions = model.forward(X_batch)

            model.backward(X_batch, y_batch)

            optimizer.update(model)

        # validation metrics
        val_pred = model.forward(X_val)

        val_loss = cross_entropy_loss(
            y_val,
            val_pred
        )

        val_acc = accuracy(
            y_val,
            val_pred
        )

        history.append((val_loss, val_acc))

        print(
            f"Epoch {epoch+1} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_acc:.4f}"
        )

    return history
