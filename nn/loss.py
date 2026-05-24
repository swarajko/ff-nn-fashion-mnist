import numpy as np


def cross_entropy_loss(y_true, y_pred):

    epsilon = 1e-10

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        np.sum(y_true * np.log(y_pred), axis=1)
    )

    return loss
