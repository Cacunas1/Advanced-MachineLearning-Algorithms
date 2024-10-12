import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense

import autils as utl
from public_tests import test_c1


def main():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)

    X: np.ndarray
    y: np.ndarray

    X, y = utl.load_data()

    print(f"X shape: ({X.shape[0]:_}, {X.shape[1]:_})")
    print(f"y shape: ({y.shape[0]:_}, {y.shape[1]:_})")
    # print(f"The first element of X is: {X[0]}")

    warnings.simplefilter(action="ignore", category=FutureWarning)

    m, n = X.shape

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1)

    for ax in axes.flat:
        random_index = np.random.randint(m)

        X_random_reshaped = X[random_index].reshape((20, 20)).T

        ax.imshow(X_random_reshaped, cmap="gray")

        ax.set_title(y[random_index, 0])
        ax.set_axis_off()

    model = Sequential(
        [
            Input(shape=(400,)),
            ### START CODE HERE ###
            Dense(25, activation="sigmoid", name="layer1"),
            Dense(15, activation="sigmoid", name="layer2"),
            Dense(1, activation="sigmoid", name="layer3"),
            ### END CODE HERE ###
        ],
    )
    model.summary()

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(0.001),
    )

    try:
        test_c1(model)
    except ValueError:
        print(model.__repr__())

    model.fit(X, y, epochs=20)

    y_p = model.predict(X[0].reshape(1, 400))

    print(y_p)


if __name__ == "__main__":
    main()
