from collections.abc import Callable
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense

import autils as utl
from public_tests import test_c1, test_c2, test_c3


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def my_dense(a_in: np.ndarray, w: np.ndarray, b: np.ndarray, g: Callable) -> np.ndarray:
    """
    Computes dense layer
    Args:
        a_in (ndarray (n, )) : Data, 1 example
        w (ndarray (n, j)) : Weight matrix, n features per unit, j units
        b (ndarray (j, )) : bias vector, j units
        g : activation function (e.g. sigmoid, relu, etc.)
    Returns
        a_out (ndarray (j, )) : j units
    """
    n, units = w.shape
    a_in.reshape(1, n)
    b.reshape(1, units)
    z: np.ndarray = a_in @ w + b
    a_out: np.ndarray = g(z)
    a_out.flatten()

    return a_out


def my_sequential(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
):
    a1 = my_dense(x, W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return a3


def my_dense_v(
    A_in: np.ndarray, W: np.ndarray, b: np.ndarray, g: Callable
) -> np.ndarray:
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
    """
    ### START CODE HERE ###
    z: np.ndarray = A_in @ W + b
    A_out: np.ndarray = g(z)
    ### END CODE HERE ###
    return A_out


def my_sequential_v(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
):
    A1 = my_dense_v(X, W1, b1, sigmoid)
    A2 = my_dense_v(A1, W2, b2, sigmoid)
    A3 = my_dense_v(A2, W3, b3, sigmoid)
    return A3


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

    L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
    L2_num_params = 25 * 15 + 15  # W2 parameters  + b2 parameters
    L3_num_params = 15 * 1 + 1  # W3 parameters  + b3 parameters
    print(
        "L1 params = ",
        L1_num_params,
        ", L2 params = ",
        L2_num_params,
        ",  L3 params = ",
        L3_num_params,
    )

    [layer1, layer2, layer3] = model.layers
    W1, b1 = layer1.get_weights()
    W2, b2 = layer2.get_weights()
    W3, b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    print(model.layers[2].weights)
    prediction = model.predict(X[0].reshape(1, 400))  # a zero
    print(f" predicting a zero: {prediction}")
    prediction = model.predict(X[500].reshape(1, 400))  # a one
    print(f" predicting a one:  {prediction}")
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print(f"prediction after threshold: {yhat}")

    x_tst = 0.1 * np.arange(1, 3, 1).reshape(
        2,
    )  # (1 examples, 2 features)
    W_tst = 0.1 * np.arange(1, 7, 1).reshape(
        2, 3
    )  # (2 input features, 3 output features)
    b_tst = 0.1 * np.arange(1, 4, 1).reshape(
        3,
    )  # (3 features)
    A_tst = my_dense(x_tst, W_tst, b_tst, sigmoid)
    print(A_tst)

    test_c2(my_dense)

    # We can copy trained weights and biases from Tensorflow.
    W1_tmp, b1_tmp = layer1.get_weights()
    W2_tmp, b2_tmp = layer2.get_weights()
    W3_tmp, b3_tmp = layer3.get_weights()

    # make predictions
    prediction = my_sequential(X[0], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp)
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0

    print("yhat = ", yhat, " label= ", y[0, 0])

    prediction = my_sequential(X[500], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp)

    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0

    print("yhat = ", yhat, " label= ", y[500, 0])

    X_tst = 0.1 * np.arange(1, 9, 1).reshape(4, 2)  # (4 examples, 2 features)
    W_tst = 0.1 * np.arange(1, 7, 1).reshape(
        2, 3
    )  # (2 input features, 3 output features)
    b_tst = 0.1 * np.arange(1, 4, 1).reshape(1, 3)  # (1,3 features)
    A_tst = my_dense_v(X_tst, W_tst, b_tst, sigmoid)
    print(A_tst)

    test_c3(my_dense_v)

    W1_tmp, b1_tmp = layer1.get_weights()
    W2_tmp, b2_tmp = layer2.get_weights()
    W3_tmp, b3_tmp = layer3.get_weights()

    # Let's make a prediction with the new model. This will make a prediction on *all of the examples at once*. Note the shape of the output.

    Prediction = my_sequential_v(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp)
    Prediction.shape

    # We'll apply a threshold of 0.5 as before, but to all predictions at once.
    Yhat = (Prediction >= 0.5).astype(int)
    print("predict a zero: ", Yhat[0], "predict a one: ", Yhat[500])


if __name__ == "__main__":
    main()
