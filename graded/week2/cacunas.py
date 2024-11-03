# import keras as kr
import logging

import autils as utl
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential

import public_tests as tst


def my_softmax(z: np.ndarray) -> np.ndarray:
    """Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    ### START CODE HERE ###
    denominator = np.sum(np.exp(z))
    a: np.ndarray = np.exp(z) / denominator
    ### END CODE HERE ###
    return a


def main():
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)

    np.set_printoptions(precision=2)

    utl.plt_act_trio()
    z = np.array([1.0, 2.0, 3.0, 4.0])
    a = my_softmax(z)
    atf = tf.nn.softmax(z)
    print(f"my_softmax(z):         {a}")
    print(f"tensorflow softmax(z): {atf}")

    # BEGIN UNIT TEST
    tst.test_my_softmax(my_softmax)

    X: np.ndarray
    y: np.ndarray
    X, y = utl.load_data()

    print(f"X dimensions: ({X.shape[0]:_}, {X.shape[1]})")
    print(f"y dimensions: ({y.shape[0]:_}, {y.shape[1]})")

    tf.random.set_seed(1234)  # for consistent results

    model = Sequential(
        [
            ### START CODE HERE ###
            Input(shape=(400,)),
            Dense(25, activation="relu"),
            Dense(15, activation="relu"),
            Dense(10, activation="linear"),
            ### END CODE HERE ###
        ],
        name="my_model",
    )

    model.summary()

    tst.test_model(model, 10, 400)


if __name__ == "__main__":
    main()
