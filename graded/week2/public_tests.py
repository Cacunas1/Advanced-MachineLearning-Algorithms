import numpy as np
import tensorflow as tf
from keras.activations import linear, relu
from keras.layers import Dense


def test_my_softmax(target):
    z = np.array([1.0, 2.0, 3.0, 4.0])
    a = target(z)
    atf = tf.nn.softmax(z)

    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"

    z = np.array([np.log(0.1)] * 10)
    a = target(z)
    atf = tf.nn.softmax(z)

    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"

    print("\033[92m All tests passed.")


def test_model(target, classes, input_size):
    target.build(input_shape=(None, input_size))

    assert (
        len(target.layers) == 3
    ), f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target.input_shape == (
        None,
        input_size,
    ), f"Wrong input shape. Expected [None,  {input_size}] but got {target.input.shape}"
    i = 0
    expected = [
        [Dense, (None, 25), relu],
        [Dense, (None, 15), relu],
        [Dense, (None, classes), linear],
    ]

    for layer in target.layers:
        assert (
            type(layer) is expected[i][0]
        ), f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert (
            layer.output.shape == expected[i][1]
        ), f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {layer.output.shape}"
        assert (
            layer.activation == expected[i][2]
        ), f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i = i + 1

    print("\033[92mAll tests passed!")
