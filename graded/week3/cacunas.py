import logging
import os

import assigment_utils as utl
import keras
import matplotlib.pyplot as plt
import numpy as np
import public_tests_a1 as tst
import tensorflow as tf
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split


def eval_mse(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m, 1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m, 1))  predicted value of each example
    Returns:
      err: (scalar)
    """
    m = len(y)
    err = 0.0

    ### START CODE HERE ###
    err_vec: np.ndarray = y - yhat
    err = np.dot(err_vec, err_vec) / (2 * m)
    ### END CODE HERE ###

    return err


def eval_cat_err(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m, 1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m, 1))  predicted value of each example
    Returns:|
      cerr: (scalar)
    """
    m: int = len(y)
    # incorrect = 0
    ### START CODE HERE ###
    err: np.ndarray = y != yhat
    cerr: float = np.sum(err) / m
    ### END CODE HERE ###

    return cerr


def main():
    # print(keras.version())
    # print(sklearn.__version__)
    dlc: dict[str, str] = dict(
        dlblue="#0096ff",
        dlorange="#FF9300",
        dldarkred="#C00000",
        dlmagenta="#FF40FF",
        dlpurple="#7030A0",
        dldarkblue="#0D5BDC",
    )
    dlblue: str = "#0096ff"
    dlorange: str = "#FF9300"
    dldarkred: str = "#C00000"
    dlmagenta: str = "#FF40FF"
    dlpurple: str = "#7030A0"
    dldarkblue: str = "#0D5BDC"
    dlcolors: list[str] = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
    plt.style.use("./deeplearning.mplstyle")

    # %% Part 1
    print("=" * 80)
    print_str: str = "=== Part 1"
    print(print_str, "=" * (80 - len(print_str) - 1))
    print("=" * 80)

    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    tf.autograph.set_verbosity(0)

    keras.backend.set_floatx("float64")

    X: np.ndarray
    y: np.ndarray
    x_ideal: np.ndarray
    y_ideal: np.ndarray

    X, y, x_ideal, y_ideal = utl.gen_data(18, 2, 0.7)

    print("X.shape", X.shape, "y.shape", y.shape)

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1
    )
    print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
    print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

    y_hat: np.ndarray = np.array([2.4, 4.2])
    y_tmp: np.ndarray = np.array([2.3, 4.1])
    eval_mse(y_hat, y_tmp)

    print("-" * 80)
    print("\tGraded Test 1:")

    tst.test_eval_mse(eval_mse)
    print("-" * 80)

    # %% Part 2
    print("=" * 80)
    print_str = "=== Part 2"
    print(print_str, "=" * (80 - len(print_str) - 1))
    print("=" * 80)

    degree: int = 10
    lmodel: utl.lin_model = utl.lin_model(degree)
    lmodel.fit(X_train, y_train)

    # predict on training data, find training error
    yhat: np.ndarray = lmodel.predict(X_train)
    err_train: float = lmodel.mse(y_train, yhat)

    # predict on test data, find error
    yhat = lmodel.predict(X_test)
    err_test = lmodel.mse(y_test, yhat)

    print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")

    x: np.ndarray = np.linspace(0, int(X.max()), 100)  # predict values for plot
    y_pred = lmodel.predict(x).reshape(-1, 1)

    utl.plt_train_test(
        X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree
    )

    X, y, x_ideal, y_ideal = utl.gen_data(40, 5, 0.7)
    print("X.shape", X.shape, "y.shape", y.shape)

    # split the data using sklearn routine
    X_: np.ndarray
    y_: np.ndarray
    X_cv: np.ndarray
    y_cv: np.ndarray
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_, y_, test_size=0.50, random_state=1
    )

    print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
    print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
    print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
    ax.set_title("Training, CV, Test", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(X_train, y_train, color="red", label="train")
    ax.scatter(X_cv, y_cv, color=dlc["dlorange"], label="cv")
    ax.scatter(X_test, y_test, color=dlc["dlblue"], label="test")
    ax.legend(loc="upper left")
    plt.show()

    max_degree = 9
    err_train: np.ndarray = np.zeros(max_degree)
    err_cv: np.ndarray = np.zeros(max_degree)
    x: np.ndarray = np.linspace(0, int(X.max()), 100)
    y_pred: np.ndarray = np.zeros((100, max_degree))  # columns are lines to plot

    for degree in range(max_degree):
        lmodel = utl.lin_model(degree + 1)
        lmodel.fit(X_train, y_train)
        yhat = lmodel.predict(X_train)
        err_train[degree] = lmodel.mse(y_train, yhat)
        yhat = lmodel.predict(X_cv)
        err_cv[degree] = lmodel.mse(y_cv, yhat)
        y_pred[:, degree] = lmodel.predict(x)

    optimal_degree = np.argmin(err_cv) + 1

    plt.close("all")
    utl.plt_optimal_degree(
        X_train,
        y_train,
        X_cv,
        y_cv,
        x,
        y_pred,
        x_ideal,
        y_ideal,
        err_train,
        err_cv,
        optimal_degree,
        max_degree,
    )

    lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    num_steps = len(lambda_range)
    degree = 10
    err_train = np.zeros(num_steps)
    err_cv = np.zeros(num_steps)
    x = np.linspace(0, int(X.max()), 100)
    y_pred = np.zeros((100, num_steps))  # columns are lines to plot

    for i in range(num_steps):
        lambda_ = lambda_range[i]
        lmodel = utl.lin_model(degree, regularization=True, lambda_=lambda_)
        lmodel.fit(X_train, y_train)
        yhat = lmodel.predict(X_train)
        err_train[i] = lmodel.mse(y_train, yhat)
        yhat = lmodel.predict(X_cv)
        err_cv[i] = lmodel.mse(y_cv, yhat)
        y_pred[:, i] = lmodel.predict(x)

    optimal_reg_idx = np.argmin(err_cv)

    plt.close("all")
    utl.plt_tune_regularization(
        X_train,
        y_train,
        X_cv,
        y_cv,
        x,
        y_pred,
        err_train,
        err_cv,
        optimal_reg_idx,
        lambda_range,
    )

    # Above, the plots show that as regularization increases, the model moves from a high variance (overfitting) model to a high bias (underfitting) model. The vertical line in the right plot shows the optimal value of lambda. In this example, the polynomial degree was set to 10.

    # ### 3.4 Getting more data: Increasing Training Set Size (m)
    # When a model is overfitting (high variance), collecting additional data can improve performance. Let's try that here.

    X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree = (
        utl.tune_m()
    )
    utl.plt_tune_m(
        X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree
    )

    # The above plots show that when a model has high variance and is overfitting, adding more examples improves performance. Note the curves on the left plot. The final curve with the highest value of $m$ is a smooth curve that is in the center of the data. On the right, as the number of examples increases, the performance of the training set and cross-validation set converge to similar values. Note that the curves are not as smooth as one might see in a lecture. That is to be expected. The trend remains clear: more data improves generalization.
    #
    # > Note that adding more examples when the model has high bias (underfitting) does not improve performance.
    #

    # ## 4 - Evaluating a Learning Algorithm (Neural Network)
    # Above, you tuned aspects of a polynomial regression model. Here, you will work with a neural network model. Let's start by creating a classification data set.

    # ### 4.1 Data Set
    # Run the cell below to generate a data set and split it into training, cross-validation (CV) and test sets. In this example, we're increasing the percentage of cross-validation data points for emphasis.

    # Generate and split data set
    X, y, centers, classes, std = utl.gen_blobs()

    # split the data. Large CV population for demonstration
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.50, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_, y_, test_size=0.20, random_state=1
    )
    print(
        "X_train.shape:",
        X_train.shape,
        "X_cv.shape:",
        X_cv.shape,
        "X_test.shape:",
        X_test.shape,
    )

    utl.plt_train_eq_dist(X_train, y_train, classes, X_cv, y_cv, centers, std)

    y_hat = np.array([1, 2, 0])
    y_tmp = np.array([1, 2, 3])
    print(
        f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333"
    )
    y_hat = np.array([[1], [2], [0], [3]])
    y_tmp = np.array([[1], [2], [1], [3]])
    print(
        f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250"
    )

    # BEGIN UNIT TEST
    print("-" * 80)
    print("\tGraded Test 2:")
    tst.test_eval_cat_err(eval_cat_err)
    print("-" * 80)
    # END UNIT TEST

    # %% Part 3
    print("=" * 80)
    print_str: str = "=== Part 3"
    print(print_str, "=" * (80 - len(print_str) - 1))
    print("=" * 80)

    # ### Exercise 3
    # Below, compose a three-layer model:
    # * Dense layer with 120 units, relu activation
    # * Dense layer with 40 units, relu activation
    # * Dense layer with 6 units and a linear activation (not softmax)
    # Compile using
    # * loss with `SparseCategoricalCrossentropy`, remember to use  `from_logits=True`
    # * Adam optimizer with learning rate of 0.01.

    model_fp: str = os.path.join(os.getcwd(), "models", "UNQ_C3.keras")

    if os.path.isfile(model_fp):
        model = keras.models.load_model(model_fp)
    else:
        tf.random.set_seed(1234)
        model = Sequential(
            [
                ### START CODE HERE ###
                Dense(120, activation="relu"),
                Dense(40, activation="relu"),
                Dense(6, activation="linear"),
                ### END CODE HERE ###
            ],
            name="Complex",
        )
        model.compile(
            ### START CODE HERE ###
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(0.01),
            ### END CODE HERE ###
        )
        # BEGIN UNIT TEST
        print("-" * 80)
        print("\tGraded Test 3:")
        model.fit(X_train, y_train, epochs=1000)
        model.save(model_fp)
        # END UNIT TEST

    # + deletable=false editable=false
    # BEGIN UNIT TEST
    model.summary()

    tst.model_test(model, classes, X_train.shape[1])
    print("-" * 80)
    # END UNIT TEST

    # make a model for plotting routines to call
    model_predict = lambda Xl: np.argmax(
        tf.nn.softmax(model.predict(Xl)).numpy(), axis=1
    )
    utl.plt_nn(
        model_predict, X_train, y_train, classes, X_cv, y_cv, suptitle="Complex Model"
    )

    # This model has worked very hard to capture outliers of each category. As a result, it has miscategorized some of the cross-validation data. Let's calculate the classification error.

    training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
    cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
    print(
        f"categorization error, training, complex model: {training_cerr_complex:0.3f}"
    )
    print(f"categorization error, cv, complex model: {cv_cerr_complex:0.3f}")

    # %% Part 4
    print("=" * 80)
    print_str: str = "=== Part 4"
    print(print_str, "=" * (80 - len(print_str) - 1))
    print("=" * 80)

    # ### Exercise 4
    #
    # Below, compose a two-layer model:
    # * Dense layer with 6 units, relu activation
    # * Dense layer with 6 units and a linear activation.
    # Compile using
    # * loss with `SparseCategoricalCrossentropy`, remember to use  `from_logits=True`
    # * Adam optimizer with learning rate of 0.01.

    # UNQ_C4
    # GRADED CELL: model_s
    print("-" * 80)
    print("\tGraded Test 4:")

    model_fp: str = os.path.join(os.getcwd(), "models", "UNQ_C4.keras")

    if os.path.isfile(model_fp):
        model_s = keras.models.load_model(model_fp)
    else:
        tf.random.set_seed(1234)
        model_s = Sequential(
            [
                ### START CODE HERE ###
                Dense(6, activation="relu"),
                Dense(6, activation="linear"),
                ### END CODE HERE ###
            ],
            name="Simple",
        )
        model_s.compile(
            ### START CODE HERE ###
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(0.01),
            ### START CODE HERE ###
        )

        # BEGIN UNIT TEST
        model_s.fit(X_train, y_train, epochs=1000)
        model_s.save(model_fp)
        # END UNIT TEST

    # BEGIN UNIT TEST
    model_s.summary()

    tst.model_s_test(model_s, classes, X_train.shape[1])
    print("-" * 80)
    # END UNIT TEST

    # make a model for plotting routines to call
    model_predict_s = lambda Xl: np.argmax(
        tf.nn.softmax(model_s.predict(Xl)).numpy(), axis=1
    )
    utl.plt_nn(
        model_predict_s, X_train, y_train, classes, X_cv, y_cv, suptitle="Simple Model"
    )

    # This simple models does pretty well. Let's calculate the classification error.

    training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
    cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
    print(
        f"categorization error, training, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}"
    )
    print(
        f"categorization error, cv, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}"
    )

    # %% Part 5
    print("=" * 80)
    print_str: str = "=== Part 5"
    print(print_str, "=" * (80 - len(print_str) - 1))
    print("=" * 80)

    # ### Exercise 5
    #
    # Reconstruct your complex model, but this time include regularization.
    # Below, compose a three-layer model:
    # * Dense layer with 120 units, relu activation, `kernel_regularizer=tf.keras.regularizers.l2(0.1)`
    # * Dense layer with 40 units, relu activation, `kernel_regularizer=tf.keras.regularizers.l2(0.1)`
    # * Dense layer with 6 units and a linear activation.
    # Compile using
    # * loss with `SparseCategoricalCrossentropy`, remember to use  `from_logits=True`
    # * Adam optimizer with learning rate of 0.01.

    # + deletable=false
    # UNQ_C5
    # GRADED CELL: model_r
    model_fp = os.path.join(os.getcwd(), "models", "UNQ_C5.keras")

    if os.path.isfile(model_fp):
        model_r = keras.models.load_model(model_fp)
    else:
        tf.random.set_seed(1234)
        model_r = Sequential(
            [
                ### START CODE HERE ###
                Dense(120, activation="relu", kernel_regularizer=l2(0.1)),
                Dense(40, activation="relu", kernel_regularizer=l2(0.1)),
                Dense(6, activation="linear"),
                ### START CODE HERE ###
            ],
            name="Complex_Regularized",
        )

        print("-" * 80)
        print("\tGraded Test 5:")
        model_r.compile(
            ### START CODE HERE ###
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(0.01),
            ### START CODE HERE ###
        )

        # + deletable=false editable=false
        # BEGIN UNIT TEST
        model_r.fit(X_train, y_train, epochs=1000)
        model_r.save(model_fp)
        # END UNIT TEST

    # + deletable=false editable=false
    # BEGIN UNIT TEST
    model_r.summary()

    tst.model_r_test(model_r, classes, X_train.shape[1])
    print("-" * 80)
    # END UNIT TEST

    model_predict_r = lambda Xl: np.argmax(
        tf.nn.softmax(model_r.predict(Xl)).numpy(), axis=1
    )

    utl.plt_nn(
        model_predict_r, X_train, y_train, classes, X_cv, y_cv, suptitle="Regularized"
    )
    # -

    # The results look very similar to the 'ideal' model. Let's check classification error.

    # + deletable=false editable=false
    training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
    cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
    test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
    print(
        f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}"
    )
    print(
        f"categorization error, cv, regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}"
    )
    # -

    # The simple model is a bit better in the training set than the regularized model but worse in the cross validation set.

    # ## 7 - Iterate to find optimal regularization value
    # As you did in linear regression, you can try many regularization values. This code takes several minutes to run. If you have time, you can run it and check the results. If not, you have completed the graded parts of the assignment!

    tf.random.set_seed(1234)
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    models: list[keras.Model] = list()

    for i in range(len(lambdas)):
        lambda_ = lambdas[i]
        models.append(
            Sequential(
                [
                    Dense(120, activation="relu", kernel_regularizer=l2(lambda_)),
                    Dense(40, activation="relu", kernel_regularizer=l2(lambda_)),
                    Dense(classes, activation="linear"),
                ]
            )
        )
        models[i].compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(0.01),
        )

        models[i].fit(X_train, y_train, epochs=1000)
        print(f"Finished lambda = {lambda_}")

    utl.plot_iterate(lambdas, models, X_train, y_train, X_cv, y_cv)

    # As regularization is increased, the performance of the model on the training and cross-validation data sets converge. For this data set and model, lambda > 0.01 seems to be a reasonable choice.

    # ### 7.1 Test
    # Let's try our optimized models on the test set and compare them to 'ideal' performance.

    utl.plt_compare(X_test, y_test, classes, model_predict_s, model_predict_r, centers)


if __name__ == "__main__":
    main()
