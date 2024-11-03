import matplotlib.pyplot as plt
import numpy as np

import public_tests as tst


def compute_entropy(y: np.ndarray) -> float:
    """
    Computes the entropy for

    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)

    Returns:
        entropy (float): Entropy at that node

    """
    # You need to return the following variables correctly
    entropy: float = 0.0
    m: int = len(y)

    ### START CODE HERE ###
    if m == 0:
        return entropy

    p1: float = len(y[y == 1]) / m
    p0: float = 1.0 - p1
    e1: float = p1 * np.log2(p1) if p1 > 0.0 else 0.0
    e0: float = p0 * np.log2(p0) if p0 > 0.0 else 0.0
    entropy = - (e1 + e0) if e1 not in {0.0, 1.0} else 0.0
    ### END CODE HERE ###

    return entropy


# ### Exercise 2
#
# Please complete the `split_dataset()` function shown below
#
# - For each index in `node_indices`
#     - If the value of `X` at that index for that feature is `1`, add the index to `left_indices`
#     - If the value of `X` at that index for that feature is `0`, add the index to `right_indices`
#
# If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.
# UNQ_C2
# GRADED FUNCTION: split_dataset
def split_dataset(X: np.ndarray, node_indices: list[int], feature: int) -> tuple[list, list]:
    """
    Splits the data at the given node into
    left and right branches

    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on

    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """

    # You need to return the following variables correctly
    left_indices: list = list()
    right_indices: list = list()

    ### START CODE HERE ###
    # include rows or data points in the dataset that the feature is class "1"
    left_indices = [i for i in node_indices if X[i][feature] == 1]
    # include rows or data points in the dataset that the feature is class "0"
    right_indices = [i for i in node_indices if X[i][feature] == 0]
    ### END CODE HERE ###

    return left_indices, right_indices


# UNQ_C3
# GRADED FUNCTION: compute_information_gain

def compute_information_gain(X: np.ndarray, y: np.ndarray, node_indices: list[int], feature: int) -> float:
    """
    Compute the information of splitting the node on a given feature
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        feature (int):           Index of feature to split on
    Returns:
        cost (float):        Cost computed
    """    
    # Split dataset
    left_indices: list[int]
    right_indices: list[int]
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    y_node: np.ndarray
    y_left: np.ndarray
    y_right: np.ndarray
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]

    # You need to return the following variables correctly
    information_gain: float = 0.0

    ### START CODE HERE ###
    ml: int = len(left_indices)
    mr: int = len(right_indices)
    # w_left
    wl: float = ml / (ml + mr)
    # w_right
    wr: float = mr / (ml + mr)
    # H_node
    H_node: float = compute_entropy(y_node)
    # H_left
    H_left: float = compute_entropy(y_left)
    # H_right
    H_right: float = compute_entropy(y_right)

    information_gain = H_node - (wl * H_left + wr * H_right)
    ### END CODE HERE ###  

    return information_gain


# UNQ_C4
# GRADED FUNCTION: get_best_split


def get_best_split(X: np.ndarray, y: np.ndarray, node_indices: list[int]) -> int:
    """
    Returns the optimal feature and threshold value
    to split the node data

    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """
    features: int = X.shape[1]
    # You need to return the following variables correctly
    best_feature: int = -1

    ### START CODE HERE ###
    gains: list[float] = [compute_information_gain(X, y, node_indices, f) for f in range(features)]
    best: float = max(gains)
    best_feature = gains.index(best) if best > 0.0 else best_feature
    ### END CODE HERE ##

    return best_feature


def main():
    print(plt.__doc__)

    X_train: np.ndarray = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )
    y_train: np.ndarray = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])

    # #### View the variables
    # Let's get more familiar with your dataset.
    # - A good place to start is to just print out each variable and see what it contains.
    #
    # The code below prints the first few elements of `X_train` and the type of the variable.

    print("First few elements of X_train:\n", X_train[:5])
    print("Type of X_train:", type(X_train))

    # Now, let's do the same for `y_train`
    print("First few elements of y_train:", y_train[:5])
    print("Type of y_train:", type(y_train))

    # #### Check the dimensions of your variables
    # Another useful way to get familiar with your data is to view its dimensions.
    # Please print the shape of `X_train` and `y_train` and see how many training examples you have in your dataset.
    print("The shape of X_train is:", X_train.shape)
    print("The shape of y_train is: ", y_train.shape)
    print("Number of training examples (m):", len(X_train))
    # ## 4 - Decision Tree Refresher
    #
    # In this practice lab, you will build a decision tree based on the dataset provided.
    #
    # - Recall that the steps for building a decision tree are as follows:
    #     - Start with all examples at the root node
    #     - Calculate information gain for splitting on all possible features, and pick the one with the highest information gain
    #     - Split dataset according to the selected feature, and create left and right branches of the tree
    #     - Keep repeating splitting process until stopping criteria is met
    #
    #
    # - In this lab, you'll implement the following functions, which will let you split a node into left and right branches using the feature with the highest information gain
    #     - Calculate the entropy at a node
    #     - Split the dataset at a node into left and right branches based on a given feature
    #     - Calculate the information gain from splitting on a given feature
    #     - Choose the feature that maximizes information gain
    #
    # - We'll then use the helper functions you've implemented to build a decision tree by repeating the splitting process until the stopping criteria is met
    #     - For this lab, the stopping criteria we've chosen is setting a maximum depth of 2

    # ### 4.1  Calculate entropy
    #
    # First, you'll write a helper function called `compute_entropy` that computes the entropy (measure of impurity) at a node.
    # - The function takes in a numpy array (`y`) that indicates whether the examples in that node are edible (`1`) or poisonous(`0`)
    #
    # Complete the `compute_entropy()` function below to:
    # * Compute $p_1$, which is the fraction of examples that are edible (i.e. have value = `1` in `y`)
    # * The entropy is then calculated as
    #
    # $$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$
    # * Note
    #     * The log is calculated with base $2$
    #     * For implementation purposes, $0\text{log}_2(0) = 0$. That is, if `p_1 = 0` or `p_1 = 1`, set the entropy to `0`
    #     * Make sure to check that the data at a node is not empty (i.e. `len(y) != 0`). Return `0` if it is
    #
    # <a name="ex01"></a>
    # ### Exercise 1
    #
    # Please complete the `compute_entropy()` function using the previous instructions.
    #
    # If you get stuck, you can check out the hints presented after the cell below to help you with the implementation.

    # + deletable=false
    # UNQ_C1
    # GRADED FUNCTION: compute_entropy

    print("Entropy at root node: ", compute_entropy(y_train))

    # UNIT TESTS
    tst.compute_entropy_test(compute_entropy)

    root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Feel free to play around with these variables
    # The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
    feature = 0

    left_indices, right_indices = split_dataset(X_train, root_indices, feature)

    print("CASE 1:")
    print("Left indices: ", left_indices)
    print("Right indices: ", right_indices)

    # Visualize the split
    # utl.generate_split_viz(root_indices, left_indices, right_indices, feature)

    print()



if __name__ == "__main__":
    main()
