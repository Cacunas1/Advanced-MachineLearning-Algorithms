import numpy as np
import utils as utl


def entropy(p: float) -> float:
    ans: float = 0.0
    ans = -p * np.log2(p) - (1 - p) * np.log2(1 - p) if p not in {0., 1.} else ans

    return ans


def split_indices(X: np.ndarray, index_feature: int) -> tuple[list, list]:
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have
    that feature = 1 and the right node those that have the feature = 0
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices: list = list()
    right_indices: list = list()
    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices


def weighted_entropy(X: np.ndarray, y: np.ndarray, left_indices: list, right_indices: list) -> float:
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left: float = len(left_indices)/len(X)
    w_right: float = len(right_indices)/len(X)
    p_left: float = sum(y[left_indices])/len(left_indices)
    p_right: float = sum(y[right_indices])/len(right_indices)
    
    weighted_entropy: float = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy

def information_gain(X: np.ndarray, y: np.ndarray, left_indices: list, right_indices: list) -> float:
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node: float = sum(y)/len(y)
    h_node: float = entropy(p_node)
    w_entropy: float = weighted_entropy(X, y, left_indices, right_indices)
    return h_node - w_entropy


def main():
    # print(np.version)

    print("=" * 80)
    print("# Course 2 - week 4 - Lab 1")
    print("-" * 80)
    print("## Intro")
    _ = utl.plot_entropy()


    X_train: np.ndarray = np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )

    print("Training set features:")
    print(X_train)
    print("Feature format: [Ear shape (Pointy=1, Floppy=0), Face shape (Round=1, ioc=0), Whiskers (Present=1, Absent=0)]")

    y_train: np.ndarray = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    print("Training set labels:")
    print(y_train)
    print("Cat class=1, Not cat class=0")

    print("-" * 80)
    print("## Entropy, Information Gain")
    print(f"Entropy of 0.5 = {entropy(0.5)}")

    li, ri = split_indices(X_train, 0)
    print("Splitting on feature 0 (ear shape)")
    print(f"Left indices: {li}")
    print(f"Right indices: {ri}")

    we: float = weighted_entropy(X_train, y_train, li, ri)
    print(f"Weighted entropy for <ear_shape> split: {we}")

    i_gain: float = information_gain(X_train, y_train, li, ri)
    print(f"Information gain from <ear_shape> split: {i_gain}")

    for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
        left_indices, right_indices = split_indices(X_train, i)
        i_gain = information_gain(X_train, y_train, left_indices, right_indices)
        print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")

    tree: list = list()
    utl.build_tree_recursive(
        X_train,
        y_train,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Root",
        max_depth=1,
        current_depth=0,
        tree=tree,
    )
    utl.generate_tree_viz([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y_train, tree)

    tree = list()
    utl.build_tree_recursive(
        X_train,
        y_train,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Root",
        max_depth=2,
        current_depth=0,
        tree=tree,
    )
    utl.generate_tree_viz([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y_train, tree)


    print("=" * 80)


if __name__ == "__main__":
    main()
