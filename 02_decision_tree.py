import pandas as pd
import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()
    fileName = args.data
    return fileName


def prepare_data(fileName):
    # training_data
    attributes = pd.read_csv(fileName, header=None)
    # name the feature_attributes
    col_name = []
    for i in attributes:
        col_name.append("att" + str(i))
    attributes.columns = col_name
    # target attribute
    target = attributes.columns[-1]
    # unique labels in the dataset
    classes = attributes[target].unique()
    # total number of unique lables
    classes_count = len(classes)
    return attributes, target, classes, classes_count


def get_total_entropy(attributes):
    entropy_S = 0
    for _class in classes:
        count = len(attributes[attributes[target] == _class])
        entropy_S_temp = 0
        if count != 0:
            p_class = (count / len(attributes))
            entropy_S_temp = - (p_class) * math.log(p_class, classes_count)
        entropy_S += entropy_S_temp
    return entropy_S


def get_feature_entropy(feature_attribute):
    entropy_Sv = 0
    class_counts = feature_attribute[target].unique()
    for _class in class_counts:
        count = len(feature_attribute[feature_attribute[target] == _class])
        p_class = (count / len(feature_attribute))
        entropy_Sv += - (p_class) * math.log(p_class, classes_count)
    return entropy_Sv


def calculate_node_entropy(feature, value, attributes):
    feature_attribute = attributes[attributes[feature] == value]
    Sv = len(feature_attribute)
    entropy_Sv = get_feature_entropy(feature_attribute)
    return entropy_Sv, Sv


def calculate_IG(feature, attributes):
    total_entropy = get_total_entropy(attributes)
    features_values = attributes[feature].unique()
    feature_entropy_temp = 0
    for value in features_values:
        entropy_Sv, Sv = calculate_node_entropy(feature, value, attributes)
        S = len(attributes)
        feature_entropy_temp += (Sv / S) * entropy_Sv
    IG = total_entropy - feature_entropy_temp
    return IG


def next_root_node(attributes):
    max_IG = 0
    selected_feature = ''
    # remove the target column
    feature_attributes = attributes.drop(target, axis=1)
    for feature in feature_attributes.columns:
        feature_IG = calculate_IG(feature, attributes)
        if feature_IG > max_IG:
            max_IG = feature_IG
            selected_feature = feature
    return selected_feature


def display(depth, feature, attname, node_entropy, leaf_class):
    if not (depth):
        attr = feature
    else:
        attr = str(feature) + "=" + str(attname)
    print(str(depth) + "," + attr + "," + str(node_entropy) + "," + leaf_class)


def grow_tree(feature, attributes, depth):
    # get unique feature values and their counts
    feature_value_count_series = attributes[feature].value_counts(sort=False)
    tree = {}  # build a new sub tree
    for feature_value, feature_value_count in feature_value_count_series.items():
        feature_value_attributes = attributes[attributes[feature]
                                              == feature_value]
        entropy, _ = calculate_node_entropy(
            feature, feature_value, feature_value_attributes)
        hit_leaf_node = False
        for _class in classes:
            # slice the data based on this feature value which belongs this class
            class_with_feature_value = feature_value_attributes[
                feature_value_attributes[target] == _class]
            class_count_with_feature_value = len(class_with_feature_value)
            # pure class
            if class_count_with_feature_value == feature_value_count:
                hit_leaf_node = True
                tree[feature_value] = _class
                # update the attributes
                attributes = attributes[attributes[feature] != feature_value]
                display(depth, feature, feature_value, entropy, _class)
        if not hit_leaf_node:
            # to be grown
            tree[feature_value] = "no_leaf"
            display(depth, feature, feature_value, entropy, "no_leaf")
    return tree, attributes


def build_tree(depth, root, prev_node_value, attributes):
    if (len(attributes) != 0):
        # find the best split feature
        curr_node = next_root_node(attributes)
        tree, attributes = grow_tree(curr_node, attributes, depth)
        # increment the depth everytime the tree grows
        depth += 1
        next_branch = None
        # find the next branch details
        if prev_node_value != None:
            root[prev_node_value] = dict()
            root[prev_node_value][curr_node] = tree
            next_branch = root[prev_node_value][curr_node]
        else:
            # create the root node of the tree
            root[curr_node] = tree
            next_branch = root[curr_node]
        # iterate through the next branch value and label
        for node_value, leaf_class in list(next_branch.items()):
            if leaf_class == "no_leaf":
                # updated the attributes
                updated_attributes = attributes[attributes[curr_node]
                                                == node_value]
                # grow the tree until a leaf node is reached
                build_tree(depth, next_branch, node_value, updated_attributes)


def decision_tree(attributes):
    # inital entropy of the entire dataset
    entropy_S = get_total_entropy(attributes)
    display(0, "root", '', entropy_S, "no_leaf")
    build_tree(1, {}, None, attributes)


if __name__ == "__main__":
    global target, classes, classes_count
    fileName = parse_args()
    attributes, target, classes, classes_count = prepare_data(fileName)
    decision_tree(attributes)
