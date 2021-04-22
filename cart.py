"""
A simple script which uses sklearn libraries to classify data from a given set
using RandomForest algorithm.

Run with required arguments only using this command:
"python cart.py -t b_depressed.csv -c depressed"

Should take about two seconds to output .pngs in the project directory.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def split_value(split_x):
    """
    Ensures that the argument --train_test_split falls between 0.2 and 0.8.
    :param split_x: the argument that's been entered by a user
    :return: throws an exception if the argument is not between 0.2 and 0.8
    """
    split_x = float(split_x)
    if split_x > 0.8 or split_x < 0.2:
        raise argparse.ArgumentTypeError("Max split is between 0.2 and 0.8")


def parse_arguments():
    """
    Parses command line arguments.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train_file',
                        type=str,
                        help='.csv file containing a dataset',
                        required=True)
    parser.add_argument('-s',
                        '--train_test_split',
                        type=split_value,
                        help='Determines how many rows will be used for tests. Between 0.2 and '
                             '0.8. 0.2 by default',
                        default=0.2,
                        )
    parser.add_argument('-c',
                        '--classification_column',
                        required=True,
                        type=str,
                        help='Name of a column in a dataset that contains labels / classes.')
    parser.add_argument('--max_depth',
                        type=int,
                        default=5,
                        help="Determines maximum depth of a tree. 5 by default")
    parser.add_argument('--acceptable_impurity',
                        type=float,
                        default=0.0,
                        help='Level of impurity at which nodes are no longer split. 0 by default')
    return parser.parse_args()


def prepare_data(data_file_name, split, classification_column='depressed'):
    """
    Drops unnecessary columns and splits that into training and testing data.
    :param classification_column: label of the column containing classes
    :param data_file_name: name of a data file as string
    :param split: float value that determines how many rows of data will be used for tests
    (default && min 0.2, max 0.8)
    :return: data split into X_train, X_test, Y_train, Y_test ready for further manipulation
    """
    d_f = pd.read_csv(data_file_name)
    d_f = d_f.drop('Survey_id', axis='columns').drop('Ville_id', axis='columns')

    x_set = d_f.loc[:, d_f.columns != classification_column].values
    x_set = np.nan_to_num(x_set.astype(np.float32))
    y_set = d_f.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=split)
    return x_train, x_test, y_train, y_test, d_f


def build_single_tree(max_depth, train_x, train_y, d_frame):
    """
    :param d_frame: dataframe of the original s
    :param max_depth: max depth of the output tree
    :param train_x: training features
    :param train_y: training labels
    :output: single tree visualised in the project directory.
    """
    decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    decision_tree.fit(train_x, train_y)

    plt.figure(figsize=(30, 25))
    print(tree.plot_tree(decision_tree, filled=True, max_depth=max_depth,
                         class_names=['healthy', 'depressed'],
                         feature_names=d_frame.loc[:, d_frame.columns != 'depressed'].columns))
    plt.savefig('single-tree-visualisation.png')


def build_forest(train_x, train_y, d_frame, test_x, test_y,
                 max_depth=5,
                 acceptable_impurity=0.0,
                 classification_column='depressed'):
    """
    :param test_y: dataset labels (classes)
    :param test_x: dataset features
    :param d_frame: dataframe of the original set
    :param classification_column: label of classification column, default is 'depressed'
    :param acceptable_impurity: determines at which point data is now longer split
    :param max_depth: max depth of a single output tree
    :param train_x: training features
    :param train_y: training labels
    :output: multiple tree visualisations (as .png) in the project directory
    """
    random_forest = RandomForestClassifier(max_depth=max_depth, n_estimators=9, bootstrap=True,
                                           min_impurity_decrease=acceptable_impurity)
    random_forest.fit(train_x, train_y)

    for tree_id, d_tree in enumerate(random_forest.estimators_):
        plt.figure(figsize=(30, 25))
        tree.plot_tree(d_tree, filled=True, max_depth=5, class_names=['healthy', 'depressed'],
                       feature_names=d_frame.loc[:, d_frame.columns != classification_column]
                       .columns)
        plt.savefig(f'decision-tree-{tree_id}')
    predict(random_forest, test_x, test_y)


def predict(classifier, test_x, test_y):
    """
    :param classifier: e.g. RandomForest from sklearn
    :param test_x: test feature set
    :param test_y: test label set (classes)
    :return: None. Output accuracy info to txt.
    """
    correct = 0
    predictions = classifier.predict(test_x)
    for idx, prediction in enumerate(predictions):
        if prediction == test_y[idx]:
            correct += 1
    accuracy = correct/len(predictions)
    output_file = open("accuracy.txt", "w")
    output_file.write(f"Accuracy level: {accuracy}")


def main(args):
    """
    :param args: command line arguments
    :return: None
    """
    args = parse_arguments()
    x_training, x_testing, y_training, y_testing, data_frame = prepare_data(args.train_file,
                                                                            args.train_test_split)
    print(x_testing, y_testing)
    build_forest(train_x=x_training, train_y=y_training,
                 acceptable_impurity=args.acceptable_impurity,
                 classification_column=args.classification_column,
                 max_depth=args.max_depth, d_frame=data_frame,
                 test_x=x_testing, test_y=y_testing)


if __name__ == '__main__':
    main(parse_arguments())
