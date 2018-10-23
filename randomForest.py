import pandas as pd
import numpy as np
import math
import sys
from InduceC45 import *
from Classifier import *
import random


def random_forest_generation(df, num_attributes, num_data_points, num_trees, dataset):
    attributes = list(df)
    result_column = attributes.pop()
    print("all original attributes = {}".format(attributes))
    print("result column = {}".format(result_column))
    file_name = "file_"
    tree_files = []
    for tree_i in range(0,num_trees):
        print("tree_i = {}".format(tree_i))
        sample_dataset = df.sample(n=num_data_points)
        random.shuffle(attributes)
        sample_attributes = attributes[0:num_attributes]
        sample_attributes.append(result_column)
        section = df[sample_attributes]
        print("section = ")
        print(section)
        # call c45
        tree_string = callC45(["temp", dataset],section, sample_attributes)
        file_name_i = file_name + str(tree_i)
        text_file = open(file_name_i, "w")
        text_file.write(tree_string)
        text_file.close()
        tree_files.append(file_name_i)

    return tree_files


        # store c45 in xml file and store file name in list
    # return xml file list


# need to make classify function that takes in a list of xml files to classify a new data point

# need to make 10 fold cv
def random_forest_classify(trees, data, dataset):
    counter = 0
    for counter in range(0, len(data)):
        predicted = []
        classification_results = pd.DataFrame(columns=['RowID', 'Actual', 'Predicted'])
        row = pd.DataFrame(data.iloc[:, counter])
        print("trying to classify row = {}".format(row))
        for tree in trees:
            predicted.append(classifier(['temp', tree, dataset], row)['predicted'])
            print("predicted = {}".format(predicted))
        c = Counter(predicted)
        final = c.most_common(1)
        print("final class = {}".format(final))
        classification_results.loc[counter] = [counter, row[-1], final]
    print("prediction = ")
    print(classification_results)


def main(argv):
    dataset = argv[1]
    num_attributes = int(argv[2])
    num_data_points = int(argv[3])
    num_trees = int(argv[4])

    print("data")
    print(dataset)
    print("num_attributes = {}".format(num_attributes))
    print("num_data_points = {}".format(num_data_points))
    print("num_trees = {}".format(num_trees))
    data = pd.read_csv(dataset)
    files = random_forest_generation(data, num_attributes, num_data_points, num_trees, dataset)
    random_forest_classify(files, data, dataset)


if __name__ == "__main__":
    main(sys.argv)