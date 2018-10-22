import pandas as pd
import numpy as np
import math
import sys
from InduceC45 import *


def random_forests(dataset, num_attributes, num_data_points, num_trees):
    attributes = list(dataset)
    attributes.pop()
    file_name = "file_"
    tree_files = []
    for tree_i in num_trees:
        print("tree_i = {}".format(tree_i))
        sample_dataset = dataset.sample(n=num_data_points)
        attributes.shuffle()
        sample_attributes = attributes[0:num_attributes]
        section = dataset[sample_attributes]
        print("section = ")
        print(section)
        # call c45
        tree_string = callC45(["temp", trainingcsv],section)
        file_name_i = file_name + tree_i
        text_file.open(file_name_i, "w")
        text_file.write(tree_string)
        text_file.close()



        # store c45 in xml file and store file name in list
    # return xml file list


# need to make classify function that takes in a list of xml files to classify a new data point

# need to make 10 fold cv



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


if __name__ == "__main__":
    main(sys.argv)