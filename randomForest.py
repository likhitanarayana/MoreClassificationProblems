import pandas as pd
import numpy as np
import math
import sys
from InduceC45 import *
from Classifier import *
import random
import itertools
from collections import Counter


def random_forest_generation(df, num_attributes, num_data_points, num_trees, dataset):
    attributes = list(df)
    result_column = attributes.pop()
    #print("all original attributes = {}".format(attributes))
    #print("result column = {}".format(result_column))
    file_name = "file_"
    tree_files = []
    for tree_i in range(0,num_trees):
        print("tree_i = {}".format(tree_i))
        sample_dataset = df.sample(n=num_data_points)
        random.shuffle(attributes)
        sample_attributes = attributes[0:num_attributes]
        sample_attributes.append(result_column)
        section = df[sample_attributes]
        #print("section = ")
        #print(section)
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
    classification_results = pd.DataFrame(columns=['RowID', 'Actual', 'Predicted'])
    correctly_classified = 0
    for counter in range(0, len(data)):
        predicted = []
        row = data.iloc[[counter]]
        #print("trying to classify row = {}".format(row))
        for tree in trees:
            prediction = classifier(['temp', tree, dataset], row)['predicted']
            predicted.append(prediction)
            #print("prediction = {}".format(prediction))
        #print("all predictions = {}".format(predicted))
        c = Counter(predicted)
        final = c.most_common(1)[0][0]
        #print("final class = {}".format(final))
        row = data.iloc[counter]
        classification_results.loc[counter] = [counter, row[-1], final]
        if final == row[-1]:
            correctly_classified += 1

    df_confusion = pd.crosstab(classification_results['Actual'], classification_results['Predicted'])
    #print("df_confusion = {}".format(df_confusion))
    #print("prediction = ")
    #print(classification_results)
    #classification_results.to_csv("final_classification")
    #print("total classified = {}".format(len(data)))
    #print("correctly classified = {}".format(correctly_classified))
    #print("incorrectly classified = {}".format(len(data) - correctly_classified))
    #print("accuracy = {}".format(correctly_classified*1.0/len(data)))
    return_values = dict()
    return_values['correct'] = correctly_classified
    return_values['dataframe'] = classification_results
    return return_values


def cross_fold_validation(data, dataset, num_attributes, num_data_points, num_trees, num_folds=10):
    total_correctly_classified = pd.DataFrame(columns=['RowID', 'Actual', 'Predicted'])
    correct_classified = 0
    for i in range(0, num_folds):
        print("i = {}".format(i))
        split_df = np.array_split(data, num_folds)
        test_section = split_df[i]
        train_section = split_df
        del train_section[i]
        #print("train section = {}".format(train_section))
        train_section = pd.concat(train_section)
        #print("len of test section = {}".format(len(test_section)))
        #print("len of train section = {}".format(len(train_section)))
        files = random_forest_generation(train_section, num_attributes, num_data_points, num_trees, dataset)
        ret_values = random_forest_classify(files, test_section, dataset)
        correct_classified += ret_values['correct']
        total_correctly_classified = total_correctly_classified.append(ret_values['dataframe'])
    print("total classified = {}".format(len(data)))
    print("correctly classifed = {}".format(correct_classified))
    print("incorrectly classified = {}".format(len(data) - correct_classified))
    accuracy = correct_classified*1.0/len(data)
    print("overall accuracy = {}".format(accuracy))
    print("overall dataframe = ")
    print(total_correctly_classified)
    df_conf = pd.crosstab(total_correctly_classified['Actual'], total_correctly_classified['Predicted'])
    print(df_conf)

    df_conf_norm = df_conf / df_conf.sum(axis=1)
    print(df_conf_norm)

    return accuracy



def main(argv):
    dataset = argv[1]
    num_attributes = int(argv[2])
    num_data_points = int(argv[3])
    num_trees = int(argv[4])

    #print("data")
    #print(dataset)
    print("num_attributes = {}".format(num_attributes))
    print("num_data_points = {}".format(num_data_points))
    print("num_trees = {}".format(num_trees))
    data = pd.read_csv(dataset)
    #files = random_forest_generation(data, num_attributes, num_data_points, num_trees, dataset)
    #random_forest_classify(files, data, dataset)
    cross_fold_validation(data, dataset, num_attributes, num_data_points, num_trees)

if __name__ == "__main__":
    main(sys.argv)