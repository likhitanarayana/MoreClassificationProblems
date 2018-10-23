import pandas as pd
import numpy as np
import math
import sys
from collections import Counter

def KNN(D, d, k):
    # need to go through categorical variables and make them into binary one hot encoded variables?
    # how to do that?
    # assumption: last column is prediction value
    print("dataset = ")
    print(D)
    distances = []

    for row in D.itertuples():
        #calculate the distance between d and row
        attribute_length = len(row)-1
        for attribute in range(1,attribute_length):
            distance_to_dprime = 0
            distance_to_dprime += np.square(float(d[attribute]) - float(row[attribute+1]))
        sqrt_distance = np.sqrt(distance_to_dprime)
        distances.append((sqrt_distance, row[len(row)-1]))

    sort_distances = sorted(distances, key=lambda x:x[0])
    #sort_distances.reverse()
    #print("distances = {}".format(sort_distances))
    neighbors_classification = []

    for x in range(k):
        neighbors_classification.append(sort_distances[x][1])

    #print("neighbors = {}".format(neighbors_classification))

    counter = Counter(neighbors_classification)
    #print("classify = {}".format(counter.most_common(1)[0][0]))
    return counter.most_common(1)[0][0]


def one_hot_encode(dataset):
    cols = list(dataset)
    print("cols = {}".format(cols))
    cols.pop(0)

    for col in cols:
        print("col = {}".format(col))
        element = dataset[col].iloc[0]
        if isinstance(element, str):
            one_hot = pd.get_dummies(dataset[[col]])
            dataset = pd.merge(dataset, one_hot, left_index=True, right_index=True)
            print("dropping col = {}".format(col))
            dataset.drop([col], axis=1, inplace=True)
    cols = dataset.columns.tolist()
    cols = cols[:-1] + cols[-1:]
    dataset = dataset[cols]
    return dataset


def classify_whole_dataset(dataset, k):
    """

    :param dataset: pandas dataframe (beacause some have headers and others don't)
    :param k: num of centroids
    :return: accuracy
    """
    dataset = one_hot_encode(dataset)
    correct = 0
    for i in range(0, len(dataset)):
        classify = dataset.iloc[i]
        predicted = KNN(dataset, classify, k)
        #print("correct = {}".format(classify[-1]))
        if predicted == classify[-1]:
            correct += 1
    total_classified = len(dataset)
    accuracy = correct*1.0/total_classified
    print("correctly classified = {}".format(correct))
    print("overall accuracy = {}".format(accuracy))
    return accuracy


def main(argv):
    training_dataset = argv[1]
    k = int(argv[2])

    training_dataset = pd.read_csv(training_dataset, header=None)

    training_dataset = one_hot_encode(training_dataset)

    #print("dataset =")
    #print(training_dataset)
    classify = training_dataset.iloc[0]
    KNN(training_dataset, classify, k)
    #classify_whole_dataset(training_dataset, k)

if __name__ == "__main__":
    main(sys.argv)
