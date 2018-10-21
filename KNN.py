import pandas as pd
import numpy as np
import math
import sys
from collections import Counter

def KNN(D, d, k):
    # need to go through categorical variables and make them into binary one hot encoded variables?
    # how to do that?
    # assumption: last column is prediction value

    distances = []

    for row in D.itertuples():
        #calculate the distance between d and row
        attribute_length = len(row)-2
        for attribute in range(attribute_length):
            distance_to_dprime = 0
            distance_to_dprime += np.square(d[attribute] - row[attribute])
        sqrt_distance = np.sqrt(distance_to_dprime)
        distances.append((sqrt_distance, row[len(row)-1]))

    sort_distances = sorted(distances, key=lambda x:x[0])
    neighbors_classification = []

    for x in range(k):
        neighbors_classification.append(sort_distances[x][1])

    print("neighbors = {}".format(neighbors_classification))

    counter = Counter(neighbors_classification)
    print("classify = {}".format(counter.most_common(1)[0][0]))
    return counter.most_common(1)[0][0]


def one_hot_encode(dataset):
    cols = list(dataset)
    cols.pop()

    for col in cols:
        element = dataset[col].iloc[0]
        if isinstance(element, str):
            one_hot = pd.get_dummies(dataset[[col]])
            dataset = pd.merge(dataset, one_hot, left_index=True, right_index=True)
            dataset.drop([col])#, axis=1, inplace=True)

    return dataset


def main(argv):
    training_dataset = argv[1]
    k = int(argv[2])

    training_dataset = pd.read_csv(training_dataset, header=None)

    training_dataset = one_hot_encode(training_dataset)

    #print("dataset =")
    #print(training_dataset)
    classify = training_dataset.iloc[0]
    KNN(training_dataset, classify, k)


if __name__ == "__main__":
    main(sys.argv)
