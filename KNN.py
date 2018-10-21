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
        #print("row = ")
        #print(row)
        #calculate the distance between d and row
        attribute_length = len(row)-2
        for attribute in range(attribute_length):
            distance_to_dprime = 0
            #print("attribute = ")
            #print(attribute)
            distance_to_dprime += np.square(d[attribute] - row[attribute])
        sqrt_distance = np.sqrt(distance_to_dprime)
        distances.append((distance_to_dprime, row[len(row)-1]))

    sort_distances = sorted(distances, key=lambda x:x[0])
    neigbors_classification = []

    for x in range(k):
        neigbors_classification.append(sort_distances[x][1])

    print("neighbors = {}".format(neigbors_classification))

    counter = Counter(neigbors_classification)
    print("classify = {}".format(counter.most_common(1)[0][0]))
    return counter.most_common(1)[0][0]



def main(argv):
    training_dataset = argv[1]
    to_be_classified = argv[2]
    k = int(argv[3])

    training_dataset = pd.read_csv(training_dataset, header=None)
    #print("dataset =")
    #print(training_dataset)
    classify = training_dataset.iloc[0]
    KNN(training_dataset, classify, k)


if __name__ == "__main__":
    main(sys.argv)
