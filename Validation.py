#!/usr/bin/python3

from Classifier import *
from InduceC45 import *
import pandas as pd
import numpy as np

def main(argv):
    domainxml = argv[1]
    trainingcsv = argv[2]
    numfolds = int(argv[3])
    optrestrictionsfile = None

    if len(argv) > 4:
        optrestrictionsfile = argv[4]

    if numfolds == 0:
        print("no cross-val. use entire training set to construct a single classifier")

    elif numfolds == -1:
        print("all-but-one cross validation")

    else:
        print("{} fold cross val".format(numfolds))

    calcnums(numfolds, domainxml, trainingcsv, optrestrictionsfile)


def calcnums(numfolds, domainxml, trainingcsv, optrestrictionsfile):
    # True class A (TP) - correctly classified into class A
    # False class A (FP) - incorrectly classified into class A
    # True class B (TN) - correctly classified into class B
    # False class B (FN) - incorrectly classified into class B

    # compute these numbers w respect to recogniing obama's voters
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # pf(probability formula): probablity of appearing.  
    # f-measure = 2 * ((precision * recall)/(precision + recall))
    data = pd.read_csv(trainingcsv, header=0, skiprows=[1, 2])
    total_classified = len(data)

    if numfolds == -1:
        numfolds = len(data)

    split_df = np.array_split(data, numfolds)
    total_classified = len(data)


    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total_correct = 0
    average_accuracy = 0
    for section in split_df:
        # send section into InduceC45
        xmlstring = callC45(["temp",domainxml, trainingcsv, optrestrictionsfile],section)
        f = open("xml_tree.xml", 'w')
        f.write(xmlstring)
        f.close()

        temp = ["temp", "xml_tree.xml", domainxml, trainingcsv, optrestrictionsfile]
        ret = classifier(temp, section)
        #print("ret = {}".format(ret))
        tp += ret['tp']
        tn += ret['tn']
        fp += ret['fp']
        fn += ret['fn']
        total_correct += ret['correctly_classified']
        average_accuracy += ret['accuracy']

    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    pf = fp * 1.0 / (fp + tn)
    overall_accuracy = total_correct*1.0/total_classified
    average_accuracy = average_accuracy*1.0/numfolds
    errorRate = 1 - average_accuracy

    print("Given correctly prediction Obama is True Positive:")
    print("TP = {}".format(tp))
    print("TN = {}".format(tn))
    print("FP = {}".format(fp))
    print("FN = {}".format(fn))
    print("Overall Recall = {}".format(recall))
    print("Overall Precision = {}".format(precision))
    print("Overall F-Measure = {}".format(f_measure))
    print("PF = {}".format(pf))
    print("Overall Accuracy = {}".format(overall_accuracy))
    print("Average Accuracy = {}".format(average_accuracy))
    print("Error Rate of Prediction = {}".format(errorRate))


if __name__ == "__main__":
    main(sys.argv)
