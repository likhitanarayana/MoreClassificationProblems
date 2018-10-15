#!/usr/bin/python3
import math
import itertools
import sys
import pandas as pd
from itertools import islice
from pandas import Series
import xml.etree.ElementTree as ET
from InduceC45 import *


def main(argv):
    temp = argv
    classifier(temp)

def classifier(temp, data_section=None):
    xml_file = temp[1]
    temp.pop(0)
    data, attributes, opt_restrictions_file, domain_file_xml = parse(temp, data_section)
    #print("data = {}".format(data))
    # print("attributes = {}".format(attributes))
    # print("opt_restrictions_file = {}".format(opt_restrictions_file))
    # print("xml_file = {}".format(xml_file))
    total_classified = len(data)
    correctly_classified = 0
    tp = 0 # obama, obama
    fn = 0 # obama, mccain
    fp = 0 # mccain, mccain
    tn = 0 # mccain, obama
    counter = 0
    classification_results = pd.DataFrame(columns=['RowID', 'Actual', 'Predicted'])
    #print("classification df = {}".format(classification_results))
    for index, row in data.iterrows():
        #print("row = {}".format(row))
        #print("xml_file = {}".format(xml_file))
        tree = ET.parse(xml_file)
        root = tree.getroot()
        stopReading = False
        choice = ''
        final = None
        for child in root.iter():
            #print("child tag = {}, child attrib = {}".format(child.tag, child.attrib))
            if 'var' in child.attrib and 'num' not in child.attrib and stopReading is False:
                #print("row choice = {}".format(row[child.attrib['var']]))
                choice = row[child.attrib['var']]
                stopReading = True
            if child.tag == 'edge' and stopReading is True and child.attrib['var'] == choice:
                #print("found next choice = {}".format(child.attrib))
                stopReading = False
            if child.tag == 'decision' and stopReading is False:
                final = child
                #print("final decision = {}".format(final.attrib))
                break
        #print("choice = '{}', type = {}".format(final.attrib['choice'], type(final.attrib['choice'])))
        #print("vote = '{}', type = {}".format(row['Vote'], type(row['Vote'])))
        if final.attrib['choice'] == row['Vote']:
            #print("correct classification")
            correctly_classified += 1
            if final.attrib['choice'] == 'McCain':
                fp += 1
            else:
                tp += 1
        else:
            if row['Vote'] == 'Obama': #picked mccain, actual is obama
                fn += 1
            else:
                tn += 1
        classification_results.loc[counter] = [counter, row['Vote'], final.attrib['choice']]
        counter += 1

    incorrectly_classified = total_classified - correctly_classified
    precision = tp*1.0/(tp+fp)
    recall = tp*1.0/(tp+fn)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    pf = fp*1.0/(fp+tn)
    accuracy = correctly_classified*1.0/total_classified

    #print("final dataframe = {}".format(classification_results))
    #print("correctly classified = {}".format(correctly_classified))
    #print("incorrectly classified = {}".format(incorrectly_classified))
    #print("total classified = {}".format(total_classified))
    #print("precision = {}".format(precision))
    #print("recall = {}".format(recall))
    #print("f_measure = {}".format(f_measure))

    return_values = dict()
    return_values['fn'] = fn
    return_values['tn'] = tn
    return_values['fp'] = fp
    return_values['tp'] = tp
    return_values['precision'] = precision
    return_values['recall'] = recall
    return_values['f_measure'] = f_measure
    return_values['num_classified'] = total_classified
    return_values['pf'] = pf
    return_values['correctly_classified'] = correctly_classified
    return_values['accuracy'] = accuracy
    return return_values




if __name__ == "__main__":
    main(sys.argv)
