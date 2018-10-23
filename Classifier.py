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
    print("temp = {}".format(temp))
    xml_file = temp[1]
    temp.pop(0)
    data, attributes, opt_restrictions_file = parse(temp, data_section)
    print("in classifier data = ")
    print(data)
    print("\n")
    print("data type = {}".format(type(data)))
    print("data columns = {}".format(list(data)))
    #print("data = {}".format(data))
    # print("attributes = {}".format(attributes))
    # print("opt_restrictions_file = {}".format(opt_restrictions_file))
    # print("xml_file = {}".format(xml_file))
    total_classified = len(data)
    correctly_classified = 0
    counter = 0
    classification_results = pd.DataFrame(columns=['RowID', 'Actual', 'Predicted'])
    #print("classification df = {}".format(classification_results))
    for index, row in data.iterrows():
        print("row = {}".format(row))
        print("row type = {}".format(type(row)))
        #print("xml_file = {}".format(xml_file))
        tree = ET.parse(xml_file)
        root = tree.getroot()
        stopReading = False
        choice = ''
        find = ''
        final = None
        for child in root.iter():
            print("child tag = {}, child attrib = {}".format(child.tag, child.attrib))
            if 'var' in child.attrib and 'num' not in child.attrib and stopReading is False:
                #print("-----{}".format(child.attrib['var']))
                print("row choice = {}".format(row[child.attrib['var']]))
                choice = row[child.attrib['var']]
                stopReading = True
            if child.tag == 'edge' and stopReading is True and (child.attrib['var'] == choice or child.attrib['var'] == find):
                print("found next choice = {}".format(child.attrib))
                find = ''
                choice = ''
                stopReading = False
            elif child.tag == 'edge' and stopReading is True and find == '':
                if "greater_than" in child.attrib['var']:
                    split = child.attrib['var'].split()
                    val = split[-1]
                    #print("choice = {}".format(choice))
                    if choice > float(val):
                        #print("found next choice = {}".format(child.attrib))
                        stopReading = False
                    else:
                        find = ''
                        #print("val = {}".format(val))
                        find = split[0] + " less_than_or_equal_to " + val
                        stopReading = True

            #print("find = {} and choice = {}".format(find, choice))
            if child.tag == 'decision' and stopReading is False:
                final = child
                #print("final decision = {}".format(final.attrib))
                break
        #print("choice = '{}', type = {}".format(final.attrib['choice'], type(final.attrib['choice'])))
        #print("vote = '{}', type = {}".format(row['Vote'], type(row['Vote'])))
        print("final = {}".format(final))
        if final.attrib['choice'] == row[-1]:
            #print("correct classification")
            correctly_classified += 1
        classification_results.loc[counter] = [counter, row[-1], final.attrib['choice']]
        counter += 1

    incorrectly_classified = total_classified - correctly_classified
    precision = tp*1.0/(tp+fp)
    recall = tp*1.0/(tp+fn)
    f_measure = 2 * ((precision * recall) / (precision + recall))
    pf = fp*1.0/(fp+tn)
    accuracy = correctly_classified*1.0/total_classified

    print("final dataframe = {}".format(classification_results))
    print("correctly classified = {}".format(correctly_classified))
    print("incorrectly classified = {}".format(incorrectly_classified))
    print("total classified = {}".format(total_classified))
    print("precision = {}".format(precision))
    print("recall = {}".format(recall))
    print("f_measure = {}".format(f_measure))

    return_values = dict()
    return_values['precision'] = precision
    return_values['recall'] = recall
    return_values['f_measure'] = f_measure
    return_values['num_classified'] = total_classified
    return_values['pf'] = pf
    return_values['correctly_classified'] = correctly_classified
    return_values['accuracy'] = accuracy
    return_values['predicted'] = final
    return return_values




if __name__ == "__main__":
    main(sys.argv)
