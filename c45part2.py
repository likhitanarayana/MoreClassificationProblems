#!/usr/bin/python3
import math
import itertools
import sys
import pandas as pd
from itertools import islice
from pandas import Series

import xml.etree.ElementTree as ET

xmlRoot = 'Tree name = \"test\"'
xmltree = ET.Element("tree", name="test")

def main(argv):
    final = callC45(argv)
    print(final)

def callC45(argv, data_section=None):
    data, attributes, opt_restrictions_file = parse(argv, data_section)

    # Reading and saving domain file
    #domain_tree = ET.parse(xmlfile)
    #root = domain_tree.getroot()
    #for child in root:
    #    counter = 0
    #    for grandchild in child:
    #        temp = str(grandchild.attrib['name'])
    #        domain_dictionary[temp] = counter
    #        counter += 1

    tree = Tree()
    c45(data, attributes, tree, 1.0, '')

    tree = tree.children[0][1]
    tree.makexml(xmltree, data)

    #ET.dump(xmltree)
    final_str = ET.tostring(xmltree, method='xml').decode("utf-8")

    return final_str


# splitOnVariable: variable name <Ex. Political Party>
# currentEdge: group name of the variable name <Ex. Democrat if splitOnVariable is Political Party>
# we might not need these last two values???
def c45(df, attributes, tree, threshold, currentEdge):
    print("\nc45 attributes = {}, df =".format(attributes))
    print(df)
    if df[df.columns[-1]].nunique() == 1:  # all votes are same value
        # Add leaf node
        print("1 found leaf!\n")
        print("df col: {}".format(df[[df.columns[-1]]]))
     
        vote_value = df[df.columns[-1]].unique()
        #print("vote_value = {}".format(vote_value))

        #print("vote_value[0] = {}".format(vote_value[0]))
        leafR = Tree(vote_value[0])
        #print("1. leafR = {}, currentEdge = {}".format(leafR, currentEdge))
        tree.add_children(leafR, currentEdge)
    elif len(attributes) == 0:
        print("2\n")
        label_c = find_most_frequent_label(df)
        leafR = Tree(label_c)
        #print("2. leafR = {}, currentEdge = {}".format(leafR, currentEdge))
        tree.add_children(leafR, currentEdge)
    else:
        print("in else")
        (best_attribute_g, best_split_x) = selectSplittingAttribute(attributes, df, threshold)
        print("in c45 best attribute to split on = {} with value {}".format(best_attribute_g, best_split_x))
        if best_attribute_g is None:
            print("3.1\n")
            label_d = find_most_frequent_label(df)
            leafR = Tree(label_d)
            #print("3. leafR = {}, currentEdge = {}".format(leafR, currentEdge))
            tree.add_children(leafR,currentEdge)
        else:
            #print("best_attribute_g = {}".format(best_attribute_g))
            print("#3.2\n")
            treeR = Tree(best_attribute_g)

            if isinstance(df[best_attribute_g].iloc[0], str):
                for attribute_value_v in df[best_attribute_g].unique():
                    data_frame_attribute_v = df[df[best_attribute_g] == attribute_value_v]
                    if len(data_frame_attribute_v) != 0:
                        print("bag: {}".format(best_attribute_g))
                        remove_current_attribute = [a for a in attributes if a != best_attribute_g]


                        data_frame_attribute_v = df[df[best_attribute_g] == attribute_value_v]
                        remove_current_attribute = [a for a in attributes if a != best_attribute_g]
                        tree_v = Tree()
                        c45(data_frame_attribute_v, remove_current_attribute, tree_v, threshold, attribute_value_v)
                            #print("4. tree_v = {}, attribute_value_v = {}".format(tree_v, best_attribute_g))
                            #print("4. tree_v = {}, attribute_value_v = {} and treeR = {}".format(tree_v, attribute_value_v, treeR))
                            #print("tree_v")
                            #tree_v.printT()
                        treeR.add_children(tree_v.children[0][1], attribute_value_v)
            else:
                remove_current_attribute = attributes
                tree_v = Tree()
                tree_l = Tree()
                tree_r = Tree()

                #IDK LOL HELP IDK I JUST DONT KNOW HELP ME
                data_frame_attribute_v = df[df[best_attribute_g] <= best_split_x]
                c45(data_frame_attribute_v, remove_current_attribute, tree_l, threshold, '<= {}'.format(best_attribute_g))
                data_frame_attribute_v = df[df[best_attribute_g] > best_split_x]
                c45(data_frame_attribute_v, remove_current_attribute, tree_r, threshold, '> {}'.format(best_attribute_g))
                treeR.add_children(tree_r.children[0][1], attribute_value_v)
                treeR.add_children(tree_l.children[0][1], attribute_value_v)

            tree.add_children(treeR, currentEdge)
    #return tree


def parse(argv, data_section=None):
    trainingSetFileCSV = argv[1]
    optRestrictionsFile = None

    if len(argv) > 2:
        optRestrictionsFile = argv[2]

    if data_section is not None:
        data = data_section
    else:
        data = pd.read_csv(trainingSetFileCSV, header=0, skiprows=[1, 2])

    attributes = []
    removeAttributes = []
    with open(trainingSetFileCSV) as f:
        for line in islice(f, 0, 1):
            attributes.extend(line.strip('\t\r\n\r').split(','))
        #for line in islice(f, 0, 1):
        #    removeAttributes.extend(line.strip('\t\n').split(','))

    optAttributes = []
    if optRestrictionsFile != None:
        optFile = open(optRestrictionsFile, 'r')
        for line in optFile:
            if (line.strip('\t\n') != ""):
                optAttributes.extend(line.strip('\t\n').split(','))
    removeAttributes = list((map(int, removeAttributes)))
    optAttributes = list(map(int, optAttributes))

    removeAttributesIndices = [i for i, v in enumerate(optAttributes) if v != 1]

    for i in range(len(removeAttributesIndices), -1, -1):
        attributes.pop(i)

    return data, attributes, optRestrictionsFile


def selectSplittingAttribute(attributes, df, threshold):
    # uses information gain
    p0 = enthropy(df)
    p = dict()
    gain = dict()
    bestsplit_x = None
    if len(attributes) == 1:
        return None
    for attribute in attributes[:-1]:
        #if attribute is continuous
        print("in selectSplittingAttribute = {}".format(attribute))
        if not isinstance(df[attribute].iloc[0], str):
            (bestsplit_x, one_gain) = find_best_split(df, attribute)
            print("called enthropy_split with attribute = {} and best split x= {}".format(attribute, bestsplit_x))
            p[attribute] = one_gain #enthropy_split(df, attribute, bestsplit_x)
            print("p[attribute] = {}".format(p[attribute]))
        else:
            # has no continuous attributes, only discrete
            bestsplit_x=None
            p[attribute] = enthropy_attribute(df, attribute)

        gain[attribute] = (p0 - p[attribute])

    print("gain = {}".format(gain))
    #gg = sorted(gain, key=lambda x:x[0], reverse=True)
    #print("gg = {}".format(gg))
    #best = gg[0][0]

    best = max(gain.keys(), key=(lambda key: gain[key][0]))
    print("best = {}".format(best))

    if gain[best][0] > threshold:
        print("returning in selectSplittingAttribute {},{}".format(best, gain[best][1]))
        return (best, gain[best][1])
    return None

def enthropy_split(df, attribute, split_x):
    split_x = float(split_x)
    print("\nin enthropy_split: attributes {}. splitx: {}".format(attribute, split_x))
    
    data_left = df[df[attribute] <= split_x]#df.loc[df[attribute] <= split_x]
    data_right = df[df[attribute] > split_x]#df.loc[df[attribute] > split_x]
    print("len of data left = {}, len of data right = {}".format(len(data_left), len(data_right)))
    left_calc = (-1.0*data_left.shape[0]/df.shape[0])*enthropy(data_left) 
    right_calc = (-1.0*data_right.shape[0]/df.shape[0])*enthropy(data_right) 
    result = left_calc + right_calc
    print("results = {}\n".format(float(result)))
    return float(result)

def find_best_split(df, attribute):
    #initialize associative arrays counts1[],...countsk[]
    p0 = float(enthropy(df))
    gain = []
    unique_col_values = df[attribute].unique()
    for col_val in unique_col_values:
        entropy = enthropy_split(df, attribute, col_val)
        gain.append((col_val, float(p0 - entropy)))

    #print("{}".format(gain))
    gain = sorted(gain, key=lambda x:x[1], reverse=True)
    print("gains {}\n".format(gain))
    return gain[0]

def enthropy(df):
    numTotal = len(df)
    att_entropy = 0;
    #totalObama = df[df[final_colname] == 'Obama']
    #probObama = len(totalObama) * 1.0 / numTotal
    #totalMcCain = numTotal - len(totalObama)
    #probMcCain = totalMcCain * 1.0 / numTotal
    #att_entropy = -probMcCain * math.log(probMcCain, 2) + (-probObama * math.log(probObama, 2))

    unique_col_values = df[df.columns[-1]].unique()
    for col_val in unique_col_values:
        numAppearance = len(df[df[df.columns[-1]] == col_val])
        probAppearance = numAppearance*1.0 / numTotal
        if probAppearance != 0:
            att_entropy += (-probAppearance*math.log(probAppearance,2))
    return att_entropy


def enthropy_attribute(df, attribute):
    numTotal = len(df)
    enthropy = 0
    unique_col_values = df[attribute].unique()
    for i in unique_col_values:
        total_attribute_i = df[df[attribute] == i]
        total_attribute_length = len(total_attribute_i)
        att_entropy = enthropy(total_attribute_i)
        #totalObama = total_attribute_i[total_attribute_i[final_colname] == 'Obama']
        #probObama = len(totalObama) * 1.0 / total_attribute_length
        #totalMcCain = total_attribute_length - len(totalObama)
        #probMcCain = totalMcCain * 1.0 / total_attribute_length
        #if probMcCain == 0:
        #    att_entropy = -(probObama * math.log(probObama, 2))
        #elif probObama == 0:
        #    att_entropy = -probMcCain * math.log(probMcCain, 2)
        #else:
        #    att_entropy = (-probMcCain * math.log(probMcCain, 2) - (probObama * math.log(probObama, 2)))
        enthropy += att_entropy * (total_attribute_length / len(df))
    return enthropy


def find_most_frequent_label(df):
    groupby_Votes = df.groupby(df.columns[-1]).size()
    max = Series.idxmax(groupby_Votes)
    return max


def iterparent(tree):
    for parent in tree.getiterator():
        for child in parent:
            yield parent, child


class Tree(object):
    def __init__(self, name=None):
        self.name = name
        self.children = []

    def __str__(self):
        if self.name is not None:
            return self.name
        else:
            return 'NONE'

    def add_children(self, other, edge=None):
        if self.name is None and type(other) is str:
            #assert(other, str)
            self.name = other
        else:
            #assert(other, Tree)
            self.children.append((edge, other))

    def printT(self, indents=0):
        for child in self.children:
            if True:#child[1].name is not None:
                print("  " * indents + "EDGE: {}, ATTRIBUTE: {}".format(child[0], child[1].name))#.name))
                indents +=1
            child[1].printT(indents)


    def returnprintT(self):
        for child in self.children:
            print("child name = {} and child children = {}".format(child[0], child[1].name))
            for grandchild in child[1].children:
                print("grandchild name = {} and grandchild children = {}".format(grandchild[0], grandchild[1].name))
                if grandchild[1].name is None:
                    print("make child = {} and {}".format(grandchild[1].children[0][0], grandchild[1].children[0][1]))
                    temp = grandchild[1].children[0]
                    grandchild = temp
                    print("child = {} and {}".format(child[0], child[1].name))
                    print("in else")
                    child[1].printT()
                    print("\n")
                print("about to recurse: child[1] = {} and {}".format(child[0], child[1]))
                grandchild[1].returnprintT()


    def makexml(self, root, df):
        unique_col_values = df[df.columns[-1]].unique()
        if (self.name in unique_col_values):
            
            dec_attr = {"choice": "{}".format(self.name)}
            b = ET.SubElement(root, "decision", attrib=dec_attr)
        else:
            node_attr = {"var": "{}".format(self.name)}
            b = ET.SubElement(root, "node", attrib=node_attr)

        for child in self.children:
            if child[1].name is not None:
                edge_attr = {"var": "{}".format(child[0])}
                c = ET.SubElement(b, "edge", attrib=edge_attr)
                child[1].makexml(c,df)



if __name__ == "__main__":
    main(sys.argv)


