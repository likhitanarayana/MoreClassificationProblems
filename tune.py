from randomForest import *
from KNN import *
from Validation import *

def main(argv):
    #tree_output(argv)
    #knn_output(argv)
    c45_output(argv)

def c45_output(argv):
    c45_result = tune_c45(argv)
    f = open("tuning_c45.csv","w+")
    f.write("accuracy,threshold\n")
    for (accuracy, threshold) in c45_result:
        f.write("%f,%f\n" %(accuracy, threshold))
    f.close()

def tune_c45(argv):
    dataset = argv[1]
    c45_result = []
    data = pd.read_csv(dataset)
    print("data {}".format(data))
    for threshold in range(10,90,10):
        xmlstring = callC45(["temp",argv[1]], threshold/100.0,data)
        f = open("xml_tree.xml", 'w')
        f.write(xmlstring)
        f.close()
        temp = ["temp", "xml_tree.xml", argv[1]]
        ret = classifier(temp, data)
        accuracy = ret['accuracy']
        print("acc: {} threshold: {}".format(accuracy, threshold/100.0))
        c45_result.append((accuracy, threshold/100.0))

    c45_result = sorted(c45_result, key=lambda x:x[0], reverse=True)
    print("c45_tune results {}".format(c45_result))
    return c45_result

def knn_output(argv):
    knn_result = tune_knn(argv)
    f = open("tuning_knn.csv","w+")
    f.write("accuracy,k\n")
    for (accuracy, k) in knn_result:
        f.write("%f,%d\n" %(accuracy, k))
    f.close()
      

def tune_knn(argv):
    dataset = argv[1]
    knn_result = []
    data = pd.read_csv(dataset)
    for k in range(1, len(dataset)):
        accuracy = classify_whole_dataset(data, k)
        print("acc: {} k: {}".format(accuracy, k))
        knn_result.append((accuracy, k))

    knn_result = sorted(knn_result, key=lambda x:x[0], reverse=True)
    print("knn_tune results {}".format(knn_result))
    return knn_result

def tree_output(argv):
    tree_result = tune_tree(argv)
    f = open("tuning_forest.csv","w+")
    f.write("accuracy,num_attributes,num_data_points,num_trees\n")
    for (accuracy, num_attributes, num_data_points, num_trees) in tree_result:
        f.write("%f,%d,%d,%d\n" %(accuracy, num_attributes, num_data_points, num_trees))
    f.close()
            
def tune_tree(argv):
    dataset = argv[1]
    tree_tune = []
    data = pd.read_csv(dataset)
    print("num data points = {}".format(int(data.shape[0]/2)+1))
    print("num attributes = {}".format(len(data.columns)-1))
    for num_attributes in range(1, len(data.columns)):
        for num_data_points in range(20, int(data.shape[0]/2)+1, 25):
            for num_trees in range(5, 15, 5):
                accuracy = cross_fold_validation(data, dataset, num_attributes, num_data_points, num_trees)
                print("acc: {} num att: {} num points: {} num trees: {}".format(accuracy, num_attributes, num_data_points, num_trees))
                tree_tune.append((accuracy, num_attributes, num_data_points, num_trees))

    tree_tune = sorted(tree_tune, key=lambda x:x[0], reverse=True)
    print("tree_tune results {}".format(tree_tune))
    return tree_tune


if __name__ == "__main__":
    main(sys.argv)
