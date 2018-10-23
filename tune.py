from randomForest import *

def main(argv):
    tree_output(argv)
            
def tree_output(argv):
    tree_tune = tune_tree(argv)
    f = open("report_tree.csv","w+")
    for (accuracy, num_attributes, num_data_points, num_trees) in tree_tune:
        f.write("%f,%d,%d,%d\n" %(accuracy, num_attributes, num_data_points, num_trees))
    f.close()
            
def tune_tree(argv):
    dataset = argv[1]
    #num_attributes = int(argv[2]) 1-> numattributes
    #num_data_points = int(argv[3]) 10 -> half of dataset. increments of 1
    #num_trees = int(argv[4])

    tree_tune = []
    data = pd.read_csv(dataset)
    for num_attributes in range(1, len(data.columns)-1):
        for num_data_points in range(10, int(data.shape[0]/5)+1):
            for num_trees in range(5, 15):
                accuracy = cross_fold_validation(data, dataset, num_attributes, num_data_points, num_trees)
                print("acc: {} num att: {} num points: {} num trees: {}".format(accuracy, num_attributes, num_data_points, num_trees))
                tree_tune.append((accuracy, num_attributes, num_data_points, num_trees))

    tree_tune = sorted(tree_tune, key=lambda x:x[0], reverse=True)
    print("tree_tune results {}".format(tree_tune))
    return tree_tune


if __name__ == "__main__":
    main(sys.argv)
