# MoreClassificationProblems
Likhita Narayana
Jennifer Thai

To run the Random Forest classifier:
python3 randomForest.py dataset, num_attributes, num_data_points, num_trees
where dataset is the CSV file where your dataset is located, num_attributes is the number of attributes you want your trees to be constructed with, num_data_points is the number of data points to be used during the construction of the trees and num_trees is the total number of trees constructed.

To run KNN:
python3 KNN <dataset> <k>
where dataset is the file name of the CSV where your dataset is located and k is the number of neighbors KNN looks at when classifying your data.

To run C45:
python3 c45part2 <data> <attributes> <opt_restrictions_file>
where data is the CSV file where your dataset is located, attributes is the domainxml of the attributes and the optional opt restrictions file species which columns to not consider when creating our C4.5 decision tree. 

