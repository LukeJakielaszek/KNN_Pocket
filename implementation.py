# Note: this is just a template for homework 1 and the code is for references only.
# Feel free to design the pipeline of the *main* function. However, one should keep
# the interfaces for the other functions unchanged.

import csv
import numpy as np

def compute_accuracy(test_y, pred_y):

    # TO-DO: add your code here

    return None

def test_knn(train_x, train_y, test_x, num_nn):

    # TO-DO: add your code here

    return None

def test_pocket(w, test_x):

    # TO-DO: add your code here

    return None

def train_pocket(train_x, train_y, num_iters):

    # TO-DO: add your code here

    return None

# return temple accessnet account
def get_id():
    return 'tug52339'

def run_knn(sample_size, num_nn, train_x, train_y, test_x, test_y):
    print("Sample Size [%d]\nNumber of Nearest Neighbors [%d]"
          % (sample_size, num_nn))

    pred_y = test_knn(train_x, train_y, test_x, num_nn)
    acc = compute_accuracy(test_y, pred_y)
    
    return None

def run_pocket(sample_size, train_x, train_y, test_x, test_y):
    print("Sample Size [%d]"
          % (sample_size))
    w = train_pocket(train_x, train_y, num_iters)
    pred_y = test_pocket(w, test_x)
    acc = compute_accuracy(test_y, pred_y)
    
    return None


def main():

    # Read the data file
    # Put this file in the same place as this script
    szDatasetPath = './letter-recognition.data' 
    listClasses = []
    listAttrs = []
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            listClasses.append(row[0])
            listAttrs.append(list(map(float, row[1:])))

    # Generate the mapping from class name to integer IDs
    mapCls2Int = dict([(y, x) for x, y in enumerate(sorted(set(listClasses)))])

    # Store the dataset with numpy array
    dataX = np.array(listAttrs)
    dataY = np.array([mapCls2Int[cls] for cls in listClasses])

    # Split the dataset as the training set and test set
    nNumTrainingExamples = 15000
    trainX = dataX[:nNumTrainingExamples, :]
    trainY = dataY[:nNumTrainingExamples]
    testX = dataX[nNumTrainingExamples:, :]
    testY = dataY[nNumTrainingExamples:]

    # TO-DO: add your code here
    print(get_id())

    # subsample sizes
    num_train = {100, 1000, 2000, 5000, 10000, 15000}

    # number of nearest neighbors for KNN
    num_nn = {1, 3, 5, 7, 9}
    
    # KNN EXP
    # Run through all 6 subsamples
    for exp_num, sample_size in enumerate(num_train, 1):
        print("\nKNN EXP [%d]" % exp_num)
        # Run through all 5 KNN versions
        for neighbors in num_nn:
            run_knn(sample_size, neighbors, trainX, trainY, testX, testY)

    # Pocket EXP
    # Run through all 6 subsamples
    for exp_num, num in enumerate(num_train, 1):
        print("\nPocket EXP [%d]" % exp_num)
        run_pocket(sample_size, trainX, trainY, testX, testY)
    
    return None

if __name__ == "__main__":
    main()
