# Note: this is just a template for homework 1 and the code is for references only.
# Feel free to design the pipeline of the *main* function. However, one should keep
# the interfaces for the other functions unchanged.

import csv
import numpy as np
import heapq
import time

def compute_accuracy(test_y, pred_y):
    print("\tCOMPUTING ACCURACY...")

    
    num_correct = 0.0
    for i in range(test_y.shape[0]):
        if(test_y[i] == pred_y[i]):
            num_correct += 1.0

    return num_correct / test_y.shape[0]

def test_knn(train_x, train_y, test_x, num_nn):
    print("\tTESTING KNN...")
    pred_y = []

    for count, test in enumerate(test_x, 1):
        if(count % 1000 == 0):
            print("\tCount [" + str(count) + "]")

        max_heap = []
        INDEX = 1
        VAL = 0
        for i, train in enumerate(train_x, 0):
            # compute euclidean distance between vects
            temp = (-1*np.linalg.norm(test-train), i)
            if(i < num_nn):
                # fill heap up to k items
                heapq.heappush(max_heap, temp)
            elif(max_heap[0] < temp):
                # remove the largest item each iteration
                heapq.heapreplace(max_heap, temp)

        indices = []
        # find k mininum indices
        for k in range(num_nn):
            indices.append(heapq.heappop(max_heap)[INDEX])

        # find the n minimum classes
        classes = []
        for index in indices:
            classes.append(train_y[index])

        # initialize array to count number of occurences of each class (26 classes)
        counts = [0]*26

        # get a count of each class
        for class_a in classes:
            counts[class_a] += counts[class_a] + 1

        # get index of max count
        max = np.array(counts).argsort()[-1]
    
        pred_y.append(max)
            
    return np.array(pred_y)

def test_pocket(w, test_x):

    # TO-DO: add your code here

    return None

def train_pocket(train_x, train_y, num_iters):

            
    return None

# return temple accessnet account
def get_id():
    return 'tug52339'

def run_knn(sample_size, num_nn, train_x, train_y, test_x, test_y):
    pred_y = test_knn(train_x, train_y, test_x, num_nn)

    acc = compute_accuracy(test_y, pred_y)
    
    print("\t\tSample Size [%d]\n\t\tNumber of Nearest Neighbors [%d]"
          % (sample_size, num_nn))

    print("\t\tAccuracy [" + str(acc) + "]")

def run_pocket(sample_size, train_x, train_y, test_x, test_y, num_iters):
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

    # print tug52339
    print(get_id())

    # subsample sizes
    num_train = {100, 1000, 2000, 5000, 10000, 15000}

    # number of nearest neighbors for KNN
    num_nn = {1, 3, 5, 7, 9}
    
    # KNN EXP
    # Run through all 6 subsamples
    for exp_num, sample_size in enumerate(num_train, 1):
        start = time.time()
        print("\nKNN EXP [%d]" % exp_num)
        
        # Run through all 5 KNN versions
        for neighbors in num_nn:
            run_knn(sample_size, neighbors, trainX[:sample_size],trainY[:sample_size], testX, testY)
        print("TOTAL TIME " + str(time.time()-start))
            
    #num_iters = 500
    # Pocket EXP
    # Run through all 6 subsamples
    #for exp_num, num in enumerate(num_train, 1):
    #    print("\nPocket EXP [%d]" % exp_num)
    #    run_pocket(sample_size, trainX[:sample_size], trainY[:sample_size],
    #               testX, testY, num_iters)
    
    return None

if __name__ == "__main__":
    main()
