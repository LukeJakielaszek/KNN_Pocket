# Note: this is just a template for homework 1 and the code is for references only.
# Feel free to design the pipeline of the *main* function. However, one should keep
# the interfaces for the other functions unchanged.

import csv
import numpy as np
import threading

def compute_accuracy(test_y, pred_y):
    print("COMPUTING ACCURACY...")

    num_correct = 0.0
    for i in range(test_y.shape[0]):
        if(test_y[i] == pred_y[i]):
            num_correct += 1.0

    return num_correct / test_y.shape[0]

def test_knn(train_x, train_y, test_x, num_nn):
    print("TESTING KNN...")
    
    pred_y = []

    for count, test in enumerate(test_x, 1):
        if(count % 250 == 0):
            print("Count [" + str(count) + "]")
        temp = []

        for train in train_x:
            # compute euclidean distance between vects
            temp.append(np.linalg.norm(test-train))

        # convert distances to numpy array
        arr = np.array(temp)

        # find k mininum indices
        indices = arr.argsort()[:num_nn]

        # find the n minimum classes
        classes = []
        for index in indices:
            classes.append(train_y[index])

        # initialize array to count number of occurences of each class (26 classes)
        counts = [0]*26

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
    
    print("Sample Size [%d]\nNumber of Nearest Neighbors [%d]"
          % (sample_size, num_nn))

    print(acc)

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
        print("\nKNN EXP [%d]" % exp_num)
        
        # keep reference to the threads
        threads = []
        
        # Run through all 5 KNN versions
        for thread_index, neighbors in enumerate(num_nn, 0):
            # send required material to threads
            threads.append(threading.Thread(target=run_knn,
                                            args=(sample_size, neighbors, trainX[:sample_size],
                                                  trainY[:sample_size], testX, testY)))
            # start each thread
            threads[thread_index].start()

        # wait for all threads to complete
        for thread in threads:
            thread.join()
            
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
