import csv
import numpy as np
import heapq
import time
import random
from numpy.random import permutation

# computes accuracy by comparing predictions to known values
def compute_accuracy(test_y, pred_y):
    # count correctly identified classes
    num_correct = 0.0

    # loop through the dataset
    for i in range(test_y.shape[0]):
        # if a prediction was correct, increment the count
        if(test_y[i] == pred_y[i]):
            num_correct += 1.0

    # return the percent of correctly classified data
    return num_correct / test_y.shape[0]

# test a single KNN experiment
def test_knn(train_x, train_y, test_x, num_nn):
    print("\tTESTING KNN...")
    
    # stores the predicted classes
    pred_y = []

    # loop through the test set
    for count, test in enumerate(test_x, 1):
        # print our count every 1000 iterations
        if(count % 1000 == 0):
            print("\tTest Samples Processed [" + str(count) + "]")

        # heap to find K nearest neighbors
        max_heap = []

        # declare how we store information in the heap
        INDEX = 1
        VAL = 0

        # loop through the training set
        for i, train in enumerate(train_x, 0):
            # compute euclidean distance between vects
            # value is negative to transform python's minheap to a maxheap
            temp = (-1*np.linalg.norm(test-train), i)

            # initialize our max heap
            if(i < num_nn):
                # fill heap up to k items
                heapq.heappush(max_heap, temp)
            elif(max_heap[VAL] < temp):
                # remove the largest item after each iteration
                heapq.heapreplace(max_heap, temp)

        # stores the K nearest indices in order
        indices = []
        
        # find k mininum indices
        for k in range(num_nn):
            indices.append(heapq.heappop(max_heap))

        # find the n minimum classes
        classes = []
        for index in indices:
            # convert distances to positive for processing
            # find the neighbors actual classes
            classes.append([-1*index[VAL], train_y[index[INDEX]]])

        # initialize array to count number of occurences of each class (26 classes)
        counts = [0]*26

        # get a count of each class
        for class_a in classes:
            # increment the count
            counts[class_a[INDEX]] += counts[class_a[INDEX]] + 1

        # get index of max count (this corrresponds direct to the class since counts is
        # 26 indices long
        max_a = np.array(counts).argsort()[-1]

        # track the prediction
        pred_y.append(max_a)

    # return our predicted classes
    return np.array(pred_y)

# obtains predictions for the 26 pocket models and determines the most confident
def test_pocket(all_w, test_x):
    # used to store all processed predictions
    pred_y = np.zeros(test_x.shape[0], dtype=int)

    # run through our test set
    for j, sample in enumerate(test_x, 0):
        # store each OVA prediction
        predictions = np.empty(26)

        # run through all OVA models
        for i in range(26):
            # predict the OVA class of our training sample
            predictions[i] = np.dot(all_w[i][1:], test_x[j]) + all_w[i][0]

        # get index of most confident OVA prediction
        pred_y[j] = predictions.argsort()[-1]

    # return our processed predictions
    return pred_y

# trains all 26 OVA pocket models
def train_pocket(train_x, train_y, num_iters):
    # current model weights
    all_w = np.empty(shape=(26,train_x.shape[1] + 1))

    # previous model weights
    prev_w = np.copy(all_w)

    # The saved accuracy of our model
    prev_acc = 0
    
    # tracks our confirmed changes
    cur_iter = 1
    while(cur_iter < num_iters):        
        # initialize model weights
        for i in range(26):
            if(cur_iter > num_iters):
                break

            # vector to store labels in a OVA format
            ova_labels = np.empty(shape=(train_y.shape[0],))
            
            # process our label vector into OVA format for each class
            for index, class_a in enumerate(train_y, 0):
                if class_a == i:
                    ova_labels[index] = 1
                else:
                    ova_labels[index] = -1

            # to improve speed of convergence and avoid local minimum, we
            # update 30 times rather than one
            for j in range(30):
                # update our weight values
                all_w[i] = update_pocket_weights(train_x, ova_labels, all_w[i])

            # check if performance improved
            # accuracy is calculated based on 26 class classification accuracy rather than
            # OVA classification accuracy to improve convergence of accuracy
            pred_y = test_pocket(all_w, train_x)
            acc = compute_accuracy(train_y, pred_y)

            if(acc <= prev_acc):
                # if current model is worse
                # rollback weights
                all_w = np.copy(prev_w)
            else:
                # if current model is better
                # update our baseline performance
                prev_w = np.copy(all_w)
                prev_acc = acc

                # only print our accuracy on improvement
                print("Iteration [" + str(cur_iter) + "] Train Accuracy [" + str(prev_acc) + "]")
            
            # count our iteration
            cur_iter += 1

            # equally shuffle both arrays to ensure new weights are found
            perm = permutation(len(train_x))
            train_x = train_x[perm]
            train_y = train_y[perm]

    # return all 26 models weights
    return all_w

# updates the pocket model weights on the first incorrect prediction
def update_pocket_weights(train_x, train_y, w):    
    # run until a misclassified point is found
    for j in range(train_y.shape[0]):
        # predict the class of our training sample
        prediction = np.sign(np.dot(w[1:], train_x[j]) + w[0])*train_y[j]
            
        # update weight vector when a misclassified point was identified
        if(prediction < 0):
            # update our weights
            w = w + np.concatenate(([1],train_x[j]))*train_y[j]

            # break out of loop
            break
        
    # return our updated weight vector
    return w

# return temple accessnet account
def get_id():
    return 'tug52339'

# Runs an individual KNN experiment and computes accuracy / confusion matrix
def run_knn(sample_size, num_nn, train_x, train_y, test_x, test_y):
    # get predictions from knn
    pred_y = test_knn(train_x, train_y, test_x, num_nn)

    # compute KNN classification accuracy
    acc = compute_accuracy(test_y, pred_y)

    # display KNN metrics
    print("\t\tSample Size [%d]\n\t\tNumber of Nearest Neighbors [%d]"
          % (sample_size, num_nn))

    print("\t\tAccuracy [" + str(acc) + "]")

    # display KNN confusion matrix
    compute_confusion_matrix(test_y, pred_y)
    
# Runs pocket algorithm experiment by training a pocket model, predicting using the model,
# and then computing a confusion matrix from those predictions
def run_pocket(sample_size, train_x, train_y, test_x, test_y, num_iters):
    print("Sample Size [%d]" % (sample_size))

    # train all 26 OVA pocket models and store their weights in a matric
    all_w = train_pocket(train_x, train_y, num_iters)

    # get predictions using all 26 pocket models
    pred_y = test_pocket(all_w, test_x)

    # compute the accuracy of our 26 class classifier
    acc = compute_accuracy(test_y, pred_y)

    print("Test ACCURACY [" + str(acc) + "]")

    # display a confusion matrix for our 26 classes
    compute_confusion_matrix(test_y, pred_y)

# displays a confusion matrix
def compute_confusion_matrix(test_y, pred_y):
    # create a 26 x 26 zeroed matrix
    cm = np.zeros((26,26), dtype=int)

    # count up all the classes in a true vs predicted manner
    for test, pred in zip(test_y, pred_y):
        cm[pred][test] += 1

    # display our confusion matrix graphically
    print("\tTrue")
    print("Predicted")
    for line in cm:
        print("\t", end="")
        for count in line:
            print("%5d" % (count), end=" ")
        print()
        
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
    num_train = [100, 1000, 2000, 5000, 10000, 15000]

    # number of nearest neighbors for KNN
    num_nn = [1, 3, 5, 7, 9]
    
    # KNN EXP
    # Run through all 6 subsamples
    for exp_num, sample_size in enumerate(num_train, 1):
        # track experiment time
        start = time.time()
        print("\nKNN EXP [%d]" % exp_num)
        
        # Run through all 5 KNN versions
        for neighbors in num_nn:
            run_knn(sample_size, neighbors, trainX[:sample_size],trainY[:sample_size],
                    testX, testY)
        print("TOTAL TIME " + str(time.time()-start))
            
    # Pocket EXP
    # Run through all 6 subsamples
    for exp_num, sample_size in enumerate(num_train, 1):
        # track experiment runtime
        start = time.time()
        
        print("\nPocket EXP [%d]" % exp_num)

        # run all 26 pocket models and evaluate the model performance
        run_pocket(sample_size, trainX[:sample_size], trainY[:sample_size],
                   testX, testY, sample_size)

        # print total computation time
        print("TOTAL TIME " + str(time.time()-start))        
    return None

if __name__ == "__main__":
    main()
