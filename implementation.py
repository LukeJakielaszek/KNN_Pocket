# Note: this is just a template for homework 1 and the code is for references only.
# Feel free to design the pipeline of the *main* function. However, one should keep
# the interfaces for the other functions unchanged.

import csv
import numpy as np
import heapq
import time
import random
from numpy.random import permutation

def compute_accuracy(test_y, pred_y):
    num_correct = 0.0
    for i in range(test_y.shape[0]):
        if(test_y[i] == pred_y[i]):
            num_correct += 1.0

    return num_correct / test_y.shape[0]

def compute_metrics(test_y, pred_y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    count = 0
    for i in range(test_y.shape[0]):
        if(pred_y[i] == test_y[i]):
            count+=1
        if(pred_y[i] == 1 and test_y[i] == 1):
            tp+=1
        elif(pred_y[i] == -1 and test_y[i] == -1):
            tn+=1
        elif(pred_y[i] == 1 and test_y[i] == -1):
            fp+=1
        else:
            fn+=1

    acc = count/test_y.shape[0]

    if(tp == 0):
        precision = 0
        recall = 0
    else:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

    return (precision, recall, acc)

def test_knn(train_x, train_y, test_x, num_nn):
    print("\tTESTING KNN...")
    pred_y = []
    for count, test in enumerate(test_x, 1):
        if(count % 1000 == 0):
            print("\tTest Samples Processed [" + str(count) + "]")

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
            indices.append(heapq.heappop(max_heap))

        # find the n minimum classes
        classes = []
        for index in indices:
            classes.append([-1*index[VAL], train_y[index[INDEX]]])

        # initialize array to count number of occurences of each class (26 classes)
        counts = [0]*26

        # get a count of each class
        for class_a in classes:
            counts[class_a[INDEX]] += counts[class_a[INDEX]] + class_a[VAL]

        # get index of max count
        max_a = np.array(counts).argsort()[-1]

        pred_y.append(max_a)
            
    return np.array(pred_y)

def test_pocket(all_w, all_bias, test_x):
    # Store all processed predictions
    pred_y = np.zeros(test_x.shape[0], dtype=int)
    
    for j, sample in enumerate(test_x, 0):
        # store each OVA prediction
        predictions = np.empty(26)
        
        for i in range(26):
            # predict the class of our training sample
            predictions[i] = np.dot(all_w[i], test_x[j]) + all_bias[i]

        # get index of most confident prediction
        pred_y[j] = predictions.argsort()[-1]
        
    return pred_y

def test_pocket_single(w, bias, test_x):
    # Store all processed predictions
    pred_y = np.zeros(test_x.shape[0], dtype=int)
    
    for j, sample in enumerate(test_x, 0):
        # predict the class of our training sample
        pred_y[j] = np.sign(np.dot(w, test_x[j]) + bias)

    return pred_y
"""
def train_pocket(train_x, train_y, num_iters):
    # weight vector is 16 long to match our features
    # initialize to random values
    w = np.random.rand(train_x.shape[1],)
    bias = random.random()
    # number of iterations to perform
    for i in range(num_iters):
        # run until a misclassified point is found
        for j in range(train_y.shape[0]):
            # predict the class of our training sample
            prediction = np.sign(np.dot(w, train_x[j]) + bias)*train_y[j]
            
            # update weight vector when a misclassified point was identified
            if(prediction < 0):
                # update our weights
                w = w + train_x[j]*train_y[j]
                
                # update our bias
                bias += prediction
                break

    # return our weight vector and bias
    return (w, bias)
"""


def train_pocket(train_x, train_y, num_iters):
    # current model weights
    all_w = np.empty(shape=(26,16))
    all_bias = np.empty(shape=(26,))

    # previous model weights
    prev_w = np.copy(all_w)
    prev_bias = np.copy(all_bias)

    prev_acc = 0
    
    # tracks our confirmed changes
    cur_iter = 0
    while(cur_iter < num_iters):        
        # initialize model weights
        for i in range(26):
            # vector to store labels in a OVA format
            ova_labels = np.empty(shape=(train_y.shape[0],))
            
            # process our label vector into OVA format for each class
            for index, class_a in enumerate(train_y, 0):
                if class_a == i:
                    ova_labels[index] = 1
                else:
                    ova_labels[index] = -1


            # update our weight values
            all_w[i], all_bias[i] = update_pocket_weights(train_x, ova_labels,
                                                          all_w[i], all_bias[i])

        # check if performance improved
        pred_y = test_pocket(all_w, all_bias, train_x)
        acc = compute_accuracy(train_y, pred_y)

        if(acc < prev_acc):
            # current model is worse
            # rollback weights
            all_w = np.copy(prev_w)
            all_bias = np.copy(prev_bias)

            # equally shuffle both arrays to ensure new weights are found
            perm = permutation(len(train_x))
            
            train_x = train_x[perm]
            train_y = train_y[perm]
        else:
            # current model is better
            # update our baseline performance
            prev_w = np.copy(all_w)
            prev_bias = np.copy(all_bias)
            prev_acc = acc
            print("Iteration [" + str(cur_iter) + "] Accuracy [" + str(prev_acc) + "]")
            
            # count our successful iteration
            cur_iter += 1        
    return(all_w, all_bias)

def update_pocket_weights(train_x, train_y, w, bias):
    # run until a misclassified point is found
    for j in range(train_y.shape[0]):
        # predict the class of our training sample
        prediction = np.sign(np.dot(w, train_x[j]) + bias)*train_y[j]
            
        # update weight vector when a misclassified point was identified
        if(prediction < 0):
            # update our weights
            w = w + train_x[j]*train_y[j]
            
            # update our bias
            bias += prediction
            break
        
    # return our updated weight vector and bias
    return (w, bias)

"""
def train_pocket(train_x, train_y, num_iters):
    # weight vector is 16 long to match our features
    # initialize to random values
    w = np.random.rand(train_x.shape[1],)
    bias = random.random()

    # get baseline accuracy
    preds_y = test_pocket_single(w, bias, train_x)
    prev_prec, prev_rec, prev_acc = compute_metrics(train_y, preds_y)
    prev_w = None
    prev_bias = None

    print(prev_acc)

    i = 0
    update_count = 0
    # number of iterations to perform
    while(i < num_iters):
        if(prev_acc >= .99999999):
            break
        
        # equally shuffle both arrays to ensure a new point of failure is found
        perm = permutation(len(train_x))                    
        train_x = train_x[perm]
        train_y = train_y[perm]

        # run until a misclassified point is found
        for j in range(train_y.shape[0]):
            # predict the class of our training sample
            prediction = np.sign(np.dot(w, train_x[j]) + bias)*train_y[j]
            
            # update weight vector when a misclassified point was identified
            if(prediction < 0):
                # store old weights
                prev_w = np.copy(w)
                prev_bias = bias
                
                # update our weights
                w = w + train_x[j]*train_y[j]
                # update our bias
                bias += prediction
                update_count+=1
                print(update_count)                

                if(update_count == 80):
                    # get updated accuracy
                    preds_y = test_pocket_single(w, bias, train_x)
                    prec, rec, acc = compute_metrics(train_y, preds_y)
                    update_count = 0
                    if(prev_acc < acc):
                        prev_acc = acc
                        # better model
                        prev_w = np.copy(w)
                        prev_bias = bias
                        i+=1
                        break
                    else:
                        # worse model continue looking
                        w = np.copy(prev_w)
                        bias = prev_bias
                        break
    print("OVA ACCURACY : ", prev_acc)
    # return our weight vector and bias
    return (w, bias)
"""
"""
def train_pocket(train_x, train_y, num_iters):
    # weight vector is 16 long to match our features
    # initialize to random values
    w = np.random.rand(train_x.shape[1],)
    bias = random.random()

    # store previous weights and accuracy
    prev_weights = np.copy(w)
    prev_bias = bias
    prev_accuracy = 0    

    i = 0
    # number of iterations to perform
    while(i < num_iters):
        i +=1
        # determines goodness of our current model
        sample_accuracy = 0
        
        # run until a misclassified point is found
        for j in range(train_y.shape[0]):

            # increment accuracy count
            sample_accuracy += 1
            
            # predict the class of our training sample
            prediction = np.sign(np.dot(w, train_x[j]) + bias)*train_y[j]

            # update or roll back weight vector when a misclassified point was identified
            if(prediction < 0):
                if(sample_accuracy > prev_accuracy):
                    print("updating weights [" + str(i) +
                          "] Accuracy [" + str(sample_accuracy) + "]"
                          + " Prev Accuracy [" + str(prev_accuracy) + "]")
                    # update our memory
                    prev_weights = np.copy(w)
                    prev_bias = bias

                    print(w)

                    # update our weights
                    w = w + train_x[j]*train_y[j]
                  
                    # update our bias
                    bias += prediction

                    # update our accuracy
                    prev_accuracy = sample_accuracy
                else:
                    print("Rolling weights back [" + str(i) +
                          "] Sample Accuracy [" + str(sample_accuracy) + "]" +
                          " Prev Accuracy [" + str(prev_accuracy) + "]")
                    # rollback our weights
                    w = np.copy(prev_weights)
                    bias = prev_bias

                    # equally shuffle both arrays to ensure a new point of failure is found
                    perm = permutation(len(train_x))

                    train_x = train_x[perm]
                    train_y = train_y[perm]
                break

    print("FINISH [" + str(i) +
              "] Sample Accuracy [" + str(sample_accuracy) + "]" +
              " Prev Accuracy [" + str(prev_accuracy) + "]")

    print(w)
    exit()
    # return our weight vector and bias
    return (w, prev_bias)
"""

# return temple accessnet account
def get_id():
    return 'tug52339'

def run_knn(sample_size, num_nn, train_x, train_y, test_x, test_y):
    pred_y = test_knn(train_x, train_y, test_x, num_nn)

    acc = compute_accuracy(test_y, pred_y)
    
    print("\t\tSample Size [%d]\n\t\tNumber of Nearest Neighbors [%d]"
          % (sample_size, num_nn))

    print("\t\tAccuracy [" + str(acc) + "]")

    compute_confusion_matrix(test_y, pred_y)

    
# implement OVA pocket algorithm
def run_pocket(sample_size, train_x, train_y, test_x, test_y, num_iters):
    print("Sample Size [%d]" % (sample_size))
    
    all_w, all_bias = train_pocket(train_x, train_y, num_iters)

    pred_y = test_pocket(all_w, all_bias, test_x)

    acc = compute_accuracy(test_y, pred_y)

    print("ACCURACY [" + str(acc) + "]")

    compute_confusion_matrix(test_y, pred_y)

    
    return None

"""    
# implement OVA pocket algorithm
def run_pocket(sample_size, train_x, train_y, test_x, test_y, num_iters):
    print("Sample Size [%d]" % (sample_size))

    all_w = np.empty(shape=(26,16))
    all_bias = np.empty(shape=(26,))

    for i in range(26):
        # vector to store labels in a OVA format
        ova_labels = np.empty(shape=(train_y.shape[0],))

        # process our label vector into OVA format for each class
        for index, class_a in enumerate(train_y, 0):
            if class_a == i:
                ova_labels[index] = 1
            else:
                ova_labels[index] = -1

        all_w[i], all_bias[i] = train_pocket(train_x, ova_labels, num_iters)

    pred_y = test_pocket(all_w, all_bias, test_x)

    acc = compute_accuracy(test_y, pred_y)

    print(acc)

    compute_confusion_matrix(test_y, pred_y)

    
    return None
"""

def compute_confusion_matrix(test_y, pred_y):
    cm = np.zeros((26,26), dtype=int)

    for test, pred in zip(test_y, pred_y):
        cm[pred][test] += 1

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
    #for exp_num, sample_size in enumerate(num_train, 1):
    #    start = time.time()
    #    print("\nKNN EXP [%d]" % exp_num)
        
        # Run through all 5 KNN versions
    #    for neighbors in num_nn:
    #        run_knn(sample_size, neighbors, trainX[:sample_size],trainY[:sample_size], testX, testY)
    #    print("TOTAL TIME " + str(time.time()-start))
            
    # Pocket EXP
    # Run through all 6 subsamples
    for exp_num, sample_size in enumerate(num_train, 1):
        start = time.time()
        print("\nPocket EXP [%d]" % exp_num)

        run_pocket(sample_size, trainX[:sample_size], trainY[:sample_size],
                   testX, testY, 26*sample_size)
        print("TOTAL TIME " + str(time.time()-start))        
    return None

if __name__ == "__main__":
    main()
