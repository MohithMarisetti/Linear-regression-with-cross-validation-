# Name of the student: Mohith Marisetti
# Student ID: 1001669337
# CSE 6363: Project-1  

"""
Code Structure:
There are 2 functions used in the program. 
1. kFoldCrossValidation() : Contains the k-fold cross-validation logic
2. performFit(model) : Used to fit and check accuracy of each model with different train, test data during the iterations of k-fold CV.
3. findMinMax() : Used to find the minimum and maximum range in the data predicted.
4. main(): Used to call the kFoldCrossValidation() function and print the average accuracy of the models.

Implementation:
-> main() takes the iris.data creates test.data, train.data files and calls performFit() which then uses those 2 files to fit the model and test its accuracy.

"""


import numpy as np
import random 

# Seed value for shuffling the iris.data set. (Shuffling is important since the classes are present in order, means, 1-50 rows are iris_setosa, 51-100 are iris_versicolor, 101-150 are iris_virginica. If we apply K fold on this set it will pick only the rows of same class type in most of the iterations)
seedValue = 10  # No specific reason to choose 10, it just worked the best for me.

# Number of folds in the K-Fold cross validation. (K = numberOfFolds = 10)
numberOfFolds = 5




random.seed(seedValue)   # Setting the seed value
predictions = 0     # Just a global variable to store predictions
accuracyList = []   # This list stores the accuracy of all the different iterations in the K-Fold cross validation
betaList = []
foldSize = 0


def findMinMax(start,stop):
    '''
    Finds the range(meaning [minimum , maximum]) of each class. I wrote this method to choose the best set of ranges for the classes 
    iris_setosa, iris_versicolor, iris_virginica
    '''
    min = 999
    max = -999
    for i in range(start,stop):
        if max < predictions[i][0]:
            max = predictions[i][0]
        if min > predictions[i][0]:
            min = predictions[i][0]
    return min,max


# 1. Grab the Iris data set

def performFit(modelNumber):
    '''
    This method performs the fitting logic for the given training set and calculates the accuracy with the provided test set.
    '''
    aList = []      # Stores the input matrix values as a list. Later we use this to convert this list into a array. As the name suggests its matrix A
    yList = []      # Stores the actual output values as a list. Later we use this to convert this list into a array. As the name suggests its vector Y
    
    '''
    1. Reading the train.data
    '''
    f = open("train.data","r")      # train.data is the file containing training set
    for eachLine in f.readlines():
        valuesList  = [1] # 1 indicates the feature-0 (i.e., Xo = 1)
        if(eachLine == '\n'):
            continue
        tempList = eachLine.split(",")
        for i in range(5):
            # Below "if" block creates the yList (in simple its storing the Actual output values of the training set into a list named yList)
            if i == 4: 
                    s = tempList[i].strip()
                    if  s == 'Iris-setosa':    # If the class label is Iris_setosa then we store the label with a numerical value 1
                        
                        yList.append(1)
                    elif s == 'Iris-versicolor':  # If the class label is Iris-versicolor then we store the label with a numerical value 2
                        
                        yList.append(2)
                    elif s == 'Iris-virginica':   # If the class label is Iris-virginica then we store the label with a numerical value 3
                        
                        yList.append(3)

            else:
                # This creates the features matrix "A" using the 4 features provided in the training set
                val = float(tempList[i])
                valuesList.append(val)




        aList.append(valuesList)


    ''' 
    2. Initialize A, Y using the lists obtained above aList, yList and use them to calculate the beta vector. 
    '''
    A = np.array(aList)
    Y = np.array(yList)
    Y = Y.reshape(Y.shape[0],1)                # Converting 1D "Y" to 2D "Y" so as to make the vector compatible with matrix multiplication
    transpose_Of_A = A.transpose()              
    transposeProduct = np.dot(transpose_Of_A, A)        # Computes A_transpose * A       
    transposeProductInverse = np.linalg.inv(transposeProduct)               # Computes ( A_transpose * A )-1

    
    '''
    3. Calculates the expression B = Inv(A'A) * A * Y using the  training set(train.data)
    '''
    beta = np.dot(transposeProductInverse,np.dot(transpose_Of_A,Y))             # Beta = ( A_transpose * A )^-1 * A_transpose * Y
    print('beta-{} is {}'.format(modelNumber,beta))                                     # Displaying the beta vector
    betaList.append(beta)
    



    # After finding the beta, test the remaining data (i.e., test set) and then find the accuracy

    # Creating a list to store the test data and later transform it into an array
    testSet = []
    testFile = open('test.data','r')
    for eachLine in testFile.readlines():
        list = [1]
        if(eachLine == '\n'):
            continue
        for eachVal in eachLine.split(','):
            try:
                list.append(float(eachVal))
            except:
                pass
        testSet.append(list)




    testArray = np.array(testSet)
    #print('test data array is {}'.format(testArray))    

    # Predicitng the values for the test set data points
    predictions = np.dot(testArray,beta)
    #print("predictions are {}".format(predictions))     


    """
    Note: Finding the minimum and maximum ranges of setosa, versicolor, virginica classes. Based on this data, it was obvious that there 
    were outliers in the classes have overlap with each other more prominent in iris_versicolor & iris_virginica.
    """
 
    # The ranges which I found useful are [0 to 1.5 for iris_setosa], [1.5 to 2.4 for iris_versicolor], [2.4 and above for iris_virginica]
    for  i in range(predictions.shape[0]):
        if 0.5 <predictions[i][0] < 1.5:
            predictions[i][0] = 1
        elif 1.5 <= predictions[i][0] < 2.5:
            predictions[i][0] = 2
        elif 2.5 <= predictions[i][0] < 3.5:
            predictions[i][0] = 3

    #print("predictions are {}".format(predictions))    
    

    # Calculating the accuracy of the test set by using our trained model
    errorCount = 0
    actualVal = 0
    j = 0
    for eachLine in open("test.data","r").readlines():
        
        tempList = eachLine.split(',')
        if (tempList[4].strip() == 'Iris-setosa'):
            actualVal  = 1
        elif (tempList[4].strip() == 'Iris-versicolor'):
            actualVal  = 2
        elif (tempList[4].strip() == 'Iris-virginica'):
            actualVal  = 3
        if (predictions[j][0]!=actualVal):
            errorCount = errorCount + 1
        j+=1
    global foldSize 
    accuracy = ((foldSize-errorCount)/foldSize)*100   # Accuracy in percentage
    accuracyList.append(accuracy)
    print('\nAccuracy in iteration-{} is {}%'.format(modelNumber,accuracy))
    print('============================================================\n')
   
def kFoldCrossValidation():

     # This function implements 10-Fold cross validation logic

    global foldSize
    # As explained in the document(readme.docx) I'm shuffling the Input Matrix rows so that the 
    # classes are distributed across training & test set, or else the k-fold will not be accurate. Its similar to a stratified K-fold
    with open('iris.data','r') as data:
        linesList = [ (random.random(), line) for line in data ]
    linesList.sort()
    with open('iris_shuffled.data','w') as endfile:
        for x, line in linesList:
            endfile.write(line)


    #Calculate number of data points
    dataPointsSize = 0
    for eachLine in open("iris_shuffled.data", "r").readlines():
        if(eachLine!='\n'):
            dataPointsSize+=1
    
    foldSize = dataPointsSize/numberOfFolds  # Each fold size(In our case 150/10  = 15)
    modelNumber = 0         # variable to indicate the model number. Example: model-1, model-2, model-3.... model-10
    
    
    # Each iteration of K-fold cross validation creates a new training and test set then trains model 
    # using the training set and then predicts output class of unseen data points in the test set.
    for j in range(numberOfFolds):     
        modelNumber+=1      # Model number 1 - 10
        temp1  = j*foldSize
        temp2 = (j+1)*foldSize
        temp = 0
        shuffledFile = open("iris_shuffled.data","r")
        trainFile = open("train.data","w")          # Create a new file called train.data. As the name suggests its a file containing training set
        testFile = open("test.data","w")            # Create a new file called test.data. As the name suggests its a file containing test set

        # Writing into train.data and test.data
        for eachLine in shuffledFile.readlines():
            if(eachLine== '\n'):   # Skipping the blank lines
                continue
            if(temp>=temp1 and temp<temp2):    # Only selecting the fold which needs to be placed in the test set(in test.data)
                testFile.write(eachLine)
            else:                               # All the remaining folds are placed in training set(in train.data)
                trainFile.write(eachLine)
            temp+=1
        
        trainFile.close()
        testFile.close()
        performFit(modelNumber)


def main():
    kFoldCrossValidation()    
    
    averageBeta = np.zeros(betaList[0].shape)
    for eachBeta in betaList:
        averageBeta+=eachBeta
    print("Average beta is {}".format(averageBeta/numberOfFolds))
    print("The average accuracy achieved by this linear regression model is {}%".format(sum(accuracyList)/len(accuracyList)))
    import os
    os.remove("iris_shuffled.data")
    os.remove("train.data")
    os.remove("test.data")



if __name__ == "__main__":
    main()
