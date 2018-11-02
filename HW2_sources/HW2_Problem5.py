#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import norm 
from scipy.stats import bernoulli
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import sys
import csv
import math
import copy
import os


# In[ ]:


# Global Variables

kDim = 10 # the number of dimension
# kUsers = 610 # 610 is total, but we use 300 users 
# kMovies = 9724 # 9724 is total, but we use 6000 movies

# These data are used to draw a kDim-vector that follows Gaussian Distribution (kMean, kCov)
kMean = []
for i in range(kDim):
    kMean.append(0)
kCov = 0.1 * np.identity(n=kDim,dtype='float')

# Rating matrix, which is sparsew
gR = dict() # Key = (userId, movieId)
gD = dict() # Key = (userId, moveId) value = existence
gU = dict() # Key = (userId), value = user vector
gV = dict() # Key = (movieId), value = movie vector


# In[ ]:


# Generate all model variables reading given file(csv)
# It should be executed first before calling MakeTestSet which split the whole file into 
# training file and test file
def GenerateModelVariables (in_file_name, in_U, in_V):
    file = open(in_file_name, 'r')
    fileReader = csv.reader(file)
    
    for row in fileReader:
        if row[0] == 'userId':
            continue
        else:
            currentUserID = int(row[0])
            currentMovieID = int(row[1])
            currentRating = float(row[2])

            if type(in_U.get(currentUserID)) == type(None):
                in_U[currentUserID] = np.random.multivariate_normal(kMean, kCov, 1).T
            if type(gV.get(currentMovieID)) == type(None):
                in_V[currentMovieID] = np.random.multivariate_normal(kMean, kCov, 1).T


# In[ ]:


# Make test set... from given training file
def MakeTestSet(in_file_name, in_ratio, out_trainingFileName, out_testFileName):
    originalFile = open(in_file_name, 'r')
    fileReader = csv.reader(originalFile)
    
    trainingFile = open(out_trainingFileName,'w')
    testFile = open(out_testFileName, 'w')
    
    trainingCSVWriter = csv.writer(trainingFile, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
    testCSVWriter = csv.writer(testFile, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
    
    trainingCSVWriter.writerow(['userId', 'movieId', 'rating'])
    testCSVWriter.writerow(['userId', 'movieId', 'rating'])
    
    for row in fileReader:
        if row[0] == 'userId':
            continue
        else:
            bTrainingSet = bernoulli.rvs(1 - in_ratio, size = 1)[0] # 1 indicates training data
            if bTrainingSet == 1:
                trainingCSVWriter.writerow([row[0], row[1], row[2]])
            else:
                testCSVWriter.writerow([row[0], row[1], row[2]])


# In[ ]:


# Choose data in a set... from given file
def ChooseSomeOfDataSet(in_file_name, in_ratio, out_trainingFileName):
    originalFile = open(in_file_name, 'r')
    fileReader = csv.reader(originalFile)
    
    trainingFile = open(out_trainingFileName,'w')
    
    trainingCSVWriter = csv.writer(trainingFile, delimiter=',',quotechar=',', quoting=csv.QUOTE_MINIMAL)
    trainingCSVWriter.writerow(['userId', 'movieId', 'rating'])
    
    for row in fileReader:
        if row[0] == 'userId':
            continue
        else:
            bTrainingSet = bernoulli.rvs(in_ratio, size = 1)[0] # 1 indicates training data
            if bTrainingSet == 1:
                trainingCSVWriter.writerow([row[0], row[1], row[2]])


# In[ ]:


# Read training file
# This function extract all information on the set of given ratings and existence
def ReadTrainingCSVFile(in_file_name, out_D, out_R):
    training_file = open(in_file_name, 'r')
    fileReader = csv.reader(training_file)
    
    for row in fileReader:
        if row[0] == 'userId':
            continue
        else:            
            currentUserID = int(row[0])
            currentMovieID = int(row[1])
            currentRating = float(row[2])
                        
            out_D[(currentUserID, currentMovieID)] = 1
            out_R[(currentUserID, currentMovieID)] = currentRating
            


# In[ ]:


# It is used to verify that the objective function is minimizing
def CalculateJointLikelihood(in_R, in_U, in_V, in_D):
    
    logLikelihood = 0.0
    
    for eachKey in in_D.keys():
        logLikelihood = logLikelihood + ((in_R[eachKey] - (in_U[eachKey[0]].T @ in_V[eachKey[1]])) ** 2)
    
    return logLikelihood


# In[ ]:


# We solve this problem iteratively. 
# For example, first set all U, V = 0
#
# For the first iteration:
# Calculate U using given V
# Next, 
# Calculate V using calculated U above.

def CalculateMLE(in_init_U, in_init_V, in_D, in_R, in_nIterations):
    
    U = copy.deepcopy(in_init_U) # init U
    V = copy.deepcopy(in_init_V) # init V
    
    for t in range(in_nIterations):
        for i in U.keys():
            firstTerm = np.zeros([kDim, kDim], float)
            secondTerm = np.zeros([kDim, 1], float)

            for j in V.keys():
                if type(in_D.get((i,j))) != type(None):
                    
                    firstTerm = firstTerm + (V[j] @ V[j].T)
                    secondTerm = secondTerm + (in_R[(i, j)] * V[j])
            
            U[i] = np.linalg.pinv(firstTerm) @ secondTerm

        for j in V.keys():
            firstTerm = np.zeros([kDim, kDim], float)
            secondTerm = np.zeros([kDim, 1], float)

            for i in U.keys():
                if type(in_D.get((i,j))) != type(None):
                    
                    firstTerm = firstTerm + (U[i] @ U[i].T)
                    secondTerm = secondTerm + (in_R[(i, j)] * U[i])
            
            V[j] = np.linalg.pinv(firstTerm) @ secondTerm
        
        # calculate current joint likelihood
        currentLikelihood = CalculateJointLikelihood(in_D=gD, in_R= gR, in_U=U, in_V=V)
        
        # Show the current value of objective function
        print("Value => ", currentLikelihood)
        
    return U, V


# In[ ]:


def GetSample_UV(in_init_U, in_init_V, in_R, in_D, in_nSamples, in_stepSize):
    
    U = copy.deepcopy(in_init_U) # init U
    V = copy.deepcopy(in_init_V) # init V
    
    nUsers = len(U)
    nMovies = len(V)
    c = 1.0
    
    u_samples = []
    v_samples = []
    
    for n in range(in_nSamples):
        for t in range(in_stepSize):
            print("iteration => ", t)
            for i in U.keys():
                firstTerm = np.identity(n=kDim,dtype='float') / c
                secondTerm = np.zeros([kDim,1],dtype='float')

                for j in V.keys():
                    if type(in_D.get((i,j))) != type(None):
                        firstTerm = firstTerm + (V[j] @ V[j].T)
                        secondTerm = secondTerm + (in_R[(i, j)] * V[j])

                covTerm = np.linalg.inv(firstTerm)
                muTerm = (covTerm @ secondTerm).T

                U[i] = np.random.multivariate_normal(muTerm[0], covTerm, 1).T

            for j in V.keys():
                firstTerm = np.identity(n=kDim,dtype='float') / c
                secondTerm = np.zeros([kDim,1],dtype='float')

                for i in U.keys():
                    if type(in_D.get((i,j))) != type(None):

                        firstTerm = firstTerm + (U[i] @ U[i].T)
                        secondTerm = secondTerm + (in_R[(i, j)] * U[i])

                covTerm = np.linalg.inv(firstTerm)
                muTerm = (covTerm @ secondTerm).T

                V[j] = np.random.multivariate_normal(muTerm[0], covTerm, 1).T
        
        print("We've got one sample from full posterior distribution")
        u_samples.append(U)
        v_samples.append(V)
    return u_samples,v_samples


# In[ ]:


def GetAllErrors_GibbsVersion(in_testFileName, in_D, in_R, in_init_U, in_init_V, in_nSamples):
    
    USamples, VSamples = GetSample_UV(in_D=in_D,
                                      in_R=in_R,
                                      in_init_U=in_init_U,
                                      in_init_V=in_init_V,
                                      in_nSamples=in_nSamples,
                                      in_stepSize=10)
    
    trainingSquaredErrorSum = 0.0
    testSquaredErrorSum = 0.0
    
    ##### training error
    for eachKey in in_D.keys():
        sum = 0.0
        for n in range(in_nSamples):
            sum = sum + (USamples[n][eachKey[0]].T @ VSamples[n][eachKey[1]])[0][0]
        sum = sum / float(in_nSamples)
        trainingSquaredErrorSum = trainingSquaredErrorSum + ((in_R[eachKey] - sum) ** 2)
    
    trainingSquaredErrorSum = trainingSquaredErrorSum / len(gR)
    print('Training Error (Gibbs)', trainingSquaredErrorSum)
    
    #### test error
    test_file = open(in_testFileName, 'r')
    fileReader = csv.reader(test_file)
    
    testSquaredErrorSum = 0.0
    testSetCount = 0
    for row in fileReader:
        if row[0] == 'userId':
            continue
        else:
            testSetCount = testSetCount + 1
            currentUserID = int(row[0])
            currentMovieID = int(row[1])
            currentRating = float(row[2])
                
            sum = 0.0
            for n in range(in_nSamples):
                sum = sum + (USamples[n][currentUserID].T @ VSamples[n][currentMovieID])[0][0]
            estimatedRating = sum / float(in_nSamples)
            
            testSquaredErrorSum = testSquaredErrorSum + ((estimatedRating - currentRating) ** 2)
    
    testSquaredErrorSum = testSquaredErrorSum / float(testSetCount)
    print('Test Error (Gibbs)', testSquaredErrorSum)
    
    return trainingSquaredErrorSum, testSquaredErrorSum
    


# In[ ]:


def GetAllErrors_MLEVersion(in_testFileName, in_D, in_R, in_init_U, in_init_V, in_nIterations):
    
    U, V = CalculateMLE(gU, gV, gD, gR, in_nIterations)
    trainingSquaredErrorSum = 0.0
    testSquaredErrorSum = 0.0
    
    ##### training error
    for eachKey in in_D.keys():
        trainingSquaredErrorSum = trainingSquaredErrorSum + (((U[eachKey[0]].T @ V[eachKey[1]])[0][0] - in_R[eachKey]) ** 2)
    
    trainingSquaredErrorSum = trainingSquaredErrorSum / len(gR)
    print('Training Error (MLE)', trainingSquaredErrorSum)
    
    #### test error
    test_file = open(in_testFileName, 'r')
    fileReader = csv.reader(test_file)
    
    testSquaredErrorSum = 0.0
    testSetCount = 0
    testErrorList = []
    
    for row in fileReader:
        if row[0] == 'userId':
            continue
        else:
            testSetCount = testSetCount + 1
            currentUserID = int(row[0])
            currentMovieID = int(row[1])
            currentRating = float(row[2])
                
            estimatedRating = (U[currentUserID].T @ V[currentMovieID])[0][0]
            testErrorList.append(((estimatedRating - currentRating) ** 2))
    
    print('Test Error (MLE)', np.mean(testErrorList))
    
    return trainingSquaredErrorSum, np.mean(testErrorList), testErrorList
    


# In[ ]:


gU.clear()
gV.clear()

# os.remove('wholeTrainingSet.csv')   # fixed    ex) 80% are training
# os.remove('testSet.csv')            # fixed    ex) 20% are test

# Get all user id and movie id in order to avoid encontering unseen user or movie 
# init gU, gV
GenerateModelVariables (in_file_name='movie_ratings.csv', in_U=gU, in_V=gV)

# 20% of whole set is a test set
testSetRatio = 0.2
MakeTestSet('movie_ratings.csv', testSetRatio, 'wholeTrainingSet.csv', 'testSet.csv')


# In[ ]:


gD.clear()
gR.clear()
ChooseSomeOfDataSet(in_file_name='wholeTrainingSet.csv', in_ratio = 0.6, out_trainingFileName='someTrainSet.csv')
ReadTrainingCSVFile(in_file_name='someTrainSet.csv',out_D=gD, out_R=gR)
GetAllErrors_MLEVersion(in_D=gD, in_R=gR, in_init_U=gU,in_init_V=gV,in_nIterations=30,in_testFileName='testSet.csv')


# In[ ]:


trainingRatios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

trainingMLE_error_list = []
testMLE_error_list = []
trainingGibbs_error_list = []
testGibbs_error_list = []

for i in range(len(trainingRatios)):
    gD.clear()
    gR.clear()
    
    print('=====================================')
    print('ratio is ==> ', trainingRatios[i], ' ')
    print('=====================================')
    
    # some of trainingSet is also a training set
    ChooseSomeOfDataSet(in_file_name='wholeTrainingSet.csv', in_ratio = trainingRatios[i], out_trainingFileName='someTrainSet.csv')
    ReadTrainingCSVFile(in_file_name='someTrainSet.csv',out_D=gD, out_R=gR)
    GetAllErrors_MLEVersion(in_D=gD, in_R=gR, in_init_U=gU,in_init_V=gV,in_nIterations=30,in_testFileName='testSet.csv')
    print(' ')
    GetAllErrors_GibbsVersion(in_testFileName='testSet.csv', in_D=gD, in_R=gR, in_init_U=gU, in_init_V=gV, in_nSamples=5)
    
    os.remove('someTrainSet.csv')


# In[ ]:


trainingRatioIndex = ["30%", "40%", "50%", "60%", "70%"]
MLE_test_errors = [335414, 273182, 126636, 1225665, 143401.4539753]
Gibbs_test_errors = [3.8439273060086645, 3.1661662479381523, 2.6981003461349196, 2.5245830148313915,2.2656395976138404]

MLE_training_errors = [0.10897821251500237, 0.10423049514512372, 0.12896267601352068, 0.1687827658906558, 0.17746496307287785]
Gibbs_training_errors = [0.7276150374115209,  0.69941911662465, 0.6766866324487065, 0.683105207595826, 0.6630517063772686]

plt.figure(figsize=(10,7))
plt.plot(MLE_test_errors,'o--', color='darkgreen', label='MLE Test error')
plt.plot(Gibbs_test_errors, 'o--', color='blue', label='Our model Test error')
plt.legend()
plt.title('Test Errors')
plt.xlabel('Training Data Size')
plt.xticks(range(len(trainingRatios)), trainingRatioIndex)

plt.figure(figsize=(10,7))
plt.plot(MLE_training_errors ,'o--', color='darkgreen', label='MLE Training error')
plt.plot(Gibbs_training_errors, 'o--', color='blue', label='Our model Training error')
plt.legend()
plt.title('Training Errors')
plt.xlabel('Training Data Size')
plt.xticks(range(len(trainingRatios)), trainingRatioIndex)


# In[ ]:





# In[ ]:




