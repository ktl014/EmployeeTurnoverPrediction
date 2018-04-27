'''
preprocessing

Created on Apr 24 2018 17:04 
#@author: Kevin Le 
'''
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.python.data import Dataset
import tensorflow as tf

class Preprocesser(object):
    def __init__(self, csvFileName, shuffle=True, target='turnover'):
        '''

        :param csvFileName:
        :param shuffle: Bool type
        '''
        assert os.path.exists(csvFileName)

        self.csvFilename = csvFileName
        self.df = pd.read_csv(csvFileName)
        self.totalSmpls = self.df.shape[0]
        self.target = target

        if shuffle:
            self.df = self.df.reindex(np.random.permutation(self.df.index))

    def __len__(self):
        return self.df.shape[0]

    def cleanData(self):
        print('Cleaning data... ')
        self.df = self.df.rename(columns={'satisfaction_level': 'satisfaction',
                                 'last_evaluation': 'evaluation',
                                 'number_project': 'projectCount',
                                 'average_montly_hours': 'averageMonthlyHours',
                                 'time_spend_company': 'yearsAtCompany',
                                 'Work_accident': 'workAccident',
                                 'promotion_last_5years': 'promotion',
                                 'sales': 'department',
                                 'left': 'turnover'
                                 })

        # Convert these variables into categorical variables
        self.df["department"] = self.df["department"].astype ('category').cat.codes
        self.df["salary"] = self.df["salary"].astype ('category').cat.codes


    def trainTestSplit(self, split=0.10, tensorSplit=False):
        print('Partitioning data into training & testing... ')
        self.targetDF = self.df[self.target]
        self.df = self.df.drop(self.target, axis=1)

        if tensorSplit:
            trainSize = int(self.totalSmpls * (1 - split))
            X_train = self.df.head(trainSize)
            y_train = self.targetDF.head(trainSize)

            testSize = self.totalSmpls - trainSize
            X_test = self.df.tail(testSize)
            y_test = self.targetDF.tail(testSize)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.df, self.targetDF, test_size=split, random_state=123, stratify=self.targetDF)

        return X_train, X_test, y_train, y_test

    def constructFeatureColumns(self, inputFeatures):
        return set([tf.feature_column.numeric_column(feature) for feature in inputFeatures])

    def convertInputFn(self, features, labels, numEpochs=None, shuffle=True, batchSize=1):
        '''
        Maps feature columns and labels to Tensors
        :return:
        '''
        features = {key: np.array (value) for key, value in dict (features).items ()}

        ds = Dataset.from_tensor_slices((features, labels))
        ds = ds.batch(batchSize).repeat(numEpochs)

        if shuffle:
            ds = ds.shuffle(10000)

        tensorFeatures, tensorLabels = ds.make_one_shot_iterator().get_next()
        return tensorFeatures, tensorLabels

if __name__ == '__main__':
    root = '/Users/ktl014/PycharmProjects/PersonalProjects/EmployeeTurnOverPrediction/'
    data = root + 'data/HR_comma_sep.csv'
    dataPreprocesser = Preprocesser(csvFileName=data)
    dataPreprocesser.cleanData()
    X_train, X_test, y_train, y_test = dataPreprocesser.trainTestSplit(tensorSplit=True)
    trainInputFN = lambda: dataPreprocesser.convertInputFn(X_train, y_train, batchSize=20)