'''
main

Created on Apr 15 2018 16:47 
#@author: Kevin Le 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from model.preprocessing import Preprocesser

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

SEED = 123

def main():
    print 'Loading dataset... '
    root = '/Users/ktl014/PycharmProjects/PersonalProjects/EmployeeTurnOverPrediction/'
    data = root + 'data/HR_comma_sep.csv'

    dataPreprocesser = Preprocesser(csvFileName=data)
    dataPreprocesser.cleanData()
    trainFeatures, valFeatures, trainLbls, valLbls = dataPreprocesser.trainTestSplit(tensorSplit=True)

    linearClassifier = trainLogRegressionModel(
        learningRate=0.000005,
        steps=500,
        batchSize=20,
        preprocesser=dataPreprocesser,
        trainingExamples=trainFeatures,
        trainingTargets=trainLbls,
        validationExamples=valFeatures,
        validationTargets=valLbls)

    print('Model size: {}'.format(computeModelSize(estimator=linearClassifier)))

    valInputFN = lambda: dataPreprocesser.convertInputFn(valFeatures, valLbls,
                                                       numEpochs=1,
                                                       shuffle=False)

    valOutput = linearClassifier.predict (input_fn=valInputFN)
    # Get just the probabilities for the positive class.
    valProb, valPred = [], []
    for item in valOutput:
        valProb.append(item['probabilities'][1])
        valPred.append(item['class_ids'][0])

    falsePositiveRate, truePositiveRate, thresholds = metrics.roc_curve (
        valLbls, valProb)
    plt.plot (falsePositiveRate, truePositiveRate, label="our model")
    plt.plot ([0, 1], [0, 1], label="random classifier")
    _ = plt.legend (loc=2)
    plt.show()

    evaluationMetrics = linearClassifier.evaluate (input_fn=valInputFN)

    print "AUC on the validation set: %0.2f" % evaluationMetrics['auc']
    print "Accuracy on the validation set: %0.2f" % evaluationMetrics['accuracy']

    evaluateModel(gtruth=valLbls, predictions=valPred)

def computeModelSize(estimator):
    variables = estimator.get_variable_names()
    size = 0
    dontCountVariables = ['global_step',
                          'centered_bias_weight',
                          'bias_weight',
                          'Ftrl']
    for variable in variables:
        if not any(x in variable for x in dontCountVariables):
            size += np.count_nonzero(estimator.get_variable_value(variable))
    return size

def trainLogRegressionModel(learningRate,
                            steps,
                            batchSize,
                            preprocesser,
                            trainingExamples,
                            trainingTargets,
                            validationExamples,
                            validationTargets):
    """ Trains a linear classification model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learningRate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batchSize: A non-zero `int`, the batch size.
      trainingExamples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      trainingTargets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validationExamples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validationTargets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A `LinearClassifier` object trained on the training data.
    """
    periods = 10
    stepsPerPeriod = steps / periods

    # Create a linear regressor object.
    optimizer = tf.train.GradientDescentOptimizer (learning_rate=learningRate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm (optimizer, 5.0)
    linearClassifier = tf.estimator.LinearClassifier (
        feature_columns=preprocesser.constructFeatureColumns(trainingExamples),
        optimizer=optimizer
    )
    # Create input functions.
    trainingInputFN = lambda: preprocesser.convertInputFn(trainingExamples,
                                             trainingTargets,
                                             batchSize=batchSize)
    predictTrainingInputFN = lambda: preprocesser.convertInputFn(trainingExamples,
                                                     trainingTargets,
                                                     numEpochs=1,
                                                     shuffle=False)
    predictValidationInputFN = lambda: preprocesser.convertInputFn(validationExamples,
                                                       validationTargets,
                                                       numEpochs=1,
                                                       shuffle=False)
    print 'Training model...'
    print 'RMSE (on training data)'
    trainingLogLosses = []
    validationLogLosses = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linearClassifier.train (
            input_fn=trainingInputFN,
            steps=stepsPerPeriod
        )
        # Take a break and compute predictions.
        trainingProbabilities = linearClassifier.predict (input_fn=predictTrainingInputFN)
        trainingProbabilities = np.array ([item['probabilities'] for item in trainingProbabilities])

        validationProbabilities = linearClassifier.predict (input_fn=predictValidationInputFN)
        validationProbabilities = np.array ([item['probabilities'] for item in validationProbabilities])

        trainingLogLoss = metrics.log_loss (trainingTargets, trainingProbabilities)
        validationLogLoss = metrics.log_loss (validationTargets, validationProbabilities)
        # Occasionally print the current loss.
        print "  period %02d : %0.2f" % (period, trainingLogLoss)
        # Add the loss metrics from this period to our list.
        trainingLogLosses.append (trainingLogLoss)
        validationLogLosses.append (validationLogLoss)
    print "Model training finished."

    # Output a graph of loss metrics over periods.
    plt.ylabel ("LogLoss")
    plt.xlabel ("Periods")
    plt.title ("LogLoss vs. Periods")
    plt.tight_layout ()
    plt.plot (trainingLogLosses, label="training")
    plt.plot (validationLogLosses, label="validation")
    plt.legend ()
    plt.show()

    return linearClassifier

def trainRandomForestModel(trainingExamples,
                           trainingTargets,
                           validationExamples,
                           validaitonTargets):

    estimator = RandomForestClassifier(n_estimators=1000,
                                       min_samples_split=10,
                                       max_features='sqrt',
                                       random_state=SEED)
    estimator.fit(trainingExamples, trainingTargets)
    pred = estimator.predict(validationExamples)
    return estimator, pred

def evaluateModel(predictions, gtruth, normalizeCM=False):
    '''
    Evaluates a model by outputting a confusion matrix


    :return:
    N/A
    '''
    cm = metrics.confusion_matrix(gtruth, predictions)
    print('Confusion Matrix w/out Normalization')
    print(cm)
    if normalizeCM:
        CM = cm.astype ('float') / cm.sum (axis=1)[:, np.newaxis]
    print('Confusion Matrix with Normalization')
    print(CM)
    print(metrics.classification_report(gtruth, predictions))


if __name__ == '__main__':
    main()