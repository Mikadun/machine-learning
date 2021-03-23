import numpy as np
import logging

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    prediction = np.where(prediction > 0, 0, 1)
    ground_truth = np.where(ground_truth > 0, 0, 1)
    
    TP = np.sum(np.logical_and(prediction, ground_truth))
    FN = np.sum(ground_truth - np.logical_and(prediction, ground_truth))
    precision = TP / np.sum(prediction)
    recall = TP / (TP + FN)
    accuracy = multiclass_accuracy(prediction, ground_truth)

    logging.info('Prediction: {}'.format(prediction.astype(np.int)))
    logging.info('Right answ: {}'.format(ground_truth.astype(np.int)))
    logging.info('TP: {}'.format(TP))
    logging.info('TP + FP: {}'.format(np.sum(prediction)))
    logging.info('FN: {}'.format(FN))

    f1 = 2 * precision * recall / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return np.sum(prediction == ground_truth) / prediction.shape[0]
