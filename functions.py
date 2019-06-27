# Import packages and change some pandas display options
import pandas as pd
import numpy as np
import re
import warnings
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import auc, f1_score, roc_curve, roc_auc_score


def get_simple_features(data):
    """ Creates simple features from the comments and adds them as new columns to the dataset."""  #

    data['words'] = data.Comment_text.apply(lambda x: len(x.split()))
    data['uppercase'] = data.Comment_text.apply(lambda x: len(re.findall("[A-Z]", x)))
    data['uppercase_per_word'] = data['uppercase'] / data['words']
    data['punctuation'] = data.Comment_text.apply(lambda x: len(re.findall("[,.!?]", x)))
    data['punctuation_per_word'] = data['punctuation'] / data['words']
    data['numbers'] = data.Comment_text.apply(lambda x: len(re.findall("[1-9]+", x)))
    data['numbers_per_word'] = data['numbers'] / data['words']

    simple_features = ['words', 'uppercase', 'uppercase_per_word', 'punctuation', 'punctuation_per_word',
                       'numbers', 'numbers_per_word']

    return data, simple_features


def evaluate_model(y_test, y_pred, plot=False, model_name="", features=""):
    """ Evaluates model and plots ROC. """

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred > 0.5)
    print(model_name)
    print("F1 Score : {}".format(round(f1, 3)))
    print("AUC : {}".format(round(roc_auc, 3)))

    if plot:
        # Compute micro-average ROC curve and ROC area
        plt.figure(figsize=(9, 6))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic {} -- {}'.format(model_name, features))
        plt.legend(loc="lower right")
        plt.show()
        print("-----------------------------------------------")
