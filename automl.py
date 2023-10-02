from __future__ import print_function

import logging
logging.getLogger("matplotlib.font_manager").disabled = True

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import os
import shutil
import pickle

import sklearn.model_selection
import sklearn.metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from tpot import TPOTRegressor
from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

def cleanup(directory):
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

def load_csv(filename, localfile, discardfeatures, header, sep):
    header = 0 if header else None
    separators = {"comma":",", "tab":r"\t", "space":r"\s", "white_spaces":r"\s+","colon":";"}

    if len(filename) == 0:
        data_df = pd.read_csv(localfile, sep=separators.get(sep), header=header, error_bad_lines=False)
    else:
        data_df = pd.read_csv(filename, sep=separators.get(sep), header=header, error_bad_lines=False)

    for feature in discardfeatures:
        if feature > data_df.shape[1]-1:
            raise ValueError('discarded feature index out dataframe dimesion - df.shape[1]')

    data_df.drop(data_df.columns[discardfeatures], axis=1, inplace=True)
    
    verboseprint("Loaded file %s with dimensions %s" % (filename, data_df.shape))
    verboseprint("Data types:\n")
    verboseprint(data_df.dtypes)
    verboseprint(data_df.head())
    
    return data_df

def plot_roc(fpr, tpr, roc_auc, output_dir):
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, 'b', label = "AUC = %0.2f" % roc_auc)
    plt.legend(loc = "lower right")
    plt.plot([0, 1], [0, 1],"r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(output_dir + "_roc.png")


def run_automl(filename, localfile, discardfeatures, header, sep, target, task, missing, test, validation, seed, output_dir, tmp_dir):
    try:
        discardfeatures = eval(discardfeatures)
    except:
        discardfeatures = []
        
    for feature in discardfeatures:
        if feature < 0:
            raise ValueError('discarded features indices should be greater than or equal to 0')

    data_df = load_csv(filename, localfile, discardfeatures, header, sep)

    if target > 0: # if features are discarded target index can change unless its '0' or a '-1'
        for feature in discardfeatures:
            if feature < target:
                target -= 1

    if target < -1:
        for feature in discardfeatures:
            if feature > data_df.shape[1]+target:
                target += 1
    
    object_dt = [ col  for col, dt in data_df.dtypes.items() if dt == object]
    if len(object_dt) > 0:
        verboseprint("Excluding object attributes from data frame:")
        verboseprint(object_dt)
        data_df = data_df.select_dtypes(exclude=['object'])

    if data_df.isnull().values.any():
        verboseprint("NaN's detected.")
        if missing == "infer":
            verboseprint("Replacing NaN's with mean. This only works for numeric attributes (e.g. NaN's in object dtype columns will be ignored).")
            # Caution: This is wrong from ML perspective, as it leaks information about the mean into the test set
            # For proper implementation imputation must take place on the training set alone
            data_df = data_df.fillna(data_df.mean()) 
        else:
            verboseprint("Dropping records with NaN's.")
            data_df = data_df.dropna()
    
    verboseprint("Creating X & y...")
    y = data_df.iloc[:, target]
    #X = data_df.drop(target, axis=1)
    X = data_df.drop(data_df.columns[[target]], axis=1)

    verboseprint("Creating a test set...")
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=seed)
    verboseprint("%i observations in test." % X_test.shape[0])
    
    #cleanup(output)
    
    if task == "regression":
        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
        tpot.fit(X_train, y_train)
        print("Final score: ", tpot.score(X_test, y_test))
        tpot.export(output_dir + "_regression_pipeline.py")
    elif task == "classification":
        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
        tpot.fit(X_train, y_train)
        #print("Final score: ", tpot.score(X_test, y_test))
        tpot.export(output_dir + "_classification_pipeline.py")
        pred = tpot.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        fpr, tpr, threshold = roc_curve(y_test, pred)
        roc_auc = auc(fpr, tpr)
        plot_roc(fpr, tpr, roc_auc, output_dir)
        print("True positive: %i" % tp)
        print("False positive: %i" % fp)
        print("True negative: %i" % tn)
        print("False negative: %i" % fn)
        print("Accuracy score: %.2f" % accuracy_score(y_test, pred))
    else:
        raise Exception("Unsupported task type: %s" % task)

if __name__ == "__main__":

    def restricted_float(x):
        try:
            x = float(x)
        except ValueError:
            raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

        if x <= 0.0 or x >= 1.0:
            raise argparse.ArgumentTypeError("%r not in range (0.0, 1.0)"%(x,))
        return x

    parser = argparse.ArgumentParser(description="AutoML script using auto-sklearn. (C) 2020 Nikolay Manchev, Domino Data Lab")
    parser.add_argument("--file", type=str, required=False, help="a CSV file containing the dataset")
    parser.add_argument("--localfile", type=str, required=False, help="path to local DFS file",default="/mnt/data/raw/heart.csv")
    parser.add_argument("--discardfeatures",type=str,required=False,help="features (list of indices) to discard in training",default=[])  
    parser.add_argument("--target", type=int, required=True, help="target variable column index")
    parser.add_argument("--task", type=str, required=True, help="task type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--verbose", type=bool, required=False, help="output additional information", default=True)
    parser.add_argument("--header", type=bool, required=False, help="the first row contains a header", default=False)
    parser.add_argument("--sep", type=str, required=False, help="delimiter to use", default="comma", choices=["comma", "tab", "space", "white_spaces","colon"])
    parser.add_argument("--missing", type=str, required=False, help="missing values treatment", choices=["drop", "infer"], default="infer")
    parser.add_argument("--test", type=restricted_float, required=False, help="size of the test test (0,1)", default=0.2)
    parser.add_argument("--valid", type=restricted_float, required=False, help="size of the validation set (0,1)", default=0.2)
    parser.add_argument("--seed", type=int, required=False, help="random seed used for sampling and fitting", default=1234)
    parser.add_argument("--output", type=str, required=False, help="default output directory", default="/mnt/results/myrun")
    parser.add_argument("--tmp", type=str, required=False, help="temporary directory", default="tmp")

    args = parser.parse_args()

    verboseprint = print if args.verbose else lambda *a, **k: None

    # output_dir = os.getcwd() + os.sep + args.output
    output_dir = args.output
    tmp_dir = os.getcwd() + os.sep + args.tmp

    
    run_automl(args.file, args.localfile, args.discardfeatures, args.header, args.sep, args.target, args.task, args.missing, 
               args.test, args.valid, args.seed, output_dir, tmp_dir)