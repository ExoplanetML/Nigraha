
'''Script that can be used to evaluate a trained model'''

from data import records_io
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os, sys
import pprint
import csv, json
from train import get_absolute_path_listings
from predict import prepare_one_record
import argparse
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

def evaluate(args, config):
        batchsize = 1  
        epochs = 1
        print(f'Working on model: {args.ckpt}')
        model = keras.models.load_model(args.ckpt)
        filenames = get_absolute_path_listings(args.test)
        stellar_params = None
        if 'Stellar Features' in config:
            stellar_params = list(config['Stellar Features'])
        test_ds = records_io.create_dataset(filenames, stellar_params, batchsize=batchsize, epochs=epochs)
        count = 0
        skipped = 0
        num_positive = 0
        y_pred = []
        y_actual = []
        y_pred_probs = []

        #try:
        for i in test_ds:
                tess_id = i['TIC_ID'].numpy()[0]
                datum = prepare_one_record(i, stellar_params)
                if datum is None:
                        #print(f'Got null datum')
                        skipped += 1
                        continue
                # gah!
                s = datum['pred'].numpy()[0]
                v = s.decode('utf-8')
                y_actual.append(config['Disposition'][v])
                if (config['Disposition'][v]):
                    num_positive += 1

                prediction = model.predict(datum)
                count += 1
                #print(f'{tess_id} : {prediction[0]}')
                if np.isnan(prediction[0]):
                    prediction[0] = 0.0
                if prediction[0] > args.threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                y_pred_probs.append(prediction[0])

        print(f'{skipped}, {count}, {num_positive}')
        # confusion matrix
        matrix = confusion_matrix(y_actual, y_pred)
        print('Confusion matrix:')
        print(matrix)
        accuracy = accuracy_score(y_actual, y_pred)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(y_actual, y_pred)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(y_actual, y_pred)
        print('Recall: %f' % recall)
        # average precision
        avg_precision = average_precision_score(y_actual, y_pred)
        print('Average precision: %f' % avg_precision)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_actual, y_pred)
        print('F1 score: %f' % f1)
        
        # kappa
        # kappa = cohen_kappa_score(y_actual, y_pred)
        # print('Cohens kappa: %f' % kappa)
        # AUC
        precision, recall, _ = precision_recall_curve(y_actual, y_pred_probs)
        auc_score = auc(recall, precision)
        print('AUC: %f' % auc)

        
if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Evaluate a DL model for detecting PC/EB or PC")
        parser.add_argument("--ckpt", type=str, required=True, help="Path to load the model checkpoint")
        parser.add_argument("--test", type=str, required=True, help="Path to files where .tfRecords for test are.")
        parser.add_argument("--config", type=str, required=True, help="Config file")
        parser.add_argument("--threshold", type=float, required=True, help="Threshold for planet candidates (0 < p < 1)")
        args = parser.parse_args()
        with open(args.config) as config_file:
            config = json.load(config_file)
        evaluate(args, config)
