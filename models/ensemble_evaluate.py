'''Script that evaluates model on test set using an ensemble predictor and writes their output to a file.'''

from data import records_io
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os, sys
import pprint
import csv, json
from train import get_absolute_path_listings
#import cross_match
import argparse
from predict import prepare_one_record
from collections import defaultdict
from prune import (index_to_disposition, all_disposition_to_index)

def generate_predictions(args, config, print_output=True):
        models = []
        files = os.listdir(args.ckpt)
        print(f'models are: {files}')
        for f in files:
                model = keras.models.load_model(os.path.join(args.ckpt, f))
                models.append(model)
        batchsize = 1  
        epochs = 1
        filenames = get_absolute_path_listings(args.test)
        stellar_params = None
        if 'Stellar Features' in config:
            stellar_params = list(config['Stellar Features'])
        predictions_ds = records_io.create_dataset(filenames, stellar_params, batchsize=batchsize, epochs=epochs)
        count = 0
        skipped = 0
        num_positive = 0
        results = []
        #try:
        if config['Binary Classification']:
                is_binary_classifier = True
        else:
                is_binary_classifier = False
        
        for i in predictions_ds:
                tess_id = i['TIC_ID'].numpy()[0]
                datum = prepare_one_record(i, stellar_params, config['Use Shifted Global View'])
                if datum is None:
                        #print(f'Got null datum')
                        skipped += 1
                        continue
                count += 1
                predictions = []
                for i in range(len(models)):
                        predictions.append(models[i].predict(datum))
                prediction = np.mean(predictions, axis=0)
                # Grab the label and actual value
                s = datum['pred'].numpy()[0]
                v = s.decode('utf-8')
                if (config['Disposition'][v]):
                    num_positive += 1

                count += 1
                #print(f'{tess_id} : {prediction[0]}')
                
                if is_binary_classifier:
                        if np.isnan(prediction):
                                prediction = 0.0
                        results.append((tess_id, v, prediction, config['Disposition'][v]))
                else:
                        pred_class = np.argmax(prediction)
                        results.append((tess_id, v, pred_class, all_disposition_to_index[v]))
                #if prediction > 0.05:
                #        print(f'{tess_id} : {prediction} is average of {predictions}')

        print(f'Count: {count}, Skipped: {skipped}')
        #pprint.pprint(pcs)
        if print_output:
                with open(args.output,'w') as out:
                        csv_out=csv.writer(out)
                        csv_out.writerow(['TIC_ID','Disposition','y_hat','y'])
                        csv_out.writerows(results)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Evaluate test set using ensemble predictor")
        parser.add_argument("--ckpt", type=str, required=True, help="Path to load the model checkpoint")
        parser.add_argument("--test", type=str, required=True, help="Path to files where .tfRecords for test are.")
        parser.add_argument("--config", type=str, required=True, help="Config file")
        parser.add_argument("--output", type=str, required=True, help="Output file to store <y_hat, y> (e.g., test-output.csv)")
        args = parser.parse_args()
        with open(args.config) as config_file:
                config = json.load(config_file)
        generate_predictions(args, config)
