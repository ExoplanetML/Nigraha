'''Script that generates PC predictions using an ensemble predictor and writes their output to a file.'''

from data import records_io
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os, sys
import pprint
import json
from train import get_absolute_path_listings
#import cross_match
import argparse
from predict import prepare_one_record
from collections import defaultdict

def generate_predictions(args, config, print_output=True):
        models = []
        files = os.listdir(args.ckpt)
        print(f'models are: {files}')
        for f in files:
                model = keras.models.load_model(os.path.join(args.ckpt, f))
                models.append(model)
        batchsize = 1  
        epochs = 1
        filenames = get_absolute_path_listings(args.data)
        stellar_params = None
        if 'Stellar Features' in config:
            stellar_params = list(config['Stellar Features'])

        if args.candidates is None:
                candidates = None
        else:
                candidates = pd.read_csv(args.candidates, index_col='tic_id')
        #print(f'number of candidates = len{candidates}')
        if config["Use Shifted Global View"]:
                global_view_type = "shifted"
        else:
                global_view_type = "unshifted"
        print(f'Generating predictions with {global_view_type} view')
        predictions = defaultdict()
        count = 0
        skipped = 0
        for f in files:
                model = keras.models.load_model(os.path.join(args.ckpt, f))
                print(f'Working on {f}')
                predictions_ds = records_io.create_dataset(filenames, stellar_params, batchsize=batchsize, epochs=epochs)
                for record in predictions_ds:
                        tess_id = record['TIC_ID'].numpy()[0]
                        if not candidates is None and not tess_id in candidates.index:
                                continue
                        count += 1
                        datum = prepare_one_record(record, stellar_params, config['Use Shifted Global View'])
                        if datum is None:
                                #print(f'Got null datum for {count}, {tess_id}')
                                skipped += 1
                                continue
                        if count % 100 == 0:
                                print(f'Looked at {count} candidates')
                        if not tess_id in predictions:
                                predictions[tess_id] = []
                        predictions[tess_id].append(model.predict(datum))
        
        if config['Binary Classification']:
                num_outputs = 1
                columns=['tic_id','Probability']
        else:
                num_outputs = 5
                columns=['tic_id','KP_PC', 'EB', 'V', 'IS', 'J']
        df = pd.DataFrame(columns=columns)
        for i in predictions.keys():
                prediction = np.mean(predictions[i], axis=0)[0]
                #if prediction > args.threshold:
                #        pcs.append((i, float(prediction)))
                if num_outputs == 1:
                        if np.isnan(prediction[0]):
                                prediction[0] = 0.0
                        df = df.append({'tic_id' : i, 'Probability' : prediction[0]}, ignore_index=True)
                else:
                        #print(f'{prediction}')
                        df = df.append({'tic_id' : i, 'KP_PC' : prediction[0], 'EB' : prediction[1], 
                                        'V' : prediction[2], 'IS' : prediction[3], 'J' : prediction[4]}, ignore_index=True)

        print(f'Count: {count}, Skipped: {skipped}')
        if print_output:
                df.to_csv(args.output, index_label='rowid')

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Generate PC predictions for a given sector")
        parser.add_argument("--ckpt", type=str, required=True, help="Path to folder containing the checkpoints")
        parser.add_argument("--candidates", type=str, required=False, help="Short list of candidates for ensemble prediction.")
        parser.add_argument("--data", type=str, required=True, help="Path to files with .tfRecords for prediction.")
        parser.add_argument("--output", type=str, required=True, help="Output file where PC probabilities are written to.")
        parser.add_argument("--threshold", type=float, required=True, help="Threshold for planet candidates (0 < p < 1)")
        parser.add_argument("--config", type=str, required=True, help="Config file")
        args = parser.parse_args()
        with open(args.config) as config_file:
                config = json.load(config_file)
        generate_predictions(args, config)
