
'''Script that generates PC predictions and writes their output to a file.'''

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

def prepare_one_record(item, stellar_params_keys, use_shifted_global_view):
        try:
                if use_shifted_global_view:
                        if item['shifted global view'].shape[1] == 0:
                                # print(f'Bailing out on {item["TIC_ID"]}')
                                return None
                        global_view = tf.reshape(tf.sparse.to_dense(item['shifted global view']), [1, 201, 1])
                else:
                        if item['global view'].shape[1] == 0:
                                #print(f'Skipping over record for: {item["kepid"]}')
                                return None
                        global_view = tf.reshape(tf.sparse.to_dense(item['global view']), [1, 201, 1])
        except:
                return None
        local_view = tf.reshape(tf.sparse.to_dense(item['local view']), [1, 81, 1])          
        concat_view = tf.reshape(tf.sparse.to_dense(item['odd_even view']), [1, 162, 1])
        w = 16
        av_label = item['Disposition']        
        batch = {}
        batch['global_view'] = global_view
        batch['local_view'] = local_view
        batch['odd_even_view'] = concat_view
        batch['pred'] = av_label
        if stellar_params_keys is not None:
                for k in stellar_params_keys:
                        t = tf.where(tf.math.is_nan(item[k]), tf.zeros_like(item[k]), item[k])
                        batch[k] = t.numpy()

        return batch


def generate_predictions(args, config, print_output=True):
        batchsize = 1  
        epochs = 1
        model = keras.models.load_model(args.ckpt)
        filenames = get_absolute_path_listings(args.data)
        pprint.pprint(filenames)
        stellar_params = None
        if 'Stellar Features' in config:
            stellar_params = list(config['Stellar Features'])
        predictions_ds = records_io.create_dataset(filenames, stellar_params, batchsize=batchsize, epochs=epochs)
        pcs = []
        count = 0
        skipped = 0
        #try:
        for i in predictions_ds:
                tess_id = i['TIC_ID'].numpy()[0]
                datum = prepare_one_record(i, stellar_params, config['Use Shifted Global View'])
                if datum is None:
                        #print(f'Got null datum')
                        skipped += 1
                        continue
                prediction = model.predict(datum)
                #if tess_id == wanted_tess_id:
                #        print(f'{tess_id} : {prediction}')
                #        break
                count += 1
                #print(f'{i["kepid"].numpy()} : {prediction[0]}')
                if prediction[0] > args.threshold:
                        pcs.append((tess_id, float(prediction[0])))
                # except tf.errors.InvalidArgumentError:
                #         print(f'Hit invalid arguemnt error')
                #         continue
                # except:
                #         print(f'Hit bogus error')
                #         continue
        #except:
        #        pass
        print(f'Count: {count}, Skipped: {skipped}')
        pcs.sort(key=lambda x:x[1], reverse=True)
        #pprint.pprint(pcs)
        if print_output:
                with open(args.output,'w') as out:
                        csv_out=csv.writer(out)
                        csv_out.writerow(['tic_id','Probability'])
                        csv_out.writerows(pcs)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Generate PC predictions for a given sector")
        parser.add_argument("--ckpt", type=str, required=True, help="Path to load the model checkpoint")
        parser.add_argument("--data", type=str, required=True, help="Path to files with .tfRecords for prediction.")
        parser.add_argument("--output", type=str, required=True, help="Output file where PC probabilities are written to.")
        parser.add_argument("--threshold", type=float, required=True, help="Threshold for planet candidates (0 < p < 1)")
        parser.add_argument("--config", type=str, required=True, help="Config file")
        args = parser.parse_args()
        with open(args.config) as config_file:
                config = json.load(config_file)
        generate_predictions(args, config)
