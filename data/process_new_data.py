import random
from astropy.io import fits
from astropy.table import Table 
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import tensorflow as tf
import csv
from collections import defaultdict
from tqdm import tqdm
import records_io
import multiprocessing
import math
import warnings
import preprocess
import pprint
import pandas as pd
import argparse

with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import lightkurve as lk

_BASE_PATH = ""

tic2file = None
tic_catalog = None
file2tic = None

def _set_float_feature(ex, name, value):
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].float_list.value.extend([float(v) for v in value])

def _set_bytes_feature(ex, name, value):
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].bytes_list.value.extend([str(v).encode("utf-8") for v in value])


def _set_int64_feature(ex, name, value):
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].int64_list.value.extend([int(v) for v in value])

def load_catalog(filename):
        global tic_catalog
        tic_catalog = preprocess.load_catalog(filename)
        return tic_catalog

def load_raw_catalog(filename):
        global tic_catalog
        if not os.path.isfile(filename):
                raise FileNotFoundError
        print(f'Loading catalog {filename}')
        tic_catalog = pd.read_csv(filename, index_col='TIC_ID')
        #tic_catalog = tic_catalog.drop('rowid', axis=1)
        return tic_catalog

def process_lightcurve(tess_id, lc_raw):
    global tic_catalog
    if tess_id not in tic_catalog.index:
            return None, None, None
    period = tic_catalog['Period'][tess_id]
    duration_hours = tic_catalog['Duration'][tess_id]
    t0 = tic_catalog['T0'][tess_id]
    return preprocess.process_lightcurve(tess_id, lc_raw, period, t0, duration_hours)

def get_folded_lightcurve(tess_id, lc_raw):
    global tic_catalog
    if tess_id not in tic_catalog.index:
            return None, None, None
    period = tic_catalog['Period'][tess_id]
    duration_hours = tic_catalog['Duration'][tess_id]
    t0 = tic_catalog['T0'][tess_id]
    return preprocess.get_folded_lightcurve(tess_id, lc_raw, period, t0, duration_hours)

def split_lcs(num_workers, lcs, output_dir, basename = 'predict'):
        splits = []
        split_size = int(math.ceil(len(lcs) / num_workers))
        for i in range(num_workers):
                output_file = "{}-part-{:05d}-of-{:05d}.tfRecords".format(basename, i, num_workers)
                splits.append((lcs[i * split_size: (i + 1) * split_size], output_dir, output_file))
        return splits

def write_record(writer, tess_id, metadata, global_view, local_view, 
                global_view_shifted, concat_view,
                disposition=['UNK']):
        ex = tf.train.Example()
        _set_int64_feature(ex, 'TIC_ID', [tess_id])
        _set_float_feature(ex, 'global view', global_view.flux.astype(float))
        _set_float_feature(ex, 'local view', local_view.flux.astype(float))
        _set_float_feature(ex, 'shifted global view', global_view_shifted.flux.astype(float))
        _set_float_feature(ex, 'odd_even view', concat_view.flux.astype(float))
        _set_bytes_feature(ex, 'Disposition', disposition)
        for k, v in metadata.items():
                _set_float_feature(ex, k, [v])
        writer.write(ex.SerializeToString())

def isTCE(metadata):
        if metadata['NumTransits'] < 2 or metadata['snr'] < 7.1 or metadata['sde'] < 7.0:
        #if metadata['NumTransits'] < 2 or metadata['snr'] < 7.1 or metadata['sde'] < 9.0:
                return False
        return True

def write_records(lcs, output_dir, output_file):
    global tic_catalog, tic2file
    count = 0
    output_path = os.path.join(output_dir, output_file)
    print(f'Input path: {_BASE_PATH}, Writing out to: {output_path}')
    with tf.io.TFRecordWriter(output_path) as writer:
            # for tess_id in tqdm(tic_ids):
            for lc in tqdm(lcs):
                tess_id = file2tic['tic_id'][lc]    
                if not tess_id in tic_catalog.index:
                        continue
                metadata = tic_catalog.loc[tess_id]
                if math.isnan(float(metadata['NumTransits'])) or not isTCE(metadata):
                        continue
                # filename = os.path.join(_BASE_PATH, tic2file['Filename'][tess_id])
                filename = os.path.join(_BASE_PATH, lc)
                lc_raw = preprocess.load_lightcurve(filename)
                if lc_raw is None:
                        print(f'Unable to find lc for {tess_id}')
                        continue
                global_view, local_view, global_view_shifted = process_lightcurve(tess_id, lc_raw)
                #if global_view is None or local_view is None or global_view_shifted is None:
                if global_view is None or local_view is None:
                        print(f'global/local views are none for {tess_id}')
                        continue
                metadata = tic_catalog.loc[tess_id]
                concat_view = preprocess.build_halfphase_views(tess_id, lc_raw)
                if concat_view is None:
                        print(f'half phase views are none for {tess_id}')
                        continue
                write_record(writer, tess_id, metadata, global_view, local_view, global_view_shifted, concat_view)
                count += 1
    print(f'Wrote out {count} records')
                                
if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Script for generating records for prediction purposes.")
        parser.add_argument("--tic2fileDB", type=str, required=True, help="File that maps TIC ID to associated .fits file")
        parser.add_argument("--catalog", type=str, required=True, help="TIC Catalog (such as, period_info-sec24.csv)")
        parser.add_argument("--input", type=str, required=True, help="Input folder where the .fits file are")
        parser.add_argument("--output", type=str, required=True, help="Output folder where the .tfRecords will be saved")
        parser.add_argument("--basename", type=str, default="predict", help="Base name")
        args = parser.parse_args()
        _BASE_PATH = args.input

        tic2file, file2tic = preprocess.load_tic2file(args.tic2fileDB)
        tic_catalog = load_catalog(args.catalog)

        num_workers = 8
        # splits = preprocess.split_train_lcs(num_workers, list(tic2file.index), args.output, basename=args.basename)
        splits = split_lcs(num_workers, list(file2tic.index), args.output, basename=args.basename)
        workers = multiprocessing.Pool(processes=num_workers)
        async_results = [workers.apply_async(write_records, s) for s in splits]
        workers.close()
        for async_result in async_results:
                async_result.get()
