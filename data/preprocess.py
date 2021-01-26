import random
from astropy.io import fits
from astropy.table import Table 
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from collections import defaultdict
from tqdm import tqdm
import multiprocessing
import math
import warnings
import pprint
import argparse
import wotan
from astropy.stats import sigma_clip

with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import lightkurve as lk

_BASE_PATH = ''
tic2file = None
tic_catalog = None
file2tic = None
exclude_tics = []

#
# Patterned after samples for building records in TF documentation.
#
def _set_float_feature(ex, name, value):
        """Sets the value of a float feature in a tensorflow.train.Example proto."""
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].float_list.value.extend([float(v) for v in value])

def _set_bytes_feature(ex, name, value):
        """Sets the value of a bytes feature in a tensorflow.train.Example proto."""
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].bytes_list.value.extend([str(v).encode("utf-8") for v in value])


def _set_int64_feature(ex, name, value):
        """Sets the value of an int64 feature in a tensorflow.train.Example proto."""
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].int64_list.value.extend([int(v) for v in value])

def load_tic2file(filename, exclude_filename = ""):
        global tic2file, file2tic, exclude_tics
        if not os.path.isfile(filename):
                raise FileNotFoundError
        tic2file = pd.read_csv(filename, index_col='tic_id')
        # In case we have multiple curves for the same known TIC
        tic2file = tic2file[~tic2file.index.duplicated(keep='first')]
        file2tic = pd.read_csv(filename, index_col='Filename')
        if exclude_filename != "":
                excludes = pd.read_csv(exclude_filename, index_col='tic_id')
                excludes = excludes[~excludes.index.duplicated(keep='first')]
                exclude_tics = list(excludes.index)
                print(f'Number of tics to exclude: {len(exclude_tics)}')
        return tic2file, file2tic

def print_stats():
        global tic_catalog
        counts = defaultdict()
        for f in list(file2tic.index):
                tic = file2tic['tic_id'][f]
                if tic in exclude_tics:
                        continue
                if not tic in tic_catalog.index:
                        # pprint.pprint(f'Missing {tic}')
                        continue
                d = tic_catalog['Disposition'][tic]
                if not d in counts.keys():
                        counts[d] = 0
                counts[d] = counts[d] + 1
        pprint.pprint(counts)
        print(f'Total tces: {sum(counts.values())}')


def load_catalog(filename, enableImputation = False):
        global tic_catalog
        if not os.path.isfile(filename):
                raise FileNotFoundError
        print(f'Loading catalog {filename}')

        tic_catalog = pd.read_csv(filename, index_col='TIC_ID')
        keys = ['T0', 'Depth', 'Period', 'Duration', 'TMag', 'Teff', 'Radius', 'NumTransits', 'snr', 'sde',
                'Mass', 'a', 'b', 'logg', 'distance', 'lum', 'rho', 'rp_rs', 
                'DepthOdd', 'DepthEven', 'Disposition', 'tdur']
        raw_columns = ['T0', 'Depth', 'Period', 'Duration', 'TMag', 'Disposition', 'NumTransits', 'snr', 'sde',
                        'rp_rs', 'DepthOdd', 'DepthEven', 'tdur']
        # Compute t14 for wotan based flatten()
        # Default to using maximum period given TESS 27-day orbit, in which we can get at least 2 transits.
        tdur = wotan.t14(R_s=1.0, M_s=1.0, P=13.5, small_planet=False)
        tic_catalog['tdur'] = [tdur] * len(tic_catalog)
        for tic in tic_catalog.index:
                try:
                        tic_catalog.loc[tic, 'tdur'] = wotan.t14(R_s=tic_catalog['Radius'][tess_id], M_s=tic_catalog['Mass'][tess_id], 
                                                P=tic_catalog['Period'][tess_id], small_planet=False)
                except:
                        pass 

        if 'Logg' in tic_catalog.columns:
                tic_catalog['logg'] = tic_catalog['Logg']

        for col in tic_catalog.columns:
                if not col in keys:
                        tic_catalog = tic_catalog.drop(col, axis=1)

        print(f'Here we go... {len(tic_catalog)} ')
        tic_catalog = tic_catalog[~tic_catalog.index.duplicated(keep='first')]
        #print_stats()
        if not enableImputation:
                tic_catalog = tic_catalog.dropna()

        #print(f'Eliminating and filtering... {len(tic_catalog)} ')
        #print_stats()

        # counts = defaultdict()
        # for tic in list(tic2file.index):
        #         if tic in exclude_tics:
        #                 continue
        #         if not tic in tic_catalog.index:
        #                 # pprint.pprint(f'Missing {tic}')
        #                 continue
        #         d = tic_catalog['Disposition'][tic]
        #         if not d in counts.keys():
        #                 counts[d] = 0
        #         counts[d] = counts[d] + 1
        # pprint.pprint(counts)

        for col in tic_catalog.columns:
                if col in raw_columns:
                        continue
                if tic_catalog[col].max() == 'None':
                        tic_catalog[col] = tic_catalog[col].replace('None', np.nan).astype(float)
                                
                missing_data = sum(tic_catalog[col].isna())
                if missing_data > 0:
                        print(f'Imputing for column {col}')
                        tic_catalog[col].fillna(tic_catalog[col].median(), inplace = True)
                
                # Store the normalized values in the TFRecord
                if -2 < tic_catalog[col].min() and tic_catalog[col].max() < 2:
                        # already normalized; so, skip
                        continue
                tic_catalog[col] = (tic_catalog[col] - tic_catalog[col].median()) / tic_catalog[col].std()

        # print(f'Before calling drop after all the imputing or not {len(tic_catalog)}')
        # print_stats()
        # tic_catalog = tic_catalog.dropna()
        # print(f'Verdict: Uniques {len(tic_catalog)} ')
        # print_stats()
        return tic_catalog

def load_lightcurve(filename):
        lcfs = []
        try:
                lcf = lk.search.open(filename)
                lcfs.append(lcf)
        except (OSError, TypeError):
                return None
        lc_file_collection = lk.LightCurveFileCollection(lcfs)
        lc_raw = lc_file_collection.PDCSAP_FLUX.stitch()
        return lc_raw

def load_lightcurve_from_files(filenames):
        for f in filenames:
                try:
                        lcf = lk.search.open(lc_name)
                        lcfs.append(lcf)
                except (OSError, TypeError):
                        return None
        lc_file_collection = lk.LightCurveFileCollection(lcfs)
        lc_raw = lc_file_collection.PDCSAP_FLUX.stitch()
        return lc_raw

def load_lightcurves(tess_id):
        lcfs = []
        lc_path = _BASE_PATH

        filenames = list(tic2file['Filename'][tess_id])
        for f in filenames:
                lc_name = os.path.join(lc_path, f)
                print(f'Opening {tess_id}, {lc_name}')
                if not os.path.exists(lc_name):
                        continue
                #print(f'Opening {lc_name}')
                try:
                        lcf = lk.search.open(lc_name)
                        lcfs.append(lcf)
                except (OSError, TypeError):
                        return None
        if len(lcfs) == 0:
                return None
        lc_file_collection = lk.LightCurveFileCollection(lcfs)
        lc_raw = lc_file_collection.PDCSAP_FLUX.stitch()
        return lc_raw

def get_folded_lightcurve(tess_id, lc_raw, period, t0, duration_hours):
        '''Use lightkurve's flatten() to detrend the lightcurve.  Aside, we did try to use
        wotan's flatten and the mix-n'-match with LK and wotan didnt work.  So, LK it is here...'''
        try:
                #lc_clean = lc_raw.remove_nans()
                lc_clean = lc_raw.remove_outliers(sigma=20, sigma_upper=5)
                temp_fold = lc_clean.fold(period, t0=t0)
                fractional_duration = (duration_hours / 24.0) / period
                phase_mask = np.abs(temp_fold.phase) < (fractional_duration * 1.5)
                transit_mask = np.in1d(lc_clean.time, temp_fold.time_original[phase_mask])
                lc_flat, trend_lc = lc_clean.flatten(return_trend=True, mask=transit_mask)
                lc_fold = lc_flat.fold(period, t0=t0)
                return lc_fold
        except ValueError:
                return None

def process_lightcurve(tess_id, lc_raw, period, t0, duration_hours):
        #
        # Given a raw lightcurve, convert that to global/local views.  
        # The code is based on tutorials at:
        #    https://docs.lightkurve.org/tutorials/05-advanced_patterns_binning.html
        #
        try:
                lc_fold = get_folded_lightcurve(tess_id, lc_raw, period, t0, duration_hours)
                if lc_fold is None:
                        return None, None, None

                #lc_fold = lc_fold.fill_gaps()
                # When the lightcurves stitched together, they are normalized.  Trying to normalize here generates warning.
                #lc_global = lc_fold.bin(bins=2001, method='median').normalize() - 1
                lc_global = lc_fold.bin(bins=201, method='median') - 1
                lc_global = (lc_global / np.abs(lc_global.flux.min()) ) * 2.0 + 1
                lc_global = lc_global.remove_nans()
                fractional_duration = (duration_hours / 24.0) / period
                phase_mask = (lc_fold.phase > -2.0*fractional_duration) & (lc_fold.phase < 2.0*fractional_duration)
                lc_zoom = lc_fold[phase_mask]
                #lc_local = lc_zoom.bin(bins=201, method='median').normalize() - 1
                # We tried 61; 81 seems better as there are more bins.
                lc_local = lc_zoom.bin(bins=81, method='median') - 1
                lc_local = (lc_local / np.abs(lc_local.flux.min()) ) * 2.0 + 1
                lc_local = lc_local.remove_nans()
                x = lc_local.to_pandas()
                if len(x) == 0:
                        return None, None, None
                #
                # Shift the global view by 0.25 so that secondary is seen in a view.
                #
                lc_fold = get_folded_lightcurve(tess_id, lc_raw, period, t0 + 0.25 * period, duration_hours)
                lc_global_shifted = lc_fold.bin(bins=201, method='median') - 1
                lc_global_shifted = (lc_global_shifted / np.abs(lc_global_shifted.flux.min()) ) * 2.0 + 1
                lc_global_shifted = lc_global_shifted.remove_nans()
                return lc_global, lc_local, lc_global_shifted
        except ValueError:
                return None, None, None

def _process_lightcurve(tess_id, lc_raw):
        global tic_catalog
        if tess_id not in tic_catalog.index:
            return None, None, None
        try:
                period = tic_catalog['Period'][tess_id]
                duration_hours = tic_catalog['Duration'][tess_id]
                t0 = tic_catalog['T0'][tess_id]
                return process_lightcurve(tess_id, lc_raw, period, t0, duration_hours)
        except ValueError:
                return None, None, None


def build_halfphase_views(tess_id, lc_raw):
        '''Generate zoom ins by folding around t0, t0-period, t0-period/2.  The objective here
        is to zoom on odd/even transits and the mid-point of the transit.'''
        try:
                period = tic_catalog['Period'][tess_id]
                duration_hours = tic_catalog['Duration'][tess_id]
                t0 = tic_catalog['T0'][tess_id]
                lc_fold = get_folded_lightcurve(tess_id, lc_raw, period, t0, duration_hours)
                bins = 81
                fractional_duration = (duration_hours / 24.0) / period
                phase_mask = (lc_fold.phase > -2.0*fractional_duration) & (lc_fold.phase < 2.0*fractional_duration)
                lc_zoom = lc_fold[phase_mask]
                phase_fold_t0 = lc_zoom.bin(bins=bins, method='median') - 1

                lc_fold = get_folded_lightcurve(tess_id, lc_raw, period, t0 - 0.5 * period, duration_hours)
                phase_mask = (lc_fold.phase > -2*fractional_duration) & (lc_fold.phase < 2.0*fractional_duration)
                lc_zoom = lc_fold[phase_mask]
                phase_fold_half = lc_zoom.bin(bins=bins, method='median') - 1
                lc_phase = combine_odd_even(phase_fold_half, phase_fold_t0)

                missing = np.sum(np.isnan(lc_phase.flux))
                # Fill in NaN's using neighbor values...
                if missing > 0:
                        mask = np.isnan(lc_phase.flux)
                        lc_phase.flux[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), 
                                                                lc_phase.flux[~mask])
                return lc_phase
        except:
                return None

def combine_odd_even(lc_odd_zoom, lc_even_zoom):
        concat_view_t = np.concatenate((lc_odd_zoom.time, lc_even_zoom.time + (2.5 * np.max(lc_odd_zoom.time))))
        concat_view_f = np.concatenate((lc_odd_zoom.flux, lc_even_zoom.flux))
        concat_lc = lk.LightCurve(concat_view_t, concat_view_f)
        concat_lc = (concat_lc / np.abs(concat_lc.flux.min()) ) * 2.0 + 1
        return concat_lc

                
def split_train_lcs(num_workers, lcs, output_dir, basename = 'train'):
        splits = []
        split_size = int(math.ceil(len(lcs) / num_workers))
        #
        # Split them into positive and negative examples
        #
        #print(tic_catalog.index)
        positives, negatives = [], []
        for lc in lcs:
                tess_id = file2tic['tic_id'][lc]
                if tess_id in exclude_tics or not tess_id in tic_catalog.index:
                        continue
                if tic_catalog['Disposition'][tess_id] in ['KP', 'PC', 'CP']:
                        positives.append(lc)
                else:
                        negatives.append(lc)

        num_positive_workers = int(math.ceil(len(positives) / len(lcs) * num_workers))
        split_size = int(math.ceil(len(positives) / num_positive_workers))
        print(f'Split size: {split_size}, positives = {len(positives)}, workers = {num_positive_workers}')
        for i in range(num_positive_workers):
                output_file = "{}-{}-part-{:05d}-of-{:05d}.tfRecords".format(basename, 'positive', i, num_workers)
                splits.append((positives[i * split_size: (i + 1) * split_size], output_dir, output_file))
        num_negative_workers = int(math.ceil(len(negatives) / len(lcs) * num_workers))
        split_size = int(math.ceil(len(negatives) / num_negative_workers))
        print(f'Split size: {split_size}, negatives = {len(negatives)}, workers = {num_negative_workers}')
        for i in range(num_negative_workers):
                output_file = "{}-{}-part-{:05d}-of-{:05d}.tfRecords".format(basename, 'negative', i, num_workers)
                splits.append((negatives[i * split_size: (i + 1) * split_size], output_dir, output_file))

        # print(splits)
        return splits

def create_train_test(lcs):
        random.seed(318)
        random.shuffle(lcs)
        split_1 = int(0.9 * len(lcs))
        train_lcs = lcs[:split_1]
        test_lcs = lcs[split_1:]    
        print(f'train: {len(train_lcs)}, test: {len(test_lcs)}')
        return train_lcs, test_lcs

def create_train_test_val(lcs):
        random.seed(318)
        random.shuffle(lcs)
        split_1 = int(0.8 * len(lcs))
        split_2 = int(0.9 * len(lcs))
        train_lcs = lcs[:split_1]
        test_lcs = lcs[split_1:split_2] 
        val_lcs = lcs[split_2:] 
        print(f'train: {len(train_lcs)}, test: {len(test_lcs)}, val: {len(val_lcs)}')
        return train_lcs, test_lcs, val_lcs

def write_records(lcs, output_dir, output_file):
        global file2tic, exclude_tics
        count = 0
        output_path = os.path.join(output_dir, output_file)
        print(f'Writing out to: {output_path}')
        with tf.io.TFRecordWriter(output_path) as writer:
                #for tess_id in tqdm(lcs):
                for lc in tqdm(lcs):
                        #if tces[kep_id]['av_training_set'] != 'PC':
                                # for now, only take the positive examples.
                        #        continue
                        tess_id = file2tic['tic_id'][lc]
                        if tess_id in exclude_tics:
                                continue
                        #lc = tic2file['Filename'][tess_id]
                        lc_raw = load_lightcurve(os.path.join(_BASE_PATH, lc))
                        if lc_raw is None:
                                print(f'Unable to find lc for {tess_id}')
                                continue
                        lc_global, lc_local, lc_global_shifted = _process_lightcurve(tess_id, lc_raw)
                        if lc_local is None:
                                continue
                        try:
                                lc_phase = build_halfphase_views(tess_id, lc_raw)
                        except ValueError:
                                print(f'Unable to build secondary view for {tess_id}; skipping')
                                continue

                        if lc_phase is None:
                                continue

                        if tic_catalog['Disposition'][tess_id] == 'PC':
                                count += 1
                        metadata = tic_catalog.loc[tess_id]
                        ex = tf.train.Example()
                        _set_int64_feature(ex, 'TIC_ID', [tess_id])
                        _set_float_feature(ex, 'global view', lc_global.flux.astype(float))
                        _set_float_feature(ex, 'local view', lc_local.flux.astype(float))
                        _set_float_feature(ex, 'shifted global view', lc_global_shifted.flux.astype(float))
                        _set_float_feature(ex, 'odd_even view', lc_phase.flux.astype(float))
                        _set_bytes_feature(ex, 'Disposition', [metadata['Disposition']])
                        for k, v in metadata.items():
                                if k == 'Disposition':
                                        continue
                                _set_float_feature(ex, k, [v])
                        writer.write(ex.SerializeToString())
        print(f'Processed {len(lcs)} of which {count} were PCs')

def do_work(basename, output_dir):
        # If you do just train/test split, then uncomment the following line
        # and comment out the lines with val_lcs
        #train_lcs, test_lcs = create_train_test(list(file2tic.index))
        train_lcs, test_lcs, val_lcs = create_train_test_val(list(file2tic.index))

        num_splits = 8
        num_workers = num_splits + 1
        splits = split_train_lcs(num_splits, train_lcs, os.path.join(output_dir, 'train'), basename)
        splits.append((test_lcs, os.path.join(output_dir, 'test'), basename + '-test.tfRecords'))
        # If we just do train/test, then comment out this line
        splits.append((val_lcs, os.path.join(output_dir, 'validate'), basename + '-val.tfRecords'))
        workers = multiprocessing.Pool(processes=len(splits))
        async_results = [workers.apply_async(write_records, s) for s in splits]
        workers.close()
        for async_result in async_results:
                async_result.get()

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Script for generating records for training purposes.")
        parser.add_argument("--catalog", type=str, required=True, help="TIC Catalog (such as, period_info-dl3.csv)")
        parser.add_argument("--tic2fileDB", type=str, required=True, help="File that maps TIC ID to associated .fits file")
        parser.add_argument("--input", type=str, required=True, help="Input folder where the .fits file are")
        parser.add_argument("--output", type=str, required=True, help="Output folder where the .tfRecords will be saved")
        parser.add_argument("--basename", type=str, required=True, help="Base name for .tfRecords (e.g., toi)")
        parser.add_argument("--exclude", type=str, default="", help="TIC2FileDB that lists TICs that should be excluded")
        args = parser.parse_args()

        _BASE_PATH = args.input
        tic2file, file2tic = load_tic2file(args.tic2fileDB, args.exclude)
        # We have decided not to do imputation
        tic_catalog = load_catalog(args.catalog, enableImputation=False)
        do_work(args.basename, args.output)

       
