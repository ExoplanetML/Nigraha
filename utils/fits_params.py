
from __future__ import division, print_function

import numpy as np
import pandas as pd
import pprint
from tqdm import tqdm
from astropy.io import fits
import argparse
import os

def do_work(args):
    df = pd.read_csv(args.catalog)
    tic2file = pd.read_csv(args.tic2fileDB, index_col='tic_id')
    camera = [np.nan] * len(df)
    ccd = [np.nan] * len(df)
    ra = [np.nan] * len(df)
    dec = [np.nan] * len(df)
    pmra = [np.nan] * len(df)
    pmdec = [np.nan] * len(df)
    pmtotal = [np.nan] * len(df)
    mh = [np.nan] * len(df)
    count = 0
    for i in tqdm(range(len(df))):
        tic_id = int(df['TIC_ID'][i])
        if not tic_id in tic2file.index:
                continue
        filename = os.path.join(args.input, tic2file['Filename'][tic_id])
        hdu = fits.open(filename)
        camera[i] = hdu[0].header['CAMERA']
        ccd[i] = hdu[0].header['CCD']
        ra[i] = float(hdu[0].header['RA_OBJ'])
        dec[i] = float(hdu[0].header['DEC_OBJ'])
        try:
                pmra[i] = float(hdu[0].header['PMRA'])
                pmdec[i] = float(hdu[0].header['PMDEC'])
                pmtotal[i] = float(hdu[0].header['PMTOTAL'])
                mh[i] = float(hdu[0].header['MH'])
        except:
                pass

    df['Camera'] = camera
    df['CCD'] = ccd
    df['RA'] = ra
    df['DEC'] = dec
    df['PMRA'] = pmra
    df['PMDEC'] = pmdec
    df['PMTOTAL'] = pmtotal
    df['MH'] = mh
    df.to_csv(args.output)


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Script for loading in stellar params from FITS files")
        parser.add_argument("--input", type=str, required=True, help="Folder containing the .fits files")
        parser.add_argument("--tic2fileDB", type=str, required=True, help="File that maps TIC ID to associated .fits file")
        parser.add_argument("--catalog", type=str, required=True, help="TIC Catalog (such as, period_info-sec24.csv)")
        parser.add_argument("--output", type=str, required=True, help="Updated TIC catalog output (such as, new-sec24.csv)")
        args = parser.parse_args()

        if args.catalog == args.output:
                print(f'Error: Input {args.catalog} is same as output: {args.output}')
        else:
                do_work(args)

