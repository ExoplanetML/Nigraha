from __future__ import division, print_function
from astropy.io import fits
from astropy.stats import sigma_clip
import numpy
import astropy
import os,sys,math,multiprocessing
import pprint
from tqdm import tqdm
import pandas as pd
import argparse

from wotan import (flatten, t14)
from transitleastsquares import (
    transitleastsquares,
    cleaned_array,
    catalog_info,
    transit_mask
    )

global ld
global _BASE_PATH

def load_claret_tess_info():
    global ld
    ld = numpy.genfromtxt(
            os.path.join("../lcs/", "ld_claret_tess.csv"),
            skip_header=1,
            delimiter=",",
            dtype="f8, int32, f8, f8",
            names=["logg", "Teff", "a", "b"],
        )

def get_limb_darkening_params(Teff, logg):
    '''Code snagged from TransitLeastSquares.  The call catlog_info() makes a REST call to get Teff, logg.  We
    got all that data in the file.'''
    global ld
    nearest_Teff = ld["Teff"][(numpy.abs(ld["Teff"] - Teff)).argmin()]
    idx_all_Teffs = numpy.where(ld["Teff"] == nearest_Teff)
    relevant_lds = numpy.copy(ld[idx_all_Teffs])
    idx_nearest = numpy.abs(relevant_lds["logg"] - logg).argmin()
    a = relevant_lds["a"][idx_nearest]
    b = relevant_lds["b"][idx_nearest]
    return (a, b)

def isTCE(tic, priors):
    if priors['NumTransits'][tic] < 2 or priors['snr'][tic] < 7.1 or priors['sde'][tic] < 7.0:
            return False
    return True

def compute_unconstrained(TIC_ID, time, flux):
    '''Compute period without constraining TLS on mass/radius of the TIC'''
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=TIC_ID)
    if numpy.isnan(mass) or numpy.isnan(radius):
        flatten_lc, trend_lc = flatten(time, flux, window_length=0.5, method='biweight', return_trend=True)
        flux = sigma_clip(flatten_lc, sigma_upper=3, sigma_lower=float('inf'))
    else:
        # Take the max period for a TESS sector: 27 days / 2 -> we need at least 2 transits
        period = 13.5
        tdur = t14(R_s=radius, M_s=mass, P=period, small_planet=False)
        flatten_lc, trend_lc = flatten(time, flux, window_length=3*tdur, method='biweight', return_trend=True)
        flux = sigma_clip(flatten_lc, sigma_upper=3, sigma_lower=float('inf'))

    if numpy.mean(flux) > 1.01 or numpy.mean(flux) < 0.99:
        # Normalize by the mean if needed
        flux = flux / numpy.mean(flux)
    model = transitleastsquares(time, flux)
    #results = model.power(u=ab, oversampling_factor=5)
    results = model.power(u=ab)
    return ab, mass, radius, results

def compute_constrained(TIC_ID, time, flux):
    '''Compute period with constraining TLS on mass/radius of the TIC'''
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=TIC_ID)
    if numpy.isnan(mass) or numpy.isnan(radius):
        flatten_lc, trend_lc = flatten(time, flux, window_length=0.5, method='biweight', return_trend=True)
        flux = sigma_clip(flatten_lc, sigma_upper=3, sigma_lower=float('inf'))
        if numpy.mean(flux) > 1.01 or numpy.mean(flux) < 0.99:
            # Normalize by the mean if needed
            flux = flux / numpy.mean(flux)
        model = transitleastsquares(time, flux)
        results = model.power(u=ab, oversampling_factor=5)
    else:
        period = 13.5
        tdur = t14(R_s=radius, M_s=mass, P=period, small_planet=False)
        flatten_lc, trend_lc = flatten(time, flux, window_length=3*tdur, method='biweight', return_trend=True)
        flux = sigma_clip(flatten_lc, sigma_upper=3, sigma_lower=float('inf'))
        if numpy.mean(flux) > 1.01 or numpy.mean(flux) < 0.99:
            # Normalize by the mean if needed
            flux = flux / numpy.mean(flux)
        model = transitleastsquares(time, flux)
        mstar_min = mass - 3 * mass_min
        if mstar_min < 1e-3:
            mstar_min = 0.0
        rstar_min = radius - 3 * radius_min
        if rstar_min < 1e-3:
            rstar_min = 0.0
        results = model.power(u=ab, M_star_min=mstar_min, M_star=mass, M_star_max=mass+3*mass_max, 
                        R_star_min=rstar_min, R_star=radius, R_star_max=radius+3*radius_max, 
                        oversampling_factor=5)
    return ab, mass, radius, results

def do_work(tic2file, priorFn, outputFn):
    global _BASE_PATH
    tic2fileDB = pd.read_csv(tic2file, index_col='tic_id')
    tic2fileDB = tic2fileDB[~tic2fileDB.index.duplicated(keep='first')]
    priors = pd.read_csv(priorFn, index_col='TIC_ID')
    priors = priors[~priors.index.duplicated(keep='first')]
    fh = open(outputFn, "a")
    skipping = True
    print('TIC_ID,T0,Depth,Period,Period_Uncertainity,Duration,TMag,Teff,Radius,NumTransits,a,b,logg,rp_rs,DepthEven,DepthOdd,odd_even_mismatch,snr,sde',file=fh)
    for tic in tqdm(priors.index):
        if not isTCE(tic, priors):
            continue

        # if skipping and tic != 357623383:
        #     continue

        skipping = False
        f = tic2fileDB['Filename'][tic]
        filename = os.path.join(_BASE_PATH, f)
        if not os.path.isfile(filename):
            continue
        #if priors['Disposition'][tic] != 'PC':
            # For DL3 tic's, for now, only recompute for PCs.
        #    continue
        try:
            with fits.open(filename) as hdu:
                    #pprint.pprint(hdu[0].header)
                    #print(f'{hdu[0].header["TESSMAG"]]},{hdu[0].header["TEFF"]}')
                    tessmag = hdu[0].header["TESSMAG"]
                    teff = hdu[0].header["TEFF"]
                    logg = hdu[0].header["LOGG"]
                    time = hdu[1].data['TIME']
                    flux = hdu[1].data['PDCSAP_FLUX']  # values with non-zero quality are nan or zero'ed
                    #flux = flux / numpy.median(flux)
                    TIC_ID = hdu[0].header['TICID']
                    radius = hdu[0].header['RADIUS']

                    print(f'Processing: {tic} with {teff}, {logg}, {radius}')

                    if teff is None:
                        teff = 6000
                    if logg is None:
                        logg = 4
                    if radius is None:
                        radius = 1.0

                    # With LightKurve's flatten, we get savigol filter.  But, the implementation with
                    # Lightkurve and wotan are different, and so, we will use biweight here, get the period
                    # and that period goes into flatten pathway with Lightkurve.  Mixing and matching
                    # between LightKurve and wotan doesn't quite work....
                    # Yes, I know, messy...
                    # flatten_lc, trend_lc = flatten(
                    #     time,                 # Array of time values
                    #     flux,                 # Array of flux values
                    #     method='savgol',
                    #     cval=2,               # Defines polyorder
                    #     window_length=101,     # The window length in cadences
                    #     break_tolerance=5.0,  # Split into segments at breaks longer than that
                    #     return_trend=True,    # Return trend and flattened light curve
                    #     )

                    # remove invalid values such as nan, inf, non, negative
                    time, flux = cleaned_array(time, flux)  

                    hdu.close()

                    ab, mass, radius, results = compute_unconstrained(TIC_ID, time, flux)
                    
                    print(f'{TIC_ID},{results.T0},{results.depth},{results.period},{results.period_uncertainty},{results.duration * 24},{tessmag},{teff},{radius},{results.distinct_transit_count},{ab[0]},{ab[1]},{logg},{results.rp_rs},{results.depth_mean_even[0]},{results.depth_mean_odd[0]},{results.odd_even_mismatch},{results.snr},{results.SDE}',file=fh)
                    fh.flush()
        except (OSError, TypeError, ValueError) as inst:
            print(inst)
            print(f'Failed for {tic}: {type(inst)},  R_star = {radius}, snr = {priors["snr"][tic]}, sde = {priors["sde"][tic]}')
            #time, flux = cleaned_array(time, flux) 
            #print(f'{time}')
            pass
            #raise


if __name__ == '__main__':
    global _BASE_PATH

    parser = argparse.ArgumentParser(description="Script for computing period info including detrending")
    parser.add_argument("--tic2fileDB", type=str, required=True, help="File that maps TIC ID to associated .fits file")
    parser.add_argument("--input", type=str, required=True, help="Input folder where the .fits file are")
    parser.add_argument("--prior", type=str, required=True, help="Input TIC Catalog (such as, period_info-sec25.csv)")
    parser.add_argument("--output", type=str, required=True, help="Output TIC Catalog (such as, updated_info-sec25.csv)")
    args = parser.parse_args()
    _BASE_PATH = args.input
    load_claret_tess_info()
    do_work(args.tic2fileDB, args.prior, args.output)

