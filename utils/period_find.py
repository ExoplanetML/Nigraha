from __future__ import division, print_function
from astropy.io import fits
from astropy.stats import sigma_clip
import numpy
import astropy
import os,sys,math,multiprocessing
import pprint
from tqdm import tqdm
import argparse

from transitleastsquares import (
    transitleastsquares,
    cleaned_array,
    catalog_info,
    transit_mask
    )

global ld

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


def part_worker(path, filenames, output_file):
    #filename = 'sector1/tess2018206045859-s0001-0000000025155310-0120-s_lc.fits'
    fh = open(output_file, "w")
    print('TIC_ID,T0,Depth,Period,Period_Uncertainity,Duration,TMag,Teff,Radius,NumTransits,a,b,logg,rp_rs,DepthEven,DepthOdd,odd_even_mismatch,snr,sde',file=fh)
    for f in tqdm(filenames):
        filename = os.path.join(path, f)
        if not os.path.isfile(filename):
            continue
        try:
            with fits.open(filename) as hdu:
                    #pprint.pprint(hdu[0].header)
                    #print(f'{hdu[0].header["TESSMAG"]]},{hdu[0].header["TEFF"]}')
                    tessmag = hdu[0].header["TESSMAG"]
                    teff = hdu[0].header["TEFF"]
                    logg = hdu[0].header["LOGG"]
                    time = hdu[1].data['TIME']
                    flux = hdu[1].data['PDCSAP_FLUX']  # values with non-zero quality are nan or zero'ed
                    time, flux = cleaned_array(time, flux)  # remove invalid values such as nan, inf, non, negative
                    flux = flux / numpy.median(flux)
                    TIC_ID = hdu[0].header['TICID']
                    radius = hdu[0].header['RADIUS']
                    if teff is None:
                        teff = '6000.0'
                    if logg is None:
                        logg = '4.0'
                    if radius is None:
                        radius = '1.0'
                    print('Processing: ', filename)
                    # Avoid this line...it makes a REST call when we have all the data we need.
                    #ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=TIC_ID)
                    ab = get_limb_darkening_params(float(teff), float(logg))
                    model = transitleastsquares(time, flux)
                    # For regular curves, this makes the radius a bit too noisy.  Use for fine-tune?
                    #results = model.power(u=ab, R_star=float(radius))
                    results = model.power(u=ab)
                    print(f'{TIC_ID},{results.T0},{results.depth},{results.period},{results.period_uncertainty},{results.duration * 24},{tessmag},{teff},{radius},{results.distinct_transit_count},{ab[0]},{ab[1]},{logg},{results.rp_rs},{results.depth_mean_even[0]},{results.depth_mean_odd[0]},{results.odd_even_mismatch},{results.snr},{results.SDE}',file=fh)
                    # print('Period', format(results.period, '.5f'), 'd at T0=', results.T0)
                    # print(len(results.transit_times), 'transit times in time series:', ['{0:0.5f}'.format(i) for i in results.transit_times])
                    # print('Number of data points during each unique transit', results.per_transit_count)
                    # print('The number of transits with intransit data points', results.distinct_transit_count)
                    # print('The number of transits with no intransit data points', results.empty_transit_count)
                    # print('Transit depth', format(results.depth, '.5f'), '(at the transit bottom)')
                    # print('Transit duration (days)', format(results.duration, '.5f'))
                    # print('Transit depths (mean)', results.transit_depths)
                    # print('Transit depth uncertainties', results.transit_depths_uncertainties)
                    # print('Mass', mass)
                    # print('Radius', radius)
                    fh.flush()
        except (OSError, TypeError, ValueError):
            pass

def generate_splits(num_workers, path, filenames, basename = 'train'):
    splits = []
    split_size = int(math.ceil(len(filenames) / num_workers))
    for i in range(num_workers):
            output_path = "{}-part-{:05d}-of-{:05d}.csv".format(basename, i, num_workers)
            splits.append((path, filenames[i * split_size: (i + 1) * split_size], output_path))
    return splits

def do_work(path):
    '''Don't call this. Under the covers TLS creates processes and that doesn't work'''
    filenames = os.listdir(path)
    splits = generate_splits(8, path, filenames, basename = 'period_info')
    num_workers = 8
    workers = multiprocessing.Pool(processes=num_workers)
    async_results = [workers.apply_async(part_worker, s) for s in splits]
    workers.close()
    for async_result in async_results:
            async_result.get()


if __name__ == '__main__':
    #for path in ['sector23', 'sector24']:
    #    do_work(path)
    parser = argparse.ArgumentParser(description="Script for computing period info")
    parser.add_argument("--input", type=str, required=True, help="Input folder where the .fits file are")
    parser.add_argument("--output", type=str, required=True, help="Output TIC Catalog (such as, period_info-sec25.csv)")
    args = parser.parse_args()
    _BASE_PATH = args.input
    load_claret_tess_info()
    filenames = os.listdir(args.input)
    part_worker(_BASE_PATH, filenames, args.output)

