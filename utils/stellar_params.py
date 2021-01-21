
from __future__ import division, print_function

import numpy as np
import pandas as pd
import pprint
from tqdm import tqdm
from astroquery.mast import Catalogs

from transitleastsquares import (
    transitleastsquares,
    cleaned_array,
    catalog_info,
    transit_mask
    )

if __name__ == '__main__':
    df = pd.read_csv('../catalog/period_info-sec30.csv')
    #df = pd.read_csv('../catalog/s19-pinfo.csv')
    mass = [np.nan] * len(df)
    #a = [np.nan] * len(df)
    #b = [np.nan] * len(df)
    distance = [np.nan] * len(df)
    logg = [np.nan] * len(df)
    lum = [np.nan] * len(df)
    rho = [np.nan] * len(df)
    count = 0
    for i in tqdm(range(len(df))):
        tic_id = int(df['TIC_ID'][i])
        try:
              #ab, m, _, _, r, _, _ = catalog_info(TIC_ID=tic_id)
              result = Catalogs.query_criteria(catalog='TIC', ID=tic_id)
        except:
              #ab,  m, r = (np.nan, np.nan), np.nan, np.nan
              continue
        #a[i] = ab[0]
        #b[i] = ab[1]
        mass[i] = float(result['mass'])
        distance[i] = float(result['d'])
        logg[i] = float(result['logg'])
        lum[i] = float(result['lum'])
        rho[i] = float(result['rho'])

    #df['logg'] = logg
    df['Mass'] = mass
    df['distance'] = distance
    df['lum'] = lum
    df['rho'] = rho
    df.to_csv('../catalog/period_info-sec30-new.csv')
