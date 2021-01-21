import pandas as pd
import numpy as np

orig = pd.read_csv('../catalog/period_info-toi-exofop.csv', index_col='TIC_ID')
#detrend = pd.read_csv('../catalog/detrend-v2-toi-exofop.csv', index_col='TIC_ID')
#detrend = pd.read_csv('../catalog/detrend-v3-toi-exofop.csv', index_col='TIC_ID')
detrend = pd.read_csv('../catalog/period_info-detrended-toi-exofop.csv', index_col='TIC_ID')

count = 0
for tic in detrend.index:
        if (not tic in orig.index) or np.isnan(detrend['NumTransits'][tic]) or int(detrend['NumTransits'][tic]) == int(orig['NumTransits'][tic]):
                continue
        #if orig["Disposition"][tic] != 'KP':
        #        continue
        if int(detrend['NumTransits'][tic]) < 2 or detrend['snr'][tic] < 7.1 or detrend['sde'][tic] < 9.0:
                continue
        count += 1
        print(f'{count}: {tic}  Without detrend {orig["Period"][tic]:.3f}, With detrend: {detrend["Period"][tic]:.3f}',
                f' {orig["Group Comment"][tic]}')

