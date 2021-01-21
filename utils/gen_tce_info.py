
import pandas as pd
import numpy as np
import argparse

'''Script to merge multiple catalogs to generate one for the final candidates'''
def do_work(args):
        inputDf = pd.read_csv(args.input, index_col='tic_id')
        if 'rowid' in inputDf.columns:
                inputDf = inputDf.drop('rowid', axis=1)
        catalogDf = pd.read_csv(args.catalog, index_col='TIC_ID')
        if 'rowid' in catalogDf.columns:
                catalogDf = catalogDf.drop('rowid', axis=1)
        toiDf = pd.read_csv(args.toi, index_col='TIC_ID')
        toiDf = toiDf[~toiDf.index.duplicated(keep='first')]
        outputDf = inputDf.merge(catalogDf, left_index=True, right_index=True)
        crossRefDf = pd.DataFrame(columns=['TIC_ID', 'Disposition', 'Comments'])
        for tic_id in outputDf.index:
                if tic_id in toiDf.index:
                        disposition = toiDf['TFOPWG Disposition'][tic_id]
                        try:
                                if np.isnan(toiDf['TFOPWG Disposition'][tic_id]):
                                        disposition = toiDf['Group Disposition'][tic_id]
                        except:
                                 pass
                        values = {'TIC_ID' : tic_id, 
                                'Disposition' : disposition,
                                'Comments' : toiDf['Public Comment'][tic_id]}
                else:
                        values = {'TIC_ID' : tic_id, 'Disposition' : '', 'Comments' : ''}
                crossRefDf = crossRefDf.append(values, ignore_index=True)
        crossRefDf = crossRefDf.set_index('TIC_ID')
        # print(crossRefDf)
        outputDf = outputDf.merge(crossRefDf, left_index=True, right_index=True)
        outputDf = outputDf.reset_index()
        outputDf = outputDf.rename(columns={'index' : 'TIC_ID'})
        outputDf.to_csv(args.output, index_label='rowid')

                
if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Generate TCE catalog")
        parser.add_argument("--input", type=str, required=True, help="csv file with TIC ID, probabilities" )
        parser.add_argument("--catalog", type=str, required=True, help="TIC Catalog (such as, period_info-sec24.csv)")
        parser.add_argument("--toi", type=str, required=True, help="TOI catalog file (such as, toi-catalog.csv)")
        parser.add_argument("--output", type=str, required=True, help="Output .csv with everything merged (e.g., candidates/sec24.csv) ")
        args = parser.parse_args()
        do_work(args)
