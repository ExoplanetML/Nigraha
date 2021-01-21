
for i in 14; 
do 
    python build_lcdb.py ../lcs/sector$i --output ../catalog/tic2file-sec$i.csv

    mkdir -p ../catalog/temp
    echo "Computing priors..."
    python period_find.py --input ../lcs/sector$i/ --output ../catalog/temp/prior-sec$i.csv

    echo "Detrending and computing period for TCEs..."

    python period_find_detrending.py --input ../lcs/sector$i --tic2fileDB ../catalog/tic2file-sec$i.csv \
                        --prior ../catalog/temp/prior-sec$i.csv --output ../catalog/temp/detrend-sec$i.csv

    echo "Filling in Stellar params..."

    python stellar_params.py ../catalog/temp/detrend-sec$i.csv ../catalog/temp/stellar-sec$i.csv

    echo "Adding in params from FITS header..."

    python fits_params.py --input ../lcs/sector$i/ --tic2fileDB ../catalog/tic2file-sec$i.csv \
                        --catalog ../catalog/temp/stellar-sec$i.csv --output ../catalog/period_info-sec$i.csv

    rm -rf ../catalog/temp
done
