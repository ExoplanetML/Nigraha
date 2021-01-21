#!bash
#for i in {24..26}
#for i in {18..19}
for i in 30
do
   python process_new_data.py --catalog ../catalog/period_info-sec$i.csv \
        --tic2fileDB ../catalog/tic2file-sec$i.csv \
        --input ../lcs/sector$i/ --output TFRecords/predict/sec$i/
done
