#!bash
for i in 14
do
   mkdir -p TFRecords/predict/sec$i
   python process_new_data.py --catalog ../catalog/period_info-sec$i.csv \
        --tic2fileDB ../catalog/tic2file-sec$i.csv \
        --input ../lcs/sector$i/ --output TFRecords/predict/sec$i/
done
