# This is to generate balanced datasets...we split positive and negative
python preprocess.py --catalog ../catalog/period_info-toi-exofop.csv --tic2fileDB ../catalog/tic2file-toi.csv --input ~/Personal/Astro/data/TESS/all-sector-toi/ --output TFRecords/Large/ --basename toi
python preprocess.py --catalog ../catalog/period_info-tces-dl3.csv --tic2fileDB ../catalog/tic2file-tces-dl3.csv --input ~/Personal/Astro/data/TESS/tces-dl3/ --output TFRecords/Large/ --basename dl3 --exclude ../catalog/tic2file-toi.csv 
mv TFRecords/Large/train/*positive*.tfRecords TFRecords/Large/train/positive
mv TFRecords/Large/train/*negative*.tfRecords TFRecords/Large/train/negative
