for i in 14
do 
   echo $i; 
   python gen_tce_info.py --input ../output/scores/sec$i.csv \
        --catalog ../catalog/period_info-sec$i.csv --toi ../catalog/exofop_toi_101220.csv \
        --output ../output/candidates/sec$i.csv
done
