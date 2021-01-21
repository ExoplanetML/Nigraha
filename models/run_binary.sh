#!bash

for classifier in binary
do
  for views in global_nodropout
  do
    for sec in 30
    do
      mkdir -p ../output/testing/scores/$classifier/
      ( python ensemble_predict.py --ckpt weights/$views/$classifier/ \
        --config ../config/${views}_$classifier.json \
        --data ../data/TFRecords/predict/sec$sec/ --threshold 0.4 \
        --output ../output/scores/$classifier/sec$sec.csv ) &
    done
  done
done

wait
