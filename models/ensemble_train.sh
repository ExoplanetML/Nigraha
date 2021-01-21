#!bash
for classifier in binary 
do
  for views in global_nodropout
  do
     for i in {1..10}
     do
       echo training for $classifier/$views/models_$i.hdf5 with config ${views}_$classifier.json
       mkdir -p weights/$views/$classifier
       python train.py --ckpt weights/$views/$classifier/models_$i.hdf5 \
                       --train ../data/TFRecords/Large/train \
                       --validate ../data/TFRecords/Large/validate \
                       --config ../config/${views}_$classifier.json
     done
  done
done
