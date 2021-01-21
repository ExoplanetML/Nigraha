#!bash
model='G-201-L-81'

#for classifier in binary multi
for classifier in binary
do
     #for views in global_nodropout shifted_global_nodropout
     for views in global_nodropout 
     do
        echo Running: python ensemble_evaluate.py --ckpt ckpts/paper/exofop-labels/$classifier/$views/ensemble_$model/ --test ../data/TFRecords/Large/test/ --config ../config/${views}_$classifier.json --output phase_two/paper/exofop-labels/$classifier/$views/test-output.csv
        ( python ensemble_evaluate.py --ckpt weights/detrend/lk/$classifier/ \
            --config ../config/${views}_$classifier.json \
            --test ../data/TFRecords/Large/test/ \
            --output ../output/detrend/unconstrained/predictions/binary/test-output-full.csv) &
     done
done

wait
#            --test ../data/TFRecords/G-201-L-81/exofop-labels/test/ \
#            --output phase_two/paper/exofop-labels/$classifier/$views/test-output.csv) &
            # --test ../data/TFRecords/G-201-L-81/tces/test/ \
            #--output phase_two/paper/exofop-labels/$classifier/$views/tces-test-output.csv) &
