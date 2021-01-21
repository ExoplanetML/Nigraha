
import tensorflow as tf
import numpy as np

#
# Helper functions
# 

def create_balanced_sample_dataset(pos_ds, neg_ds, batchsize=64, epochs=1):
        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        # resampled_ds = resampled_ds.repeat()
        resampled_ds = resampled_ds.batch(batchsize)
        resampled_ds = resampled_ds.shuffle(20000, reshuffle_each_iteration=True)
        #if epochs > 1:
        #        resampled_ds = resampled_ds.repeat()

        return resampled_ds

def create_dataset(file_list, extra_fields, batchsize=1, epochs=1, shuffle_records=False):
        dataset = tf.data.TFRecordDataset(file_list)
        # Create a string dataset of filenames, and possibly shuffle.
        print(f'File_list: {file_list}, extra_fields = {extra_fields}, epochs = {epochs}')
        #filename_dataset = tf.data.Dataset.from_tensor_slices(file_list)
        
        # Read serialized Example protos.
        #dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)

        #print(f'Setting batch_size = {batchsize} and epochs = {epochs}')

        def _parse_function(proto):
                # define the tfrecord again. Remember that you saved your image as a string.
                keys_to_features = {
                        'TIC_ID': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        'global view': tf.io.VarLenFeature(tf.float32),
                        'local view': tf.io.VarLenFeature(tf.float32),
                        'shifted global view': tf.io.VarLenFeature(tf.float32),
                        'odd_even view': tf.io.VarLenFeature(tf.float32),
                        'Disposition': tf.io.FixedLenFeature([], tf.string, default_value='')
                }
                if extra_fields is not None:
                        extra_features = {k: tf.io.FixedLenFeature([1], tf.float32) for k in extra_fields}
                        keys_to_features.update(extra_features)
                # Load one example
                parsed_example = tf.io.parse_single_example(proto, keys_to_features)
                return parsed_example
        #
        # Maps the parser on every filepath in the array.
        dataset = dataset.map(_parse_function)
        if epochs > 1:
                print('Setting repeat')
                dataset = dataset.repeat()
        # Set the batchsize
        if batchsize > 0:
                dataset = dataset.batch(batchsize)
        if shuffle_records:
                print('Shuffling records')
                dataset = dataset.shuffle(20000, reshuffle_each_iteration=True)

        #iterator = iter(dataset)

        #return iterator
        return dataset


if __name__ == '__main__':
        dataset = create_dataset_from_dl_paper(['/Users/srirao/Personal/OSS/Astronet-Triage/astronet/tfrecords/test-00000-of-00001'])
        iter = iter(dataset)
        item = next(iter)
        print(item)

