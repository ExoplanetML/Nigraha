
'''Script that builds a model with stellar params, evaluates it, and writes out a checkpoint.'''

from data import records_io
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os, sys
import pprint
import argparse, json
import csv
import matplotlib.pyplot as plt
from pathlib import Path

# We are mapping KP/PC to 0; so, there is one fewer class than # of keys.  Rather than taking
# the number of keys, explicitly set the # of values we expect to have.
disposition_to_index = {'KP' : 0, 'PC' : 0, 'EB' : 1, 'V' : 2, 'IS' : 3, 'J' : 4, 'Num Values' : 5}
index_to_disposition = {0 : 'KP_PC', 1 : 'EB', 2 : 'V', 3 : 'IS', 4 : 'J' }
all_disposition_to_index = {'KP' : 0, 'PC' : 0, 'EB' : 1, 'V' : 2, 'IS' : 3, 'J' : 4, 'PC_X' : 0, 'FP' : 4, 'UNK' : 4, 'O' : 4}

def get_one_hot(disposition):
        depth = disposition_to_index['Num Values']
        if not disposition in disposition_to_index.keys():
                disposition = 'J'
        one_hot = tf.one_hot(disposition_to_index[disposition], depth)
        return one_hot


def build_model(global_view_shape, local_view_shape, odd_even_view_shape,
                stellar_params,
                num_outputs = 1,
                output_bias = None, 
                use_dropout = False):
        def build_global_view_layers(g):
                x = g
                for filters in [16, 32, 64, 128, 256]: 
                      x = keras.layers.Conv1D(filters=filters, kernel_size=7, padding='same', activation='relu')(x)
                      x = keras.layers.Conv1D(filters=filters, kernel_size=7, padding='same', activation='relu')(x)
                      x = keras.layers.MaxPooling1D(pool_size=5, strides=2)(x)
                #global_path = keras.layers.GlobalAveragePooling1D()(x)
                global_path = keras.layers.Flatten()(x)
                return global_path
        
        def build_local_view_layers(l):
                x = l
                for filters in [16, 32]:
                        x = keras.layers.Conv1D(filters=filters, kernel_size=5, padding='same', activation='relu')(x)
                        x = keras.layers.Conv1D(filters=filters, kernel_size=5, padding='same', activation='relu')(x)
                        x = keras.layers.MaxPooling1D(pool_size=3, strides=2)(x)
                #local_path = keras.layers.GlobalAveragePooling1D()(x)
                local_path = keras.layers.Flatten()(x)
                return local_path

        pprint.pprint(global_view_shape)
        # Use Keras functional API to build a model.
        global_view_input = keras.Input(shape=global_view_shape, name="global_view")
        global_path = build_global_view_layers(global_view_input)

        local_view_input = keras.Input(shape=local_view_shape, name="local_view")
        local_path = build_local_view_layers(local_view_input)

        odd_even_view_input = keras.Input(shape=odd_even_view_shape, name="odd_even_view")
        odd_even_path = build_local_view_layers(odd_even_view_input)

        concat = keras.layers.Concatenate(axis=1)([global_path, local_path, odd_even_path])      

        if output_bias is not None:
                output_bias = tf.keras.initializers.Constant(output_bias)
        # Scalar params
        stellar_params_input = []
        for k in stellar_params:
                x = keras.Input(shape=((1,)), name=k)
                concat = keras.layers.Concatenate(axis=1)([concat, x])
                stellar_params_input.append(x)
        
        dense = keras.layers.Dense(512, activation='relu')(concat)
        dense = keras.layers.Dense(512, activation='relu')(dense)
        if use_dropout:
                dropout = keras.layers.Dropout(0.2)(dense)
        dense = keras.layers.Dense(512, activation='relu')(dense)
        if use_dropout:
                dropout = keras.layers.Dropout(0.2)(dense)        
        dense = keras.layers.Dense(512, activation='relu')(dense)
        if num_outputs > 1: 
                pred = keras.layers.Dense(num_outputs, name="prediction", activation='softmax')(dense)
        else:
                pred = keras.layers.Dense(1, name="prediction", activation='sigmoid', bias_initializer=output_bias)(dense)
        model = keras.Model(inputs=[global_view_input, local_view_input, odd_even_view_input,
                                stellar_params_input], 
                                outputs=[pred])

        keras_metrics = [
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'), 
                keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
        ]

        opt = keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-8)
        if num_outputs > 1:
                model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(),
                                metrics=[keras.metrics.CategoricalAccuracy()])
        else:
                # Since PC/KP are much smaller than FP's, we need precision-recall curve.
                model.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(),
                                metrics=keras_metrics)

        #model.summary()
        # keras.utils.plot_model(model, "phase_two_model.png", show_shapes=True, show_layer_names=True)
        return model

class DataGenerator(keras.utils.Sequence):
        def __init__(self, dataset, config, stellar_params, batchsize, num_outputs = 1,
                        should_reverse=True, num_lcs=986):
                #self.dataset = dataset
                self.dataset_iter = iter(dataset)
                self.config = config
                self.stellar_params_keys = stellar_params
                self.batchsize = batchsize
                self.should_reverse = False
                self.num_lcs = num_lcs
                self.batch_count = 0
                self.num_outputs = num_outputs


        def __len__(self):
                # 80% is trainset---7892; 1978 are PCs
                # 10% is devset---986; only PCs is 229
                # 10% is testset---987; 252 are PCs

                len = np.ceil(self.num_lcs / self.batchsize)
                return int(len)

        def __getitem__(self, index):
                #
                # Use dataset iterator to get a batch of items and then clean them.
                #
                items = next(self.dataset_iter)
                batch = self.clean_data(items)
                self.batch_count += 1
                d = {'global_view' : batch['global_view'], 'local_view' : batch['local_view'],
                        'odd_even_view' : batch['odd_even_view']}
                d.update({k : batch[k] for k in self.stellar_params_keys})
                # Keras expects input, output tuples in that order.
                return (d, batch['prediction'])

        def _replace_nans(self, t):
                count = len(tf.math.is_nan(t))
                if count != 0:
                        #print(f'Number of nans is {count}')
                        ret = tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)
                        #print(ret)
                        return ret
                return t

        def reverse_curve(self, m, curves):
                output_list = []
                for i in range(m):
                        rt = tf.reverse(curves[i], axis=[1])
                        output_list.append(rt)
                output_list = tf.stack(output_list)
                return output_list

        def clean_data(self, items):
                batch = {}
                num_examples = items['global view'].get_shape()
                m = np.minimum(num_examples[0], self.batchsize)

                if config['Use Shifted Global View']:
                        global_view = self._replace_nans(tf.reshape(tf.sparse.to_dense(items['shifted global view']), [m, 201, 1]))
                else:
                        global_view = self._replace_nans(tf.reshape(tf.sparse.to_dense(items['global view']), [m, 201, 1]))
                local_view = self._replace_nans(tf.reshape(tf.sparse.to_dense(items['local view']), [m, 81, 1]))
                odd_even_view = self._replace_nans(tf.reshape(tf.sparse.to_dense(items['odd_even view']), [m, 162, 1]))
                disposition = items['Disposition']

                if self.should_reverse:
                        reversed_view = self.reverse_curve(m, global_view)
                        global_view = tf.concat([global_view, reversed_view], axis=0)
                        reversed_view = self.reverse_curve(m, local_view)
                        local_view = tf.concat([local_view, reversed_view], axis=0)
                        reversed_view = self.reverse_curve(m, odd_view)
                        odd_view = tf.concat([odd_view, reversed_view], axis=0)
                        reversed_view = self.reverse_curve(m, even_view)
                        even_view = tf.concat([even_view, reversed_view], axis=0)
                        reversed_view = self.reverse_curve(m, mid_view)
                        mid_view = tf.concat([mid_view, reversed_view], axis=0)              
                        disposition = tf.concat([disposition, disposition], axis=0)
                        stellar_params = tf.concat([stellar_params, stellar_params], axis = 0)
             
                batch['global_view'] = global_view
                batch['local_view'] = local_view
                batch['odd_even_view'] = odd_even_view
                for k in self.stellar_params_keys:
                         batch[k] = self._replace_nans(items[k]).numpy()
                         #print(f'{k}: {batch[k].shape}')

                if self.num_outputs == 1:
                        pred = np.zeros((disposition.shape[0], 1))
                        count = 0
                        for i in range(disposition.shape[0]):
                                s = disposition[i].numpy()
                                v = s.decode('utf-8')
                                pred[i] = self.config['Disposition'][v]
                                count += pred[i]
                else:
                        # Generate one encoding
                        pred = []
                        entries = disposition.shape[0]
                        for i in range(disposition.shape[0]):
                                s = disposition[i].numpy()
                                v = s.decode('utf-8')
                                one_hot = get_one_hot(v)
                                pred.append(one_hot)
                                #print(f'{v} -> {one_hot}')
                        pred = tf.reshape(pred, [entries, self.num_outputs, 1])
                batch['prediction'] = pred
                #print(f'{batch["prediction"].shape}')
                #print(f'Number of PCs in batch: {count}')
                return batch

def get_absolute_path_listings(directory):
        filenames = [os.path.abspath(os.path.join(directory, p)) for p in os.listdir(directory)]
        return filenames

def plot_metrics(history):
        metrics =  ['loss', 'auc', 'precision', 'recall']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        #figure, axes = plt.subplots(2, 2)
        for n, metric in enumerate(metrics):
                name = metric.replace("_"," ").capitalize()
                plt.subplot(2,2,n+1)
                plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
                plt.plot(history.epoch, history.history['val_'+metric],
                        color=colors[0], linestyle="--", label='Val')
                plt.xlabel('Epoch')
                plt.ylabel(name)
                if metric == 'loss':
                        plt.ylim([0, plt.ylim()[1]])
                elif metric == 'auc':
                        plt.ylim([0.8,1])
                else:
                        plt.ylim([0,1])

        plt.legend()
        plt.show()

def train(args, config):
        # batchsize is really 64.  for each curve, we generate the reversed curve.
        # Since we aren't reversing any more, set the batchsize to 64.
        batchsize = 64  
        epochs = 60

        # Going back to the older model
        #batchsize = 32
        #epochs = 50

        #model = build_model(global_view.shape, local_view.shape, mp_dist.shape) 
        #model = build_model((201, 1), (61, 1), (186, 1), (8, 1))
        stellar_params = list(config['Stellar Features'])
        if config['Binary Classification']:
                num_outputs = 1
        else:
                num_outputs = disposition_to_index['Num Values']
        print(f'# of outputs: {num_outputs}')
        
        # For the DL3 data set, we have:
        # {PC: 398, EB: 1028, IS: 982, J: 5326, V:901, O:2}
        # Set the initial bias:
        # Pos: 398; neg: 1028 + 982 + 5326 + 901 + 2
        # For the EXOFOP-TOI, we have:
        # KP:142, PC: 353, CP: 30, FP: 143, FA: 10, O: 1
        initial_bias = np.log([(398+142+353+30)/(1028 + 982 + 5326 + 901 + 2+143+10+1)])
        print(f'Output bias: {initial_bias}')

        model = build_model((201, 1), (81, 1), (162, 1), 
                        stellar_params, 
                        num_outputs = num_outputs,
                        output_bias=initial_bias)
        
        filenames = get_absolute_path_listings(os.path.join(args.train, 'positive'))
        positive_ds = records_io.create_dataset(filenames, stellar_params, batchsize=-1, epochs=epochs, shuffle_records=True)
        filenames = get_absolute_path_listings(os.path.join(args.train, 'negative'))
        negative_ds = records_io.create_dataset(filenames, stellar_params, batchsize=-1, epochs=epochs, shuffle_records=True)
        dataset = records_io.create_balanced_sample_dataset(positive_ds, negative_ds, batchsize=batchsize, epochs=epochs)
        # We want to see the negatives twice per epoch
        # Magic #'s: EXOFOP-TOI:
        # train: 611, test: 68
        # Split size: 66, positives = 392, workers = 6
        # Split size: 54, negatives = 108, workers = 2
        # Number of tics to exclude: 679
        # TCES-DL3:
        # train: 7796, test: 867
        # Split size: 148, positives = 148, workers = 1
        # Split size: 871, negatives = 5223, workers = 6
        num_negative_examples = 5223 + 108
        resampled_steps_per_epoch = np.ceil(2.0 * num_negative_examples / batchsize)
        print(f'Resampled steps per epoch {resampled_steps_per_epoch}')
        if config["Use Shifted Global View"]:
                global_view_type = "shifted"
        else:
                global_view_type = "unshifted"
        print(f'Using {global_view_type} view')
        #
        # Ideally, we should able to pass the dataset into model.fit(). However, due to some
        # pre-processing that we need to do, as well as, https://github.com/tensorflow/tensorflow/issues/35925
        # dataset can't be handed into model.fit().  Hence, use a convenience wrapper.
        #
        generator = DataGenerator(dataset, config, stellar_params, batchsize, num_outputs = num_outputs,
                                should_reverse=False, num_lcs = 2 * (7796 + 611))

        filenames = get_absolute_path_listings(args.validate)
        validation_set = records_io.create_dataset(filenames, stellar_params, batchsize=batchsize, epochs=epochs)
        validation_generator = DataGenerator(validation_set, config, stellar_params, batchsize, 
                                        num_outputs = num_outputs, should_reverse=False)

        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit(generator, shuffle=True, epochs=epochs, steps_per_epoch=resampled_steps_per_epoch,
                                validation_data=validation_generator, callbacks=[callback])
        print(f'Loss history: {len(history.history["loss"])}')
        model.save(args.ckpt)
        # fig, ax = plt.subplots() 
        # ax.plot(history.history['loss'], label='train')
        # ax.plot(history.history['val_loss'], label='val')
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Loss')
        # ax.set_title('Training/Validation Loss Curves')
        # ax.legend()
        # fig.set_tight_layout(True)
        # fig.savefig(args.ckpt + '.png', bbox_inches='tight')
        # plt.close(fig)
        # plot_metrics(history)
        if args.test:
                filenames = get_absolute_path_listings(args.test)
                testset = records_io.create_dataset(filenames, stellar_params, batchsize=batchsize, epochs=epochs)
                generator = DataGenerator(testset, stellar_params, batchsize, num_outputs = num_outputs, should_reverse=False)
                history = model.evaluate(generator)
                print(f'history = {history}')

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Train DL model for detecting PC/EB or PC")
        parser.add_argument("--train", type=str, required=True, help="Path to files where .tfRecords for training are." )
        parser.add_argument("--validate", type=str, required=True, help="Path to files where .tfRecords for validation are." )
        parser.add_argument("--test", type=str, help="Path to files where .tfRecords for test are.")
        parser.add_argument("--ckpt", type=str, required=True, help="Filename where the model checkpoint should be saved")
        parser.add_argument("--config", type=str, required=True, help="Config file")
        args = parser.parse_args()

        p = Path(args.ckpt)
        if not p.parent.is_dir():
                print(f'Ckpt dir: {p.parent} does not exist!')
                sys.exit(-1)


        with open(args.config) as config_file:
                config = json.load(config_file)

        train(args, config)

        print(f'Model saved to {args.ckpt}')
        

