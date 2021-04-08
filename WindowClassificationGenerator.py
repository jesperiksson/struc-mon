import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config
class WindowClassificationGenerator():
    def __init__(
            self, 
            input_width, 
            shift,
            train_df, 
            val_df, 
            test_df,
            train_batch_size = 32,
            eval_batch_size = 32,
            test_batch_size = 32):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df   
        
        # Work out the window parameters.
        self.input_width = input_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        #Set the batch_sizes
        self.train_batch_size = train_batch_size 
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size   
        
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        print(inputs)     
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])

        return inputs
        
    def make_dataset(self, data, bs):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=bs,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df, self.train_batch_size)

    @property
    def val(self):
        return self.make_dataset(self.val_df, self.eval_batch_size)

    @property
    def test(self):
        return self.make_dataset(self.test_df, self.test_batch_size)
