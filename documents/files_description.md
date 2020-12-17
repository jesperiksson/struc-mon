# Files 
`config.py` - change file path, set the keywords for reading data (provided in the measurement files), data format, figure size, etc.

`Data.py` - Contains the code classes for data series `DataBatch` and series stack `Series_Stack`, as well as subclasses to series stack for the three different kinds of data available at Västerbron: `Acc_Series_Stack`, `Incl_Series_Stack`, `Strain_Series_Stack`. In order to modify the code to accept data from other sources in other formats a new subclass will have to be added which transforms raw data into the format requested by `DataBatch`. The `DataBatch` objects are initialized by the `populate_stack()` method from the parent class `Series_Stack`.

`Filter_Settings.py` - When applying a signal filter to the raw data the parameters are defined here. 

`inspect_data.py` Run this file to plot a random input file. This file has an argument parser, type `inspect_data.py -h` to list options.

`main.py` The main file, run this to run the program. This file has an argument parser, type `inspect_data.py -h` to list options.

`Model.py` - Contains the `Model` class, its subclass `NeuralNet` as well as its subclass `TimeSeriesNeuralNet`. The reason they are arranged in classes is to facilitate a future implementation of NN:s other than time series nets and ML-models other than NN:s. These will then share the save/ load methods, for example. 

`Settings.py` - Contains the `Settings` class. When the main file is ran without any arguments the code will use the settings defined in here. The `name` attribute is used when saving a trained model as well as its plots. The `preset` attribute defines which model-module to load and use (i.e. which neural net) 

The various files with the prefix `test_` contains Unittest-suites used during the development of the code. 

`WindowGenerator` - The same code as used here: https://www.tensorflow.org/tutorials/structured_data/time_series , apart from a couple of minor changes. Does all the wokr with windowing the data and turning pandas DataFrames into tf.Data.Dataset objects. 
