# struc-mon

## Linux
# Downloads and setup
1. Clone this repository into the desired folder: `git clone https://github.com/jesperiksson/struc-mon` .
2. Make sure you have at least Python 3.7 installed: ´python --version´. Othervise update: https://docs.python-guide.org/starting/install3/linux/
3. Make sure you either have Anaconda installed: https://docs.anaconda.com/anaconda/install/ or have the following packages installed:
  Matplotlib.pyplot - For plots \
  Numpy - For maths \
  Pandas - For data frames \
  Rainflow - For creating rainflow analysis of stress \
  SciPy - For signal analysis \
  sklearn - For normalization \
  Tensorflow 2.x - For machine learning \ 
  Otherwise they can be installed using pip: \
  `pip install pip`\
  Matplotlib: https://pypi.org/project/matplotlib/ \
  Numpy: https://numpy.org/install/ \
  Pandas: https://pypi.org/project/pandas/ \
  Rainflow: https://pypi.org/project/rainflow/ \
  SciPy: https://www.scipy.org/install.html \
  sklearn: https://scikit-learn.org/stable/install.html \
  Tensorflow: https://www.tensorflow.org/install/pip \
4. Open the file in the repo named `config.py` and provide the path to the folder where the measurement files are stored in the variable `file_path`, e.g. `file_path = 'home/user/measurements'` 

# Data.py 
Superclass for objects containing data
# AggregatedData.py
Subclass containing methods for reading and storing data using OGC Standard: http://docs.opengeospatial.org/is/15-078r6/15-078r6.html
# PostgresData.py 
Subclass containing methods for reading and storing data from a PostgreSQL server. Queries are generated with a QueryGenerator object. Also contain a further subclass for anomalies. 

# Anomaly.py
Anomaly objects

# AnomalySettings.py 
Settings for anomaly objects

# Model.py 
Superclass for model objects. Also contains subclasses TimeSeriesPredictionNeuralNet and TimeSeriesClassificationNeuralNet

# PCAAnomalies 
Class for PCA objects operating on Anomaly objects





