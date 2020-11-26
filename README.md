# struc-mon

## How to run the code (Linux)
1. Clone this repository into the desired folder: `git clone https://github.com/jesperiksson/struc-mon` .
2. Make sure you have at least Python 3.7 installed: ´python --version´. Othervise update: https://docs.python-guide.org/starting/install3/linux/
3. Make sure you either have Anaconda installed: https://docs.anaconda.com/anaconda/install/ or have the following packages installed:
  Matplotlib.pyplot - For plots
  Numpy - For maths
  Pandas - For data frames
  SciPy - For signal analysis
  sklearn - For normalization
  Tensorflow 2.3 - For machine learning 
  Otherwise they can be installed using pip:
  `pip install pip`
  Matplotlib: https://pypi.org/project/matplotlib/
  Numpy: https://numpy.org/install/
  Pandas: https://pypi.org/project/pandas/ 
  SciPy: https://www.scipy.org/install.html
  sklearn: https://scikit-learn.org/stable/install.html
  Tensorflow: https://www.tensorflow.org/install/pip
4. Open the file in the repo named `config.py` and provide the path to the folder where the measurement files are stored in the variable `file_path`,
   e.g. `file_path = 'home/user/measurements'` 
5. Open the file `Settings.py` and set the ´model´ attribute to the name of the model file you want to run (e.g. `test_NN.py`). Then set ´name´ attribute to the name of the model (these could be the same).
6. 

Toolbox for analyzing bridges using ANN and other methods. 
This project was initially developed for a Master's thesis on the topic of Structural Health Monitoring.

The program is written with the intent of being highly modular in order to make it easy to implement new methods.

Required Libraries:
Keras
Matplotlib
NumPy
SciPy
sklearn
TensorFlow 2.0

The program is executed from the run.py file. 

Data is imported and made into instances of the Databatch class. The Databatch class has several sub-classes,
where each subclass manipulates the data in a certain way. The Peaks subclass chooses the peak signals and 
their indices for instance. 

Models (neural nets) are objects belonging to the NeuralNet class which has sublclasses for each type of neural net.
Different models within each type of neural net are defined by their respective function within said NeuralNet 
sub-class. The Subclasses has their own methods for training, evaluation, prediction and forecasting. 

