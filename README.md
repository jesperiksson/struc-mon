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
  Tensorflow 2.3 - For machine learning \ 
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

# Editing models, chosing data, etc.
The code modules containing the neural nets are located in the folder `presets`. In order to set hyperparameters: pick a template file, make a copy of it and set the wanted hyperparameters and settings. In order to chose the wanted: either set the `preset` option to the name of the file or run the code with the appropriate flags (see next step)

# Initializing, training and evaluating

Run the main file in a terminal: `python3 main.py`. There are a number of additional arguments that can be set. Run the program with the help flag in order to view these: `python3 main.py -h`. Among others, the preset and weather to train a new net or load an existing one can be set. 

# Saving and loading

Currently, the program saves every trained neural net. They are saved in the folder `saved` where each neural net that has been trained creates a folder. The folder is named by the combination of preset, model name and sensors it uses. If the program is ran twice without changing any of these three settings the program will face an already exixting folder with the same name as the one it is currently trying to create. In that case the program adds `_change_name_` to the name in order to not over write anything. 

It is only the neural net that is saved i.e. its weights and states. In order to properly load the weights and states and use it again the net must have the exact same architecture as when said weights were created. 





