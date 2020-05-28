# struc-mon
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

