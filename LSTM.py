from util import *

class LongShortTermMemoryMachine():

    def __init__(self, trainset, n_samples)

        """
        Args:
            time_window: Duration of sample
            n_acceleroms: Number of accelerometers
            pred_accelerom: the accelerometer whose result is to be predicted
        """
        self.n_samples = n_samples
        self.n_accels = trainset.shape[1]
        self.n_node = trainset.shape[0]/self.n_samples

        self.activation='relu'
        self.loss='mse'

    def trainLSTM(self, trainset, lbl, n_samples)
        
        """
        Args: 
            trainset: Training dataset, containing data on this form:
            samples : 1:k = n_samples
            nodes : 1:m = n_nodes
            accelerations : 1:n = n_accels
            [[[a111, ..., a11n], [a121, ..., a12n], ..., [a1m1, ..., a1mn]], ...,
             [[ak11, ..., ak1n], [ak21, ..., ak2n], ..., [akm1, ..., akmn]]]
            lbl: the sample that is to be predicted (and therefore excluded from training)
            n_samples = number of samples contained in the trainset
        """

        X = np.zeros(np.shape(trainset)[0]-n_samples,n_accels) # Creating an empty (k*(m-1),n) array 
        for i in range(n_node-1)
            start = i*n_node
            end = (i+1)*n_node      
            try : 
                X[start:end-1] = np.concatenate((trainset[:lbl-1],trainset[lbl:]), axis=0)
            except ValueError : 
                if lbl == 1 # To handle the case where node 1 is to be predicted
                    X[start:end-1] = trainset[1:]
                else # To handle the case where last node is to be predicted
                    X[start:end-1] = trainset[:-1]   
            y[i] = trainset[lbl*(i+1)]
        
  
        # define model
        model = Sequential()
        model.add(LSTM(50, activation=self.activation, input_shape=(n_steps, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss=self.loss)
        # fit model
        model.fit(X, y, epochs=200, verbose=0)

        return

'''     Example of split secuence function
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
'''

