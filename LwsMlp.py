### Robust training of a multilayer perceptron (MLP) with a fixed architecture.
### The added value here is the formulation of the robust loss function,
### called at the appropriate place within the code, exploiting
### numerous available Keras libraries. Except for the robust loss function,
### standard tools of training machine learning are used.

### Importing libraries, preparing the prerequisities
import numpy as np
import math
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.models import load_model

### Robust MLP (with a highly robust loss function)
def LWS_MLP(input, output, weights = '', batch_size = 100, epochs = 500, model = None, tens = False, save = True):
	if len(output)% batch_size == 0:
		M = batch_size
	else:
		M = len(output)
	
	w = np.arange(M, dtype='float32')
	z = int(M*0.8)
	wLWS = 1-(w-1)/z ### the user may of course define an alternative weight function
	wLWS[z:M] = 0
	wLWS = np.flip(wLWS, 0)
	if type(weights) != str and isinstance(weights, list):
		if len(weights) == M:
			wLWS = weights
		else:
			print("Wrong parameter weights, should be size of " + str(M))
	elif weights == "expo":
		wLWS = tf.keras.backend.reverse(1/(1 + K.exp(10*((w-1)/M-1/2))), 0)
	
	def LWSLoss(y_pred, y_true): ### definition of the robust loss function
		err = (y_pred - y_true) ** 2
		rez = tf.nn.top_k(tf.transpose(err), k=M, sorted=True).values
		return K.mean(tf.multiply(rez,wLWS))

	if model == None:
		model = Sequential()
		model.add(Dense(20, input_dim=input.shape[1]))  ### Description of the (fixed) architecture
		model.add(LeakyReLU(alpha=.001)) 
		model.add(Dense(40))
		model.add(LeakyReLU(alpha=.001))
		model.add(Dense(40))
		model.add(LeakyReLU(alpha=.001))
		model.add(Dense(40))
		model.add(LeakyReLU(alpha=.001))
		model.add(Dense(1))
	
	model.compile(loss=LWSLoss, optimizer='adam', metrics=['accuracy']) 
        ### this is the main point, call the robust MLP training with the (above-defined) non-standard loss function right here
	
	if tens:
		NAME = "lwsNet"
		tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME))
		model.fit(input, output, batch_size=M, epochs=epochs, verbose=0, shuffle=True, callbacks=[tensorboard])
	else:
		model.fit(input, output, batch_size=M, epochs=epochs, verbose=0, shuffle=True)
	if save:
		model.save('LWSmodel.h5')
	return model

