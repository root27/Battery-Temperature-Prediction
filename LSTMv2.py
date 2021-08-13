from keras.engine.input_layer import Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, RepeatVector
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('LSTM_Battery_temp.csv')

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Cycle', 'Battery-M'])
for i in range(0,len(data)):
    new_data['Cycle'][i] = data['Cycle'][i]
    new_data['Battery-M'][i] = data['Battery-M'][i]

#setting index
new_data.drop('Cycle', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values
""""
#split data
split_data=int(len(data)*0.8)  
train = dataset[0:split_data,:]
valid = dataset[split_data:,:]
"""
#converting dataset into x_train and y_train
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled_data = scaler.fit_transform(dataset)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(dataset, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100, verbose=2)

test = pd.read_csv('test.csv')
new_data = pd.DataFrame(index=range(0,len(test)),columns=['Temp'])
x_test = new_data.values

X_test, y_test = split_sequence(x_test, 3, 2)
# demonstrate prediction
#x_input = scaler.transform(np.array(x_test[1:4]))
print(X_test.shape)

x_input = X_test[0].reshape((1, X_test[0].shape[0], X_test[0].shape[1]))
yhat = model.predict(x_input, verbose=2)
print(yhat.shape)

#yhat = scaler.inverse_transform(yhat)
print(yhat, y_test[0])
