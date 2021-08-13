from keras.engine.input_layer import Input
from keras.engine.saving import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, RepeatVector
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.preprocessing import MinMaxScaler

# read data
df = pd.read_csv('LSTM_Battery_temp.csv')

#path_save0='results.csv'
#path_save1='valid.csv'
#path_save2='...'

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Cycle', 'Battery-M'])
for i in range(0,len(data)):
    new_data['Cycle'][i] = data['Cycle'][i]
    new_data['Battery-M'][i] = data['Battery-M'][i]

#setting index
new_data.drop('Cycle', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values


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
#split data


#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

n_features = 1

X_train, y_train = split_sequence(scaled_data, 3, 2)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], n_features))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], n_features))
# create and fit the LSTM network
print(X_train.shape, y_train.shape)
model = Sequential()
model.add(LSTM(units=100, input_shape=(3,1)))
model.add(RepeatVector(2))
model.add(LSTM(units=100, return_sequences=True))
model.add(TimeDistributed(Dense(1)))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_squared_error"])
model.summary()
model.fit(X_train, y_train, epochs=100, verbose=2)
model.save("battery_temp_predict_v2_1(2samples)_100epochs")



# prediction process
test = pd.read_csv('test.csv')
new_data = pd.DataFrame(index=range(0,len(test)),columns=['Temp'])
x_test = new_data.values

scaled_test = scaler.transform(x_test)
X_test, y_test = split_sequence(scaled_test, 3, 2)


model = load_model("battery_temp_predict_v2_1(2samples)_100epochs")
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
results = model.predict(X_test)

print(results, y_test)
results = results.reshape(-1,1)
y_test = y_test.reshape(-1,1)
print(X_test.shape,results.shape)
print(y_test.shape)

results = scaler.inverse_transform(results)
y_test = scaler.inverse_transform(y_test)
rms = np.sqrt(np.mean(np.power((y_test-results),2)))
print (rms)

""""
#for plotting
train = new_data[:split_data]
valid = new_data[split_data:]

train = new_data[:split_data]



predict = pd.DataFrame(index=range(0,len(results)),columns=['Predictions'])
predict["Predictions"] = results
plt.plot(train['Battery-M'],lw=3)#time,t
plt.plot(valid['Battery-M'],lw=3)
plt.plot(predict['Predictions'],lw=3)


result['Predictions'].to_csv(path_save0,index=False,sep=';')
valid['Battery-M'].to_csv(path_save1,index=False,sep=';')
#valid['Predictions'].to_csv(path_save2,index=False,sep=';')

plt.show()
"""