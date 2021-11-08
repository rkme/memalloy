import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # Need this to scale inputs 


def load_file(csv_path, start_row=2):
    # We want this file
    print(f'Reading file at path {csv_path} ...')
    # starts at start_row, 2 is default due to our data generation
    all_data = pd.read_csv(csv_path, delimiter = ',', skiprows=start_row)
    print('Read complete!')
    return all_data


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def frame_dataset(train_path, mode=1):
  # loading dataset
  data = load_file(train_path, 2)
  # So there's a bunch of extra cols, here's where we drop everything that's not
  # what we want, I have labeled the cols dropped to train the four models below

  #Select which input mode, depending on model mode
  #modes is a dict that maps Test #->dropped columns to build training dataset
  modes = {1: [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20], #UniPWM \
         2: [0,1,2,3,5,6,7,8,9,11,12,13,14,15,17,18,19,20], #UniPWM+T \
         3: [0,1,2,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20], #BiPWM \
         4: [0,1,2,5,6,7,8,9,12,13,14,15,17,18,19,20]} #BiPWM+T
  irrelevant_cols = modes[mode]

  # drop it like it's (a) hot (thermocouple measurement)
  data.drop(data.columns[irrelevant_cols], axis = 1, inplace= True)

  # ensure all data is float
  values = data.values
  values = values.astype('float32')

  # normalize features
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled = scaler.fit_transform(values)

  # frame as supervised learning
  window = 1
  future = 1
  reframed = series_to_supervised(scaled, window, future)
  print(reframed.head())
  return reframed


def gen_train_test(_values, ratio = 0.67):
  n = int(len(_values)*ratio) # split point based on ratio
  train = _values[:n, :] # first 67% of data
  test = _values[n:, :] # last 33% of data
  # split into input and outputs
  train_X, train_y = train[:, :-1], train[:, -1] # all but last col, last col
  test_X, test_y = test[:, :-1], test[:, -1]
  # reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  return [train_X, train_y, test_X, test_y]
