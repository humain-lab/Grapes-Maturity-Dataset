# use mlp for prediction on multi-output regression
from numpy import asarray
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy
import time
import pandas as pd
import os
from statistics import mean
from joblib import Parallel, delayed
import matplotlib.pyplot as plt



# get the dataset
def get_dataset(index):
	file1 = 'dataFiles/ins_per_day.xlsx'

	#FOR RAW DATA PROCESSING:
	# file1 = 'dataFiles/raw-histogram-data.xlsx'

	xl = pd.ExcelFile(file1, engine='openpyxl')
	s = xl.sheet_names  # see all sheet names
	allins = []

	file2 = 'dataFiles/chemicals.xlsx'
	xl_chem = pd.ExcelFile(file2, engine='openpyxl')
	s_chem = xl_chem.sheet_names  # see all sheet names
	allchems = []

	for sheet in s:
		df = pd.read_excel(file1, engine='openpyxl', sheet_name=sheet)
		l = df.columns
		for i in l:
			mylist = df[i].tolist()
			allins.append(mylist)

	for sheet in s_chem:
		df = pd.read_excel(file2, engine='openpyxl', sheet_name=sheet)
		l = df.columns
		for i in l:
			mylist = df[i].tolist()
			allchems.append(mylist)

	# # RANDOMIZE
	# list_index = randrange(len(allins))
	list_index = index

	temp = allins[list_index]
	allins.pop(list_index)
	allins.append(temp)

	temp = allchems[list_index]
	allchems.pop(list_index)
	allchems.append(temp)

	# ########### HERE GOES FILTERING #############
	# allins = list(map(filter, allins))
	# allchems = list(map(filter, allchems))

	allins = numpy.array(allins)
	allchems = numpy.array(allchems)


	return allins[:-1],allchems[:-1], allins[-1], allchems[-1]


def filter(freq):
	resArray = []
	for i in range(len(freq)):
		if i > len(freq) / 3 and freq[i] < max(freq) * (0.5 / 100):
			return resArray
		else:
			resArray.append(freq[i])
	return resArray

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(10, input_dim=n_inputs, kernel_initializer='he_uniform', activation='10'))
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	model.compile(loss='mae', optimizer='adam')
	return model

def classify_acidity(value):
	if value < 4:
		#immature
		return 0
	# elif value > 7:
	# 	#rotten
	# 	return 0
	else:
		#mature
		return 1

def classify_ph(value):
	if value < 3.2:
		# immature
		return 0
	# elif value > 3.5:
	# 	# rotten
	# 	return 0
	else:
		# mature
		return 1

def classify_baume(value):
	if value < 11.1:
		# immature
		return 0
	# elif value > 12.8:
	# 	# rotten
	# 	return 0
	else:
		# mature
		return 1

def create_diagonal_plot(real,prediction,folder,filename,plot_title):
	plt.scatter(real, prediction, c=numpy.random.rand(len(prediction)))
	# plt.scatter(y, c='r')
	# plt.plot([min(real + prediction), max(real + prediction)], [min(real + prediction), max(real + prediction)])
	plt.plot([min(real), max(real)], [min(prediction), max(prediction)])
	# plt.xlim([min(real)-1, max(real)+1])
	plt.xlabel('Real Values')
	plt.ylabel('Predicted Values')
	plt.title(plot_title)
	plt.savefig(os.path.join(folder, (str(filename) + '.png')))
	print([min(real + prediction), max(real + prediction)],plot_title)
	plt.clf()
	plt.cla()
	# plt.show()

def main():
	startTime = time.time()
	results_i = list(range(1, 101))
	DataItems = 100

	results_real, results_pred, results_mae, results_mse, results_mape, results_msle,\
		error_acidity, error_ph, error_baume,\
		class_acidity, class_ph, class_baume,\
		class_acidity_pred, class_ph_pred, class_baume_pred = ([] for i in range(15))

	for i in range(DataItems):
		# load dataset

		X, y, X_pred, y_pred = get_dataset(i)

		n_inputs, n_outputs = X.shape[1], y.shape[1]

		# n_inputs, n_outputs = X, y
		# get model
		model = get_model(n_inputs, n_outputs)
		# fit the model on all data
		model.fit(X, y, verbose=0, epochs=1000)
		# make a prediction for new data
		row = X_pred
		# row = numpy.array(X[-1])

		newX = asarray([row])
		yhat = model.predict(newX)
		# print(row)
		# print(y[-1])

		mse = tf.keras.losses.MeanSquaredError()
		mse_result = mse(y_pred, yhat[0]).numpy()

		mae = tf.keras.losses.MeanAbsoluteError()
		mae_result = mae(y_pred, yhat[0]).numpy()

		mape = tf.keras.losses.MeanAbsolutePercentageError()
		mape_result = mape(y_pred, yhat[0]).numpy()

		msle = tf.keras.losses.MeanSquaredLogarithmicError()
		msle_result = msle(y_pred, yhat[0]).numpy()


		# f.write('Item: %s of 100\n' % str(i+1))
		# f.write('REAL: %s\n' % y_pred)
		# f.write('Predicted: %s\n' % yhat[0])
		# f.write('Loss (mse): %s\n\n' % mse_result)


		print('Iteration: ', i)
		print('REAL: ',y_pred)
		print('Predicted: %s' % yhat[0])
		print('Loss (mse): %s' % mse_result)
		if i > 0:
			print('Avg mae: %s' % mean(results_mae))

		results_real.append(y_pred)
		results_pred.append(yhat[0])
		results_mse.append(mse_result)
		results_mae.append(mae_result)
		results_mape.append(mape_result)
		results_msle.append(msle_result)
		error_acidity.append(abs(y_pred[0]-yhat[0][0]))
		error_ph.append(abs(y_pred[1]-yhat[0][1]))
		error_baume.append(abs(y_pred[2] - yhat[0][2]))


		# print(model.evaluate())

		# Evaluate the model on the test data using `evaluate`
		# print("Evaluate on test data")
		# results = model.evaluate(X, y)
		# print("test loss, test acc:", results)
		#
		# print("Generate predictions for 3 samples")
		# predictions = model.predict(X[:3])
		# print("predictions shape:", predictions.shape)
		# print(predictions)
		print('=========================')


		# model.save('SavedModel')

	# f.close()

	class_acidity = [classify_acidity(i) for i in numpy.array(results_real)[:,0]]
	class_ph = [classify_ph(i) for i in numpy.array(results_real)[:, 1]]
	class_baume = [classify_baume(i) for i in numpy.array(results_real)[:, 2]]

	class_acidity_pred = [classify_acidity(i) for i in numpy.array(results_pred)[:, 0]]
	class_ph_pred = [classify_ph(i) for i in numpy.array(results_pred)[:, 1]]
	class_baume_pred = [classify_baume(i) for i in numpy.array(results_pred)[:, 2]]

	mae_acidity = sum(error_acidity) * (1/DataItems)
	mae_ph = sum(error_ph) * (1/DataItems)
	mae_baume = sum(error_baume) * (1/DataItems)

	matches_acitidy = 0
	matches_ph = 0
	matches_baume = 0
	for match_item in range(DataItems):
		if class_acidity[match_item] == class_acidity_pred[match_item]:
			matches_acitidy += 1
		if class_ph[match_item] == class_ph_pred[match_item]:
			matches_ph += 1
		if class_baume[match_item] == class_baume_pred[match_item]:
			matches_baume += 1

	maeErrors = [mae_acidity,mae_ph,mae_baume,matches_acitidy,matches_ph,matches_baume]
	maeErrors.extend([0] * (DataItems-6)) ## fill with zeros to avoid pandas error

	folderPath = os.path.join("results",
							  str(300 - (matches_acitidy + matches_ph + matches_baume) + ((sum(maeErrors[:3])) / 10)))
	os.mkdir(folderPath)

	fileName = os.path.join(folderPath,
							str(300 - (matches_acitidy + matches_ph + matches_baume) + ((sum(maeErrors[:3])) / 10)) + '.xlsx')

	create_diagonal_plot(numpy.array(results_real)[:, 0], numpy.array(results_pred)[:, 0], folderPath, 'Acidity',
						 'Acidity')
	create_diagonal_plot(numpy.array(results_real)[:, 1], numpy.array(results_pred)[:, 1], folderPath, 'PH',
						 'PH')
	create_diagonal_plot(numpy.array(results_real)[:, 2], numpy.array(results_pred)[:, 2], folderPath, 'Baume',
						 'Baume')

	writer = pd.ExcelWriter(fileName, engine='openpyxl')
	wb = writer.book
	df = pd.DataFrame({'Item#': results_i,
					   'Real(Acidity)': numpy.array(results_real)[:,0],
					   'Class(Acidity)': class_acidity,
					   'Real(PH)': numpy.array(results_real)[:, 1],
					   'Class(PH)': class_ph,
					   'Real(Baume)': numpy.array(results_real)[:, 2],
					   'Class(Baume)': class_baume,
					   'Prediction(Acidity)': numpy.array(results_pred)[:, 0],
					   'Class_Pred(Acidity)': class_acidity_pred,
					   'Prediction(PH)': numpy.array(results_pred)[:, 1],
					   'Class_Pred(PH)': class_ph_pred,
					   'Prediction(Baume)': numpy.array(results_pred)[:, 2],
					   'Class_Pred(Baume)': class_baume_pred,
					   'MeanSquaredError': results_mse,
					   'MeanAbsoluteError': results_mae,
					   'MeanAbsolutePercentageError ': results_mape,
					   'MeanSquaredLogarithmicError ': results_msle,
					   'AcidityError': error_acidity,
					   'PhError': error_ph,
					   'BaumeError': error_baume,
					   'MaeErrors And Matches': maeErrors})

	df.to_excel(writer, index=False)
	wb.save(fileName)

	endTime = time.time()
	print("TIME: ",round(endTime - startTime, 2))

if __name__ == "__main__":
	Parallel(n_jobs=12)(delayed(main)() for k in range(12))
	# main()
