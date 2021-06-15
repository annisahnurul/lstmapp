#!/usr/bin/env python
# coding: utf-8

# In[38]:


#Inisialisasi
import streamlit as st
import time

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy as np
import pandas as pd
from pandas import read_csv
from math import sqrt
from matplotlib import pyplot
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import keras


# In[39]:
st.title("Prediksi Harga Beras")

# load dataset
file = st.file_uploader("Pilih file")
if not file:
  st.stop()

series = read_csv(file, usecols=[1], engine='python')

# In[40]:


train = int(len(series)*0.9)
test = int(len(series) - train)
print("\nPanjang train: ", train)
print("Panjang test : ", test)


# In[41]:


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# In[42]:


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# In[43]:


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# In[44]:


# scale train and test data to [0, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# In[45]:


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# In[46]:


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=50, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    return model


# In[47]:


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


# In[48]:


#Convert list to one dimension array
def toOneDimension(value):
    return np.asarray(value)


# In[49]:


#Convert to multi dimension array
def convertDimension(value):
    return (np.reshape(lastPredict, (lastPredict.shape[0], 1, lastPredict.shape[0])))


# In[50]:


# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values,1)


# In[51]:


# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values


# In[52]:


# split data into train and test-sets
train, test = supervised_values[0:-10], supervised_values[-10:]


# In[53]:


# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)


# In[54]:


# fit the model
#lstm_model = fit_lstm(train_scaled, 1, 50, 10)
#lstm_model.save('model000.h5')


# In[55]:


lstm_model = keras.models.load_model('model02.h5')


# In[56]:


# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
hasiltraining=lstm_model.predict(train_reshaped, batch_size=1)



# In[78]:


# walk-forward validation on the test data
predictions = list()
tmpPredictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # get tmp prediction data
    tmpPredictions.append(yhat)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    #print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    st.write('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# In[80]:


# line plot of observed vs predicte
pyplot.plot(raw_values[-10:])

pyplot.plot(predictions)
#pyplot.show()

yhat = forecast_lstm(lstm_model, 1, X)
yhat = y



# In[81]:


# Prepare data (get last predict)
lastPredict = tmpPredictions[-1:]
lastPredict = toOneDimension(lastPredict)
lastPredict = convertDimension(lastPredict)


# In[82]:


# Predict for n month (future)
futureMonth = 6 #predict for n month


# In[83]:


# Make backup model
model_predict = lstm_model;


# In[84]:


# do Predict
futureArray = []
for i in range(futureMonth):
    lastPredict = model_predict.predict(lastPredict)
    futureArray.append(lastPredict)
    lastPredict = convertDimension(lastPredict)


# In[85]:


# Before denormalize
newFutureData = np.reshape(futureArray,(-1,1))
pd.DataFrame(newFutureData, columns=['Future Prediction Result (Before Invert Scaling)'])


# In[86]:


# Change dimension
newFuture = np.reshape(newFutureData, (-1, 1))


dataHasilPrediksi = [];
for i in range(len(newFutureData)):
    tmpResult = invert_scale(scaler, X, newFutureData[i])
    tmpResult = inverse_difference(raw_values, tmpResult, len(newFutureData)+1-i)
    dataHasilPrediksi.append(tmpResult)
    st.write("Month",i+1,":",tmpResult)
    #print("Month",i+1,":",tmpResult)

dataHasilPrediksi

pyplot.plot(dataHasilPrediksi)
#pyplot.show()



# In[87]:


# Generate continous graphic
newFutureLine = [];
for i in range(len(raw_values)):
    newFutureLine.append(None) #Append blank value
for i in range(len(dataHasilPrediksi)):
    newFutureLine.append(dataHasilPrediksi[i])

supervised_values[-10:]


# In[88]:


# Generate testing line
newTrainingLine = [];
for i in range(len(hasiltraining)):
    tmpResult = invert_scale(scaler, X, hasiltraining[i])
    tmpResult = inverse_difference(raw_values, tmpResult, len(hasiltraining)+1-i)
    newTrainingLine.append(tmpResult)


# In[89]:


# Generate continous testing graphic
newTestingLine = [];
for i in range(len(supervised_values[0:-10])):
    newTestingLine.append(None) #Append blank value
for i in range(len(predictions[-10:])):
    newTestingLine.append(predictions[i])


# In[75]:


plt.figure(figsize = (10, 7))
plt.plot(raw_values, label = "Actual Data") # real data line
plt.plot(newTestingLine, label = "Testing line") # testing line
plt.plot(newFutureLine, label = "Future prediction line") # future prediction line
plt.xlabel("Date")
plt.ylabel("Harga Beras")
plt.title("real vs past predict and future predict")
plt.legend()
#plt.show()
st.pyplot(plt)


# In[76]:


# report performance
rmse = sqrt(mean_squared_error(raw_values[-10:], predictions))
#print('Test RMSE: %.3f' % rmse)
st.write('Test RMSE: %.3f' % rmse)

# In[77]:


mae = mean_absolute_error(raw_values[-10:], predictions)
#print('Test MAE: %.3f' % mae)
st.write('Test MAE: %.3f' % mae)

# In[ ]:




