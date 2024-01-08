# To run this file and streamlit web app simply
# First Step - (activate virtual environment/optional)
# Second Step -  pip install streamlit (optional)
# Third Step - streamlit run "c:\Users\Aashi Sharma\OneDrive\Desktop\Streamlit_project_stock_trend_prediction\App.py" (to run streamlit app)
# Streamlit Web App code


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime
import yfinance as yf

# ...



start_date = '2015-01-01'
end_date = '2023-12-31'
start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
end = datetime.datetime.strptime(end_date, '%Y-%m-%d')

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start_date, end=end_date)

# df = data.DataReader(user_input,'yahoo',start,end)
st.subheader('Data from 2015-2023')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


# Moving Average
st.subheader('Closing Price Vs Time Chart With 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart With 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, "r")
plt.plot(ma200, "g")
plt.plot(df.Close, "b")
st.pyplot(fig)


# Splitting dataset into 70% and 30% for training and testing respectively

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


# Load model

model = load_model('keras_model.h5')

# Testing part


past_100_days = data_training.tail(100)
data_testing_df = pd.DataFrame(data_testing, columns=['Close'])  # Convert data_testing to DataFrame
final_data = pd.concat([past_100_days, data_testing_df], ignore_index=True)
input_data = scaler.fit_transform(final_data)



x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

# Value of Scale Factor
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph

st.subheader('Predicted Vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
