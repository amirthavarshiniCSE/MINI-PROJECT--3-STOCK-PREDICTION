import pandas as pd
import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
stock = yf.download("AAPL","2018-01-01", "2024-11-17")
stock
prices = stock['Close'].values.reshape(-1,1)
prices.shape
prices
scaler = MinMaxScaler()
scaled_pries = scaler.fit_transform(prices)
scaled_pries.min()
X = []
y = []

for i in range(len(scaled_pries) - 60):
    X.append(scaled_pries[i:i+60])
    y.append(scaled_pries[i+60])
X, y = np.array(X), np.array(y)
len(X)*0.8
X_train , X_test = X[:1336], X[1336:]
y_train , y_test = y[:1336], y[1336:]

len(y_test)
X_train = X_train.reshape(1336,60)
#Sequential- one by one
#Parallel- everything run same time
#Dense- A simple layer (simple neural network). Types of different layer.
model = Sequential([
    Dense(64, activation="relu",input_shape=(60,) ),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])
model.compile(optimizer = "adam", loss  = "mean_squared_error", metrics = ["accuracy"])
model.fit(X_train, y_train, epochs=10)
y_pred = model.predict(X_test)
y_pred
y_test
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
last_60_days =scaled_pries[-60:]
len(last_60_days)
X_future = []
X_future.append(last_60_days)
X_future = np.array(X_future)
X_future.shape
nov15 = model.predict(X_future)
scaler.inverse_transform(nov15)
