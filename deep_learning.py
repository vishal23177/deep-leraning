import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, SimpleRNN  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_squared_error

# ANN Model for Data Preprocessing (Reduced Layers)
def create_ann_model(input_shape):
    model = Sequential()
    model.add(Dense(32, input_shape=input_shape, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Bidirectional RNN Model
def create_bidirectional_rnn_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(30, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(SimpleRNN(30)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Bidirectional LSTM Model
def create_bidirectional_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(30, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(30)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Bidirectional GRU Model
def create_bidirectional_gru_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(GRU(30, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(GRU(30)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fetch data from yfinance
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Close']]
    return df

# Preprocess data and apply ANN
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Train a simpler ANN model on the scaled data for preprocessing
    ann_model = create_ann_model((scaled_data.shape[1],))
    ann_model.fit(scaled_data, scaled_data, epochs=5, batch_size=64, verbose=0)
    
    # Transform data using the trained ANN model
    preprocessed_data = ann_model.predict(scaled_data)
    
    return preprocessed_data, scaler

# Prepare training and testing sets
def prepare_data(data, lookback=60):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Train and Predict
def train_and_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    predictions = model.predict(X_test)
    return predictions

# Make future predictions
def predict_future(model, recent_data, scaler, days):
    future_predictions = []
    input_data = recent_data[-1]
    for _ in range(days):
        pred = model.predict(input_data.reshape(1, input_data.shape[0], input_data.shape[1]))
        future_predictions.append(pred[0][0])
        input_data = np.append(input_data[1:], pred, axis=0)
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()

# Streamlit App
st.title('Stock Price Prediction with Bidirectional RNN, LSTM, and GRU')
st.write("Compare the models' performance for stock price prediction:")

# User Inputs
ticker = st.text_input("Enter Stock Ticker", 'AAPL')
start_date = st.date_input("Enter Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("Enter End Date", pd.to_datetime("2023-01-01"))
days_to_predict = st.number_input("Enter number of days to predict into the future", min_value=1, max_value=365, value=7, step=1)

# Load and Preprocess Data
data = load_data(ticker, start_date, end_date)
preprocessed_data, scaler = preprocess_data(data.values)

# Prepare Training and Testing Data
lookback = 60
X, y = prepare_data(preprocessed_data, lookback)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train and Predict with all models
if st.button("Predict"):
    # RNN Model
    rnn_model = create_bidirectional_rnn_model((X_train.shape[1], X_train.shape[2]))
    rnn_predictions = train_and_predict(rnn_model, X_train, y_train, X_test)
    rnn_predictions = scaler.inverse_transform(rnn_predictions)
    
    # LSTM Model
    lstm_model = create_bidirectional_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_predictions = train_and_predict(lstm_model, X_train, y_train, X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    
    # GRU Model
    gru_model = create_bidirectional_gru_model((X_train.shape[1], X_train.shape[2]))
    gru_predictions = train_and_predict(gru_model, X_train, y_train, X_test)
    gru_predictions = scaler.inverse_transform(gru_predictions)

    # Inverse transform actual values
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Plot Historical Results Comparison
    st.subheader("Model Comparison (Historical Data)")
    result_df = pd.DataFrame({
        'Actual': actual.flatten(),
        'Bidirectional RNN': rnn_predictions.flatten(),
        'Bidirectional LSTM': lstm_predictions.flatten(),
        'Bidirectional GRU': gru_predictions.flatten()
    })
    st.line_chart(result_df)

    # Future Predictions for Each Model
    rnn_future = predict_future(rnn_model, X_test, scaler, days_to_predict)
    lstm_future = predict_future(lstm_model, X_test, scaler, days_to_predict)
    gru_future = predict_future(gru_model, X_test, scaler, days_to_predict)

    # Display future predictions
    st.subheader(f"Future Predictions for the Next {days_to_predict} Days")
    future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Bidirectional RNN': rnn_future,
        'Bidirectional LSTM': lstm_future,
        'Bidirectional GRU': gru_future
    }).set_index('Date')
    st.write(future_df)
    st.line_chart(future_df)

    # Calculate accuracy for historical data predictions
    def calculate_mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    rnn_mape = calculate_mape(actual, rnn_predictions)
    lstm_mape = calculate_mape(actual, lstm_predictions)
    gru_mape = calculate_mape(actual, gru_predictions)

    def calculate_rmse(actual, predicted):
        return np.sqrt(mean_squared_error(actual, predicted))

    rnn_rmse = calculate_rmse(actual, rnn_predictions)
    lstm_rmse = calculate_rmse(actual, lstm_predictions)
    gru_rmse = calculate_rmse(actual, gru_predictions)

    # Display Accuracy
    st.subheader("Model Accuracy (MAPE - Historical Data)")
    st.write(f"Bidirectional RNN Accuracy: {100 - rnn_mape:.2f}%")
    st.write(f"Bidirectional LSTM Accuracy: {100 - lstm_mape:.2f}%")
    st.write(f"Bidirectional GRU Accuracy: {100 - gru_mape:.2f}%")

    # Display RMSE
    st.subheader("Model RMSE (Historical Data)")
    st.write(f"Bidirectional RNN RMSE: {rnn_rmse:.2f}")
    st.write(f"Bidirectional LSTM RMSE: {lstm_rmse:.2f}")
    st.write(f"Bidirectional GRU RMSE: {gru_rmse:.2f}")

    # Plot Accuracy
    st.subheader("Model Accuracy Graph (MAPE - Historical Data)")
    accuracy_df = pd.DataFrame({
        'Model': ['Bidirectional RNN', 'Bidirectional LSTM', 'Bidirectional GRU'],
        'Accuracy': [100 - rnn_mape, 100 - lstm_mape, 100 - gru_mape]
    })

    fig, ax = plt.subplots()
    ax.bar(accuracy_df['Model'], accuracy_df['Accuracy'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    st.pyplot(fig)

    # Plot RMSE
    st.subheader("Model RMSE Graph (Historical Data)")
    rmse_df = pd.DataFrame({
        'Model': ['Bidirectional RNN', 'Bidirectional LSTM', 'Bidirectional GRU'],
                'RMSE': [rnn_rmse, lstm_rmse, gru_rmse]
    })

    fig, ax = plt.subplots()
    ax.bar(rmse_df['Model'], rmse_df['RMSE'], color=['blue', 'green', 'orange'])
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE')
    ax.set_title('Model RMSE Comparison')
    st.pyplot(fig)

