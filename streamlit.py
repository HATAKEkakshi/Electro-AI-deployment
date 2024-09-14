import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from keras.layers import Dense,Dropout,SimpleRNN,LSTM
from keras.models import Sequential

import warnings
warnings.filterwarnings("ignore")




st.title("Electro-AI")
st.text("Welcome to Electro you can Experience Two model ")
choicedataset=['PJME_hourly','DOM_hourly']
choose=st.selectbox("Choose from following Dataset:",choicedataset)
if choose=='PJME_hourly':
    df = pd.read_csv("/Users/hemantkumar/Developer/hackathon/sih/model/dataset/PJME_hourly.csv")
    # convert the Datetime column to Datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # indexing the Datetime column after the transformation
    df.set_index('Datetime', inplace=True)
    # Let see at the years in the data set
    years = df.index.year.unique()
    # Let's see the average energy consumed per year
    df_yearly_avg = df['PJME_MW'].resample('Y').mean()
    st.write(df_yearly_avg.to_frame())
    st.title('PJME_MW - Megawatt Energy Consumption')

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 5))
    df.plot(ax=ax, legend=True)

    # Highlight region with axhspan
    ax.axhspan(0, 1, facecolor='gray', alpha=0.3)

    # Set plot title
    ax.set_title('PJME Power (PJME) - Megawatt Energy Consumption')

    # Display the plot in Streamlit
    st.pyplot(fig)
    def normalize_data(df):
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df['PJME_MW'].values.reshape(-1,1))
        df['PJME_MW'] = normalized_data
        return df, scaler

    df_norm, scaler = normalize_data(df)
    df_norm.shape
    # 2017-02-13 after this date we will choose the test set
    split_date = '2017-02-13'
    st.title("Training and Test Dataset")
    DOM_train = df_norm.loc[df_norm.index <= split_date].copy()
    DOM_test = df_norm.loc[df_norm.index > split_date].copy()
    fig, ax = plt.subplots(figsize=(15, 5))
    DOM_train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
    DOM_test.plot(ax=ax, label='Test Set')
    ax.axvline('2017-02-13', color='black', ls='--')
    ax.legend(['Training Set', 'Test Set'])
    plt.axhspan(0, 1, facecolor='gray', alpha=0.3)
    st.pyplot(fig)
    def load_data(data, seq_len):
        X_train = []
        y_train = []

        for i in range(seq_len, len(data)):
            X_train.append(data.iloc[i-seq_len : i, 0])
            y_train.append(data.iloc[i, 0])

        # last 6189 days are going to be used in test
        X_test = X_train[110000:]
        y_test = y_train[110000:]

        # first 110000 days are going to be used in training
        X_train = X_train[:110000]
        y_train = y_train[:110000]

        # convert to numpy array
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # reshape data to input into RNN&LSTM models
        X_train = np.reshape(X_train, (110000, seq_len, 1))

        X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))

        return [X_train, y_train, X_test, y_test]
    seq_len = 20

    # Let's create train, test data
    X_train, y_train, X_test, y_test = load_data(df, seq_len)

    st.write('X_train.shape = ',X_train.shape)
    st.write('y_train.shape = ', y_train.shape)
    st.write('X_test.shape = ', X_test.shape)
    st.write('y_test.shape = ',y_test.shape)
    choice=["LSTM","RNN"]
    choice1=st.selectbox("Choose from following Dataset:",choice)
    if choice1=="RNN":
        #accesing model
        with open('/Users/hemantkumar/Developer/hackathon/sih/model/model(pickle)/rnnmodel2.pkl','rb') as f :
            rnn1=pickle.load(f)
        rnn_predictions=rnn1.predict(X_test)
        rnn_score = r2_score(y_test,rnn_predictions)
        st.title("Model Accuracy")
        accuracyscore=rnn_score*100
        st.write("Accuracy RNN model = ",accuracyscore)
        # Plotting
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(y_test, color='blue', label='Actual power consumption data')
        ax.plot(rnn_predictions, alpha=0.7, color='yellow', label='Predicted power consumption data')

        # Highlight region with axhspan
        ax.axhspan(0, 1, facecolor='gray', alpha=0.3)

        # Set plot title and labels
        ax.set_title("Predictions made by LSTM model")
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized power consumption scale')

        # Add legend
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)
    else:
        ##Buliding Model
        with open('/Users/hemantkumar/Developer/hackathon/sih/model/model(pickle)/lstmmodel2.pkl','rb') as d:
            lstm=pickle.load(d)
        lstm_predicts=lstm.predict(X_test)
        lstm_score = r2_score(y_test, lstm_predicts)
        st.title("Model Accuracy")
        accuracyscore_lstm=lstm_score*100
        st.write("R^2 Score of LSTM model = ",accuracyscore_lstm)

        st.title("LSTM Model Power Consumption Predictions")

        # Plotting
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(y_test, color='blue', label='Actual power consumption data')
        ax.plot(lstm_predicts, alpha=0.7, color='yellow', label='Predicted power consumption data')

        # Highlight region with axhspan
        ax.axhspan(0, 1, facecolor='gray', alpha=0.3)

        # Set plot title and labels
        ax.set_title("Predictions made by LSTM model")
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized power consumption scale')

        # Add legend
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)

else:
    df = pd.read_csv("/Users/hemantkumar/Developer/hackathon/sih/model/dataset/DOM_hourly.csv")
    # convert the Datetime column to Datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # indexing the Datetime column after the transformation
    df.set_index('Datetime', inplace=True)
    # Let see at the years in the data set
    years = df.index.year.unique()
    # Let's see the average energy consumed per year
    df_yearly_avg = df['DOM_MW'].resample('Y').mean()
    st.write(df_yearly_avg.to_frame())
    st.title('Delhi Power (DOM) - Megawatt Energy Consumption')

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 5))
    df.plot(ax=ax, legend=True)

    # Highlight region with axhspan
    ax.axhspan(0, 1, facecolor='gray', alpha=0.3)

    # Set plot title
    ax.set_title('DOM Power (DOM) - Megawatt Energy Consumption')

    # Display the plot in Streamlit
    st.pyplot(fig)
    def normalize_data(df):
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df['DOM_MW'].values.reshape(-1,1))
        df['DOM_MW'] = normalized_data
        return df, scaler

    df_norm, scaler = normalize_data(df)
    df_norm.shape
    # 2017-02-13 after this date we will choose the test set
    split_date = '2017-02-13'

    DOM_train = df_norm.loc[df_norm.index <= split_date].copy()
    DOM_test = df_norm.loc[df_norm.index > split_date].copy()
    fig, ax = plt.subplots(figsize=(15, 5))
    DOM_train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
    DOM_test.plot(ax=ax, label='Test Set')
    ax.axvline('2017-02-13', color='black', ls='--')
    ax.legend(['Training Set', 'Test Set'])
    plt.axhspan(0, 1, facecolor='gray', alpha=0.3)
    st.pyplot(fig)
    def load_data(data, seq_len):
        X_train = []
        y_train = []

        for i in range(seq_len, len(data)):
            X_train.append(data.iloc[i-seq_len : i, 0])
            y_train.append(data.iloc[i, 0])

        # last 6189 days are going to be used in test
        X_test = X_train[110000:]
        y_test = y_train[110000:]

        # first 110000 days are going to be used in training
        X_train = X_train[:110000]
        y_train = y_train[:110000]

        #  convert to numpy array
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # reshape data to input into RNN&LSTM models
        X_train = np.reshape(X_train, (110000, seq_len, 1))

        X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))

        return [X_train, y_train, X_test, y_test]
    seq_len = 20

    # Let's create train, test data
    X_train, y_train, X_test, y_test = load_data(df, seq_len)

    st.write('X_train.shape = ',X_train.shape)
    st.write('y_train.shape = ', y_train.shape)
    st.write('X_test.shape = ', X_test.shape)
    st.write('y_test.shape = ',y_test.shape)
    choice=["LSTM","RNN"]
    choice1=st.selectbox("Choose from following Dataset:",choice)
    if choice1=="RNN":
        #accesing model
        with open('/Users/hemantkumar/Developer/hackathon/sih/model/model(pickle)/rnnmodel.pkl','rb') as g :
            rnn=pickle.load(g)
        rnn_predictions=rnn.predict(X_test)
        rnn_score = r2_score(y_test,rnn_predictions)
        st.title("Model Accuracy")
        accuracyscore=rnn_score*100
        st.write("Accuracy RNN model = ",accuracyscore)
        # Reverse transform scaler to convert to real values
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
        rnn_predictions_inverse = scaler.inverse_transform(rnn_predictions)

        # Get values after inverse transformation
        y_test_inverse = y_test_inverse.flatten()
        rnn_predictions_inverse = rnn_predictions_inverse.flatten()
        last_6169_index_dates = df.index[-6169:]

        # Now let's see our actual y and predicted y values as dataframes
        results_RNN = pd.DataFrame({"Date":last_6169_index_dates, 'Actual': y_test_inverse, 'Predicted': rnn_predictions_inverse})
        result_rnnpredict=pd.DataFrame({"Date":last_6169_index_dates,"Predicted":rnn_predictions_inverse})
        st.write(result_rnnpredict)
        # Plotting
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(y_test, color='blue', label='Actual power consumption data')
        ax.plot(rnn_predictions, alpha=0.7, color='yellow', label='Predicted power consumption data')

        # Highlight region with axhspan
        ax.axhspan(0, 1, facecolor='gray', alpha=0.3)

        # Set plot title and labels
        ax.set_title("Predictions made by LSTM model")
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized power consumption scale')

        # Add legend
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)
    else:
        ##Buliding Model
        with open('/Users/hemantkumar/Developer/hackathon/sih/model/model(pickle)/lstmmodel.pkl','rb') as e:
            lstm1=pickle.load(e)
        lstm_predicts=lstm1.predict(X_test)
        lstm_score = r2_score(y_test, lstm_predicts)
        st.title("Model Accuracy")
        accuracyscore_lstm=lstm_score*100
        st.write("R^2 Score of LSTM model = ",accuracyscore_lstm)
        # Reverse transform scaler to convert to real values
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
        lstm_predictions_inverse = scaler.inverse_transform(lstm_predicts)

        # Get values after inverse transformation
        y_test_inverse = y_test_inverse.flatten()
        lstm_predictions_inverse = lstm_predictions_inverse.flatten()
        # Now let's see our actual y and predicted y values as dataframes
        last_6169_index_dates = df.index[-6169:]
        results_lstmpredict = pd.DataFrame({"Date":last_6169_index_dates, 'Predicted': lstm_predictions_inverse})
        st.write(results_lstmpredict)
        st.title("LSTM Model Power Consumption Predictions")

        # Plotting
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(y_test, color='blue', label='Actual power consumption data')
        ax.plot(lstm_predicts, alpha=0.7, color='yellow', label='Predicted power consumption data')

        # Highlight region with axhspan
        ax.axhspan(0, 1, facecolor='gray', alpha=0.3)

        # Set plot title and labels
        ax.set_title("Predictions made by LSTM model")
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized power consumption scale')

        # Add legend
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)
