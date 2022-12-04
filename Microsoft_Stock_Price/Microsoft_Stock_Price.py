####################################################################################################################
## This is a dataset of Microsoft Stock price taken from Kaggle [1].                                              ##
##                                                                                                                ##
## "Stocks and financial instrument trading is a lucrative proposition. Stock markets across the world facilitate ##
## such trades and thus wealth exchanges hands. Stock prices move up and down all the time and having ability to  ##
## predict its movement has immense potential to make one rich.                                                   ##
## Stock price prediction has kept people interested from a long time. There are hypothesis like the Efficient    ## 
## Market Hypothesis, which says that it is almost impossible to beat the market consistently and there are       ##
## others which disagree with it." - Kaggle introduction of Yahoo Data Set [2]                                    ##
##                                                                                                                ##
## The Data set is coumposed of 7 columns :                                                                       ##
##       - Date -> Date of the information                                                                        ##
##       - High -> Highest Price of the stock for that particular date.                                           ##
##       - Low -> Lowest Price of the stock for that particular date.                                             ##
##       - Open -> Opening Price of the stock.                                                                    ##
##       - Close -> Closing Price of the stock.                                                                   ##
##       - Volume -> Total amount of Trading Activity.                                                            ##
##       - AdjClose -> Adjusted values factor in corporate actions such as dividends, stock splits, and new share ##
##         issuance.                                                                                              ##
##                                                                                                                ##
####################################################################################################################

#Import libraries of data manipulation & visualization
import os
import streamlit as st
import numpy as np 
import pandas as pd 
import math
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import plotly_express as px
from sklearn.preprocessing import MinMaxScaler

#import parallelization libraries
from multiprocessing.pool import ThreadPool

#import the test-libraries
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima.utils import ndiffs
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#Import the prediction libraries
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, LSTM

#Set the number of processor used for the parallelization on different cores :
pool = ThreadPool(processes=4)

#Import DataSet - os.abspath() does not function with my streamlit app :
path = "C:/Users/33646/Documents/GitHub/Portfolio/Microsoft_Stock_Price/Data/Microsoft_Stock.csv"

#load the data with parallelization on different processors:
@st.cache()
def load_df(file_name):
    df = pd.read_csv(file_name)
    return df
df = pool.apply_async(load_df, (path, )).get()

#Transform the Date object as DateTime in order to better handle it
df['Date'] = pd.to_datetime(df['Date'])

# graph function :
def graph_prediction(title
                     ,Train_Date
                     ,Train
                     ,Test_Date = None
                     ,Test = None
                     ,Prediction = None
                     ):
    fig, ax = plt.subplots(figsize=(16,8))
    plt.title(title)
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    plt.plot(Train_Date,Train, color='blue', label='Test')
    if Test is not None and Prediction is not None and Test_Date is not None:
        plt.plot(Test_Date, Test, color='green', label='Test')
        plt.plot(Test_Date, Prediction, color='red', label='Test')
        plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
    else :
        plt.legend("Stock price", loc='lower right')
    return st.pyplot(fig)




############################ Data Transformation ############################

diff = df.Close.diff().dropna()
result_adf = adfuller(df.Close)
result_adf1 = adfuller(diff)
d_optimum = ndiffs(df.Close.dropna(),test="adf")

n = int(len(df) * 0.8)
train_arima = df.Close[:n]
test_arima = df.Close[n:]
Date_train_arima = df.Date[:n]
Date_test_arima = df.Date[n:]


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df.Close.values.reshape(-1,1))
train_LSTM = scaled_data[:n]
test_LSTM = scaled_data[n - 60:,:]
x_train_LSTM = []
y_train_LSTM = []

for i in range(60, len(train_LSTM)):
    x_train_LSTM.append(train_LSTM[i-60:i, 0])
    y_train_LSTM.append(train_LSTM[i, 0])
        
x_test_LSTM = []
y_test_LSTM =  df.Close.values[n:]
Date_test_LSTM = df.Date[n:]
Date_train_LSTM = df.Date[:n]


for i in range(60,len(test_LSTM)):
    x_test_LSTM.append(test_LSTM[i-60:i, 0])
    
x_train_LSTM, y_train_LSTM = np.array(x_train_LSTM), np.array(y_train_LSTM)
x_test_LSTM = np.array(x_test_LSTM)

#Reshape the data
x_train_LSTM = np.reshape(x_train_LSTM, (x_train_LSTM.shape[0], x_train_LSTM.shape[1], 1))
x_test_LSTM = np.reshape(x_test_LSTM, (x_test_LSTM.shape[0], x_test_LSTM.shape[1], 1))

train = df.Close[:n]
valid = df.Close[n:]

# function to train the models :

## function to train the ARIMA

def train_ARIMA(p,q):
    arima_model = ARIMA(train_arima, order=(p,d_optimum,q))
    ARIMA_Model = arima_model.fit()
    Prediction_arima = ARIMA_Model.predict(len(train_arima), len(train_arima)+len(test_arima)-1, typ='levels')
    return ARIMA_Model,Prediction_arima

## function to train the LSTM Model
@st.cache()
def train_LSTM():
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_LSTM.shape[1], 1)))
    model_LSTM.add(LSTM(units=50, return_sequences=False))
    model_LSTM.add(Dense(units=25))
    model_LSTM.add(Dense(units=1))
    model_LSTM.compile(optimizer='adam', loss='mean_squared_error')
    model_LSTM.fit(x_train_LSTM, y_train_LSTM, batch_size=1, epochs=1)
    predictions_LSTM = model_LSTM.predict(x_test_LSTM)
    predictions_LSTM = scaler.inverse_transform(predictions_LSTM)
    #plot the data
    train = df.Close[:n]
    valid = df.Close[n:]
    valid = pd.DataFrame(valid)
    valid["Predictions"] = predictions_LSTM
    return valid

valid = train_LSTM()

# Function which retrieve the accuracy metrics

def forecast_accuracy(Model, forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return(pd.DataFrame.from_dict({'Model':[Model], 'mape': [mape], 'me':[me],
                                   'mae': [mae], 'mpe': [mpe], 'rmse':[rmse], 
                                   'corr':[corr], 'minmax':[minmax]}))
    


pages = st.sidebar.selectbox('Select the page', ['Introduction 🗺️', 'About the models 🧭', "Setting of the models ⚙️", 'Forecasting 📈'])

if pages == "Introduction 🗺️":
    
    st.title("Introduction to the project 🗺️")
    st.write("This Streamlit application will help you to predict the stock price of Microsoft.")
    st.write("To predict the price, we will use different model of machine learning and Deep Learning : ARIMA(ML) & LSTM(DL)\n")
    st.write("First, we will display the data and get some informations about the data.")
    graph_prediction("Visualization of the time serie", df['Date'],df["Close"])
    st.markdown(f"The dataframe begins on : {df.Date.dt.date.min()}, and finish at : {df.Date.dt.date.max()}.")
    st.markdown(f"The dataframe has {df.shape[0]} rows and {df.shape[1]} columns.")
    st.markdown(f"There are {df.isna().sum().sum()} missing values in the dataset.")
    
elif pages == "About the models 🧭":
    
    model = st.sidebar.selectbox('Select the model', ['ARIMA', 'LSTM'])
    st.title("About the models 🧭")
    
    if model == "ARIMA":
        
        factor = st.sidebar.selectbox('In which variable do you want information ?',['Choose a factor','P', 'D', 'Q'])
        
        if factor == "Choose a factor":
            
            st.markdown("Three factors define ARIMA model, it is defined as ARIMA(p,d,q) where p, d, and q denote\n - p is the order of the AR term or the number of lagged (or past) observations to consider for autoregression.\n - d is the number of differencing required to make the time series stationary \n - q is the order of the MA term")
        
        elif factor == "P":
            st.write(" P is the order of the Auto Regressive (AR) term. It refers to the number of lags to be used as predictors.")
            st.write("The below equation shows a typical autoregressive model. As the name suggests, the new values of this model depend purely on a weighted linear combination of its past values. Given that there are p past values, this is denoted as AR(p) or an autoregressive model of the order p. Epsilon ($\epsilon$) indicates the white noise.")
            st.latex(r'''Y_t = \alpha  + \beta_1 Y_{t-1} + \beta_2 Y_{t-2} + ... + \beta_p Y_{t-p} + \epsilon_1''')
            st.write("We can find out the required number of AR terms by inspecting the Partial Autocorrelation (PACF) plot. The partial autocorrelation represents the correlation between the series and its lags.")
            fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))
            ax1.plot(diff)
            ax1.set_title("Difference once")
            plot_pacf(diff, ax=ax2, title="Partial Autocorrelation difference once")
            st.pyplot(fig)
        
        elif factor == "D":
            
            st.write("Autoregressive models are conceptually similar to linear regression, the assumptions made by the latter hold here as well. Time series data must be made stationary to remove any obvious correlation and collinearity with the past data. In stationary time-series data, the properties or value of a sample observation does not depend on the timestamp at which it is observed.")
            st.latex(r'''y_t' = y_t - y_{t-1} = y_t - B y_t = (1-B)y_t''')
            st.write("where B denotes the backshift operator")
            st.write("We will use the augmented dickey fuller (ADF) test to check if the serie is stationnary.")
            st.write("The null hypothesis of the ADF test is that the time series is non-stationnary. So, if the p-value of the test is upper than the significance level then we can reject the null hypothesis and infer that the time serie is indeed stationnary.")
            st.markdown(f"ADF Statistics: {result_adf[0]}")
            st.markdown(f"p-value: {result_adf[1]}")
            st.write("The serie is not stationnary. We'll need to differentiate once to see.")
            st.markdown(f"ADF Statistics: {result_adf1[0]}")
            st.markdown(f"p-value: {result_adf1[1]}")
            st.write(f"The ARIMA model need {d_optimum} difference.")
            
        elif factor == "Q":
            
            st.write("q is the order of the moving average (MA) term. It refers to the number of lagged forecast errors that should go into the ARIMA model.")
            st.write("Here, the future value $Y_t$ is computed based on the errors $\epsilon_t$ made by the previous model. So, each successive term looks one step further into the past to incorporate the mistakes made by that model in the current computation. Based on the window we are willing to look past, the value of q is set. Thus, the below model can be independently denoted as a moving average order q or simply MA(q).")
            st.latex(r'''y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}''')
            st.write("We can look at the ACF plot for the number of MA terms. ")
            fig, (ax1,ax2) = plt.subplots(2,1, figsize=(10,10))
            ax1.plot(diff)
            ax1.set_title("Difference once")
            plot_acf(diff, ax=ax2, title="Autocorrelation difference once")
            st.pyplot(fig)
            
        else :
            
            st.write("Choose a factor")
            
    elif model == "LSTM":
        
        st.write("According to Korstanje in his book, Advanced Forecasting with Python: “The LSTM cell adds long-term memory in an even more performant way because it allows even more parameters to be learned. This makes it the most powerful [Recurrent Neural Network] to do forecasting, especially when you have a longer-term trend in your data. LSTMs are one of the state-of-the-art models for forecasting at the moment,” (2021)")
        st.write("What is a neural network ?")
        st.write("A neural network is a structure of layer of neural network connected. It is not an algorithm but a combination of algorithms which allow us to do complex operations on the data.")
        st.write("What is a recurrent neural network ?")
        st.write("There is a class of neural network concepted to treat time series. The neurals of RNN have cellular memory, and the input is taken depending on this internal state, which is realized thanks to the loops of the neural network. It exists reccurent module of 'tanh' layers in the RNN which allow them to keep the information. However, not for a long time, this is why we need LSTM.")
        st.write("What is LSTM ?")
        st.write("This is a particular type of recurrent neural network which is able to learn dependencie over long time on the data. This is done because the recurent module of the model has a combination of 4 layers interacting between each other.")
    
    else :
        
        st.write("Choose a model")
        
elif pages == "Setting of the models ⚙️" :
    
    model = st.sidebar.selectbox('Select the model', ['ARIMA', 'LSTM'])
    
    if model == "ARIMA":
        
        col1, col2 = st.columns(2)
        p = col1.selectbox("Select the number of P", [0,1,2,3,4,5,6,7,8,9,10])
        q = col2.selectbox("Select the number of Q", [0,1,2,3,4,5,6,7,8,9,10])
          
        ARIMA_Model,Prediction_arima = train_ARIMA(p,q)
            
        information = st.sidebar.selectbox("Select the information you want", ["Summary","Residuals","Accuracy Metrics"])
            
        if information == "Summary":
            
            st.write(ARIMA_Model.summary())
            
        elif information == "Residuals":
                
            residuals = pd.DataFrame(ARIMA_Model.resid)
            fig = ARIMA_Model.plot_diagnostics(figsize=(10,8))
            st.pyplot(fig)
            
        elif information == "Accuracy Metrics":
            
            st.dataframe(forecast_accuracy("ARIMA",Prediction_arima, test_arima))
    
    elif model == "LSTM" :
        
        st.write("In work, I have to discover how to parallelize the fitting of different model of LSTM in order that it is not too much computing intensive")
        
elif pages == "Forecasting 📈":
    
    model = st.sidebar.selectbox('Select the model', ['ARIMA', 'LSTM'])
    
    if model == "ARIMA":
        
        col1, col2 = st.columns(2)
        p = col1.selectbox("Select the number of P", [0,1,2,3,4,5,6,7,8,9,10])
        q = col2.selectbox("Select the number of Q", [0,1,2,3,4,5,6,7,8,9,10])
        arima_model = ARIMA(df.Close, order=(p,d_optimum,q))
        ARIMA_Model, Prediction_arima = train_ARIMA(p,q)
        #plot fitted values
        graph_prediction("Model ARIMA", Date_train_arima,train_arima, Date_test_arima,test_arima, Prediction_arima)
        st.write("Maybe I forecast on a too long period this is why it is totally out of context")
        
        
    else :
        
        #Visualize the data
        graph_prediction("Model LSTM", Date_train_LSTM,train, Date_test_LSTM, valid["Close"], valid["Predictions"])
        
else :
    
    st.write("Choose a page")
    