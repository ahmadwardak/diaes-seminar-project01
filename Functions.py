import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays
import itertools

#ARIMAX Models packages
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

#GARCH Model Package
from arch import arch_model

#Neural Network packages
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import l2

def calc_supply(df_gen):   
    """
    Calculate Total Power Generation Column also known as Sypply
    :param df_gen: Power Generation DataFrame
    :return: Power Generation DataFrame with the new column
    """
    
    df_gen['TotalGeneration[MWh]'] = (df_gen['Biomass[MWh]'] + 
            df_gen['Hydropower[MWh]']+
            df_gen['Wind offshore[MWh]'] +
            df_gen['Wind onshore[MWh]']+
            df_gen['Photovoltaics[MWh]']+
            df_gen['Other renewable[MWh]']+ 
            df_gen['Nuclear[MWh]']+ 
            df_gen['Fossil brown coal[MWh]']+
            df_gen['Fossil hard coal[MWh]']+ 
            df_gen['Fossil gas[MWh]']+ 
            df_gen['Hydro pumped storage[MWh]']+
            df_gen['Other conventional[MWh]'])
    return df_gen

def convert_gwh(df_gen):
    """
    Convert all columns to GWh                            
    :param df_gen: Power Generation DataFrame
    :return: Power Generation DataFrame
    """
    df_gen['TotalGeneration[MWh]']/=1000  
    df_gen['Biomass[MWh]']/=1000
    df_gen['Hydropower[MWh]']/=1000
    df_gen['Wind offshore[MWh]']/=1000
    df_gen['Wind onshore[MWh]']/=1000
    df_gen['Photovoltaics[MWh]']/=1000
    df_gen['Other renewable[MWh]']/=1000
    df_gen['Nuclear[MWh]']/=1000
    df_gen['Fossil brown coal[MWh]']/=1000
    df_gen['Fossil hard coal[MWh]']/=1000
    df_gen['Fossil gas[MWh]']/=1000
    df_gen['Hydro pumped storage[MWh]']/=1000
    df_gen['Other conventional[MWh]']/=1000
    #Rename Columns
    df_gen.rename(columns={'TotalGeneration[MWh]': 'TotalGeneration[GWh]',
                           'Biomass[MWh]':'Biomass[GWh]',
                           'Hydropower[MWh]': 'Hydropower[GWh]',
                           'Wind offshore[MWh]': 'Wind offshore[GWh]',
                           'Wind onshore[MWh]':'Wind onshore[GWh]',
                           'Photovoltaics[MWh]':'Photovoltaics[GWh]',
                           'Other renewable[MWh]':'Other renewable[GWh]',
                           'Nuclear[MWh]':'Nuclear[GWh]',
                           'Fossil brown coal[MWh]':'Fossil brown coal[GWh]',
                           'Fossil hard coal[MWh]':'Fossil hard coal[GWh]',
                           'Fossil gas[MWh]':'Fossil gas[GWh]',
                           'Hydro pumped storage[MWh]':'Hydro pumped storage[GWh]',
                           'Other conventional[MWh]':'Other conventional[GWh]'                      
                          }, inplace=True)
    return df_gen

def downsample_df(df, aggr ='sum' ):
    """
    Downsample the Data Frames to daily data
    :param df: DataFrame
    :return: Downsampled DataFrame
    """
    if aggr == 'sum':
        return df.resample('D').sum()
    else:
        return df.resample('D').median()
        
    

def set_timeIndex(df):
    """
    Set time as a Timeseries index for performance purposes
    :param df: DataFrame
    :return: Timeseries DataFrame
    """
    #Concatenate Date + Time and create TimeSeries
    df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time of day'])
    #Create Timeseries index for 
    #Drop Unused column
    df.drop('Time of day', axis=1, inplace=True)
    #Create Timeseries index for
    df.set_index('Date', inplace=True)
    return df

def test_stationarity(timeseries, window=90):
    """
    Applies the Dickey-Fuller test to check if the data is stationary
    :param df: Time Series DataFrame
    :return: No returns, just print out the results of the test and the plot of the stationary data 
    """
    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput) 
    
def gridsearch_sarima(df):
    """
    Iteratively test the model Sarima ( Seasonal Auto-Regressive Integrated Moving Average)
    :param df: Time Series DataFrame
    :return: Best Parameters for the Model 
    """
    #set parameter range
    p = range(0,3)
    q = range(1,3)
    d = range(1,3)
    s = range(6,8)
    # list of all parameter combos
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(p, d, q, s))
    best_results = '' 
    best_aic = 100000000
    # SARIMA model pipeline
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(df['Germany/Luxembourg[€/MWh]'],
                                        order=param,
                                        trend='ct',
                                        seasonal_order=param_seasonal)
                results = mod.fit(max_iter = 50, method = 'powell')
                print('SARIMA{},{} - AIC:{}'.format(param, param_seasonal, results.aic))
                if results.aic < best_aic:
                    best_results = 'SARIMA{},{} - AIC:{}'.format(param, param_seasonal, results.aic)
                    best_aic = results.aic
                    best_param = param
                    best_param_seasonal = param_seasonal
            except:
                continue

    print('Best Results: SARIMA{},{} - AIC:{}'.format(best_param,best_param_seasonal, best_aic))
    return best_param, best_param_seasonal


def gridsearch_arima(df):
    """
    Iteratively test the model Arima (Auto-Regressive Integrated Moving Average)
    :param df: Time Series DataFrame
    :return: Best Parameters for the Model 
    """
    import itertools
    #set parameter range
    p = range(0,6)
    q = range(0,6)
    d = range(0,6)
    # list of all parameter combos
    pdq = list(itertools.product(p, d, q))
    best_results = '' 
    best_aic = 100000000
    # ARIMA model pipeline
    for param in pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df['Germany/Luxembourg[€/MWh]'],
                                            order=param, 
                                            enforce_invertibility=False)
            results = mod.fit(max_iter = 50, method = 'powell')
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            if results.aic < best_aic:
                best_results = 'ARIMA{} - AIC:{}'.format(param,results.aic)
                best_aic = results.aic
                best_param = param
        except:
            continue

    print('Best Results: ARIMA{} - AIC:{}'.format(best_param, best_aic))
    return best_param


def nn_model(df_NN):
    """
    Define and train the Neural Network model
    :param df: Time Series DataFrame in the NN Format
    #Format:
    #SARIMAX-GARCH/ARMAX-GARCH model price prediction on day T;  
    #historical price on day T-1; 
    #historical price on day T-7; 
    #historical price on day T-14; 
    #non-base electricity demand on day T;  
    #non-base electricity demand on day T-1.  
    :return: scaler_x, scaler_y, model 
    """
    feature = 'Germany/Luxembourg[€/MWh]'
    dim = len(df_NN.columns) - 1


    #SPLIT TRAINING AND VALIDATION SET
    X_train, X_val, y_train, y_val = train_test_split(df_NN, df_NN.pop(feature))
    #SCALE X AND Y FOR TRAINING AND VALIDATION DS
    scaler_x = QuantileTransformer()
    scaler_y = QuantileTransformer()

    y_train=y_train.values.reshape(-1,1)
    y_val=y_val.values.reshape(-1,1)

    print(scaler_x.fit(X_train))
    xtrain_scale=scaler_x.transform(X_train)

    print(scaler_x.fit(X_val))
    xval_scale=scaler_x.transform(X_val)

    print(scaler_y.fit(y_train))
    ytrain_scale=scaler_y.transform(y_train)

    print(scaler_y.fit(y_val))
    yval_scale=scaler_y.transform(y_val)

    #CONFIGURE HYPERPARAMETER
    size = 48 

        #DEFINE NN ARCHITECTURE
    model = keras.Sequential([
        keras.layers.Dense(50, activation='relu',input_shape=(dim,)),
        keras.layers.Dense(size*10, activation='relu'),
        keras.layers.Dropout(.1), #here
        keras.layers.Dense(13, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
        keras.layers.Dense(size*9, activation='relu'),
        keras.layers.Dense(13, kernel_regularizer='l2'),#hyperparameter optimization
        keras.layers.Dense(size*8, activation='relu'),
        keras.layers.Dense(13, kernel_regularizer='l1'),
        keras.layers.Dense(size*7, activation='relu'),
        # NOT USE REGULARIZATION AT THE LAST LAYER
        keras.layers.Dense(1, activation='linear') 
        ])

    #SHOW MODEL CHARACTERISTICS
    model.summary()

    opt = keras.optimizers.Adam(learning_rate=0.0005)

    #COMPILE MODEL
    model.compile(#optimizer='adam',
                  #optimizer='sgd',
                  optimizer=opt,
                  loss=tf.keras.losses.MeanSquaredError(),
                  #metrics=['accuracy']
                  metrics=['mse','mae','mean_absolute_percentage_error'])
    #RUN MODEL
    history =  model.fit(xtrain_scale, ytrain_scale, epochs=50, validation_split=0.25, batch_size=300)
    print(history.history.keys())
    print('PLOT TRAINING/VALIDATION CURVE AGAINST LOSS') 
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    print('RUN PREDICTIONS ON VALIDATION DS')
    predictions = model.predict(xval_scale) 
    predictions = scaler_y.inverse_transform(predictions)

    print("y-val mean: " + str(np.mean(y_val)))
    print("predictions mean: " + str(np.mean(predictions)))
    print("mean squared error: " + str(mean_squared_error(y_val,predictions)))

    print("PLOT REAL VS PREDICTED FOR EACH SAMPLE") 

    plt.xlabel("Row Index")
    plt.ylabel(feature)
    plt.plot(y_val) 
    plt.plot(predictions)
    plt.legend(['real', 'predicted'], loc='upper left')
    plt.figure(figsize=(20,8))
    plt.show()

    #features.remove(feature)
    plt.xlabel("error distribution")
    plt.hist(y_val - predictions) 
    plt.figure(figsize=(20,8))
    plt.show()
    
    return scaler_x, scaler_y, model
