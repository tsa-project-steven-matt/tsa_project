import math
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd

def plot_data_and_predictions(predictions, label):
    plt.figure(figsize=(10, 8))

    plt.plot(train,label='Train')
    plt.plot(test, label='Test')
    plt.plot(predictions, label=label, linewidth=5)

    plt.legend(loc='best')
    plt.show()


def evaluate(actual, predictions, output=True):
    mse = metrics.mean_squared_error(actual, predictions)
    rmse = math.sqrt(mse)

    if output:
        print('MSE:  {}'.format(mse))
        print('RMSE: {}'.format(rmse))
    else:
        return mse, rmse    

def plot_and_eval(predictions, actual, train, test, metric_fmt='{:.2f}', linewidth=4):
    if type(predictions) is not list:
        predictions = [predictions]

    plt.figure(figsize=(16, 8))
    plt.plot(train,label='Train')
    plt.plot(test, label='Test')

    for yhat in predictions:
        mse, rmse = evaluate(actual, yhat, output=False)        
        label = f'{yhat.name}'
        if len(predictions) > 1:
            label = f'{label} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse)
        plt.plot(yhat, label=label, linewidth=linewidth)

    if len(predictions) == 1:
        label = f'{label} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse)
        plt.title(label)

    plt.legend(loc='best')
    plt.show()    


def impute_missing_prophet(d_df, list):
    d_df['y'] = df.calories_burned
    d_df = d_df[['ds', 'y']]
    d_df['cap'] = 3700
    d_df['floor'] = 2000
    m = Prophet(growth='logistic', weekly_seasonality=True)
    m.fit(d_df)
    future = m.make_future_dataframe(periods=180)
    future['cap'] = 3700
    future['floor'] = 2000
    forecast = m.predict(future)

    missing_vals = forecast.loc[225:246][['ds','yhat']]
    missing_vals.rename(columns={'yhat': 'calories_burned', 'ds': 'date'}, inplace=True)

    for feature in list:
        floor = float(df[[feature]].quantile(.05))
        cap = float(df[[feature]].quantile(.75))
        d_df['y'] = df[feature]
        d_df = d_df[['ds', 'y']]
        d_df['cap'] = cap
        d_df['floor'] = floor
        m = Prophet(growth='logistic', weekly_seasonality=True)
        m.fit(d_df)

        future = m.make_future_dataframe(periods=180)
        future['cap'] = cap
        future['floor'] = floor

        forecast = m.predict(future)

        missing_pred = forecast[['yhat']].loc[225:246][['yhat']]

        missing_pred.rename(columns={'yhat': feature}, inplace=True)
        missing_vals[feature] = missing_pred[feature]
    return missing_vals

def impute_missing_rolling(df, list):
    missing_vals = pd.date_range(start='12/07/2018', periods=22)
    missing_vals = pd.DataFrame(missing_vals)
    missing_vals = missing_vals.rename(columns={0:'date'})
    for feature in list:
        train = df[:'2018-11-08'][feature].resample('D').agg('mean')
        test = df['2018-11-08':][feature].resample('D').agg('mean')
        yhat = pd.DataFrame(dict(actual=test))
        yhat['moving_avg_forecast_7'] = train.rolling(7).mean().iloc[-1]
        missing_vals[feature] = float(yhat.moving_avg_forecast_7.max())
    return missing_vals