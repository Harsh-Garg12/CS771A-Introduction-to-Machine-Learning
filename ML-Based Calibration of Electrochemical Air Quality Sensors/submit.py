import numpy as np
import pickle
import pandas as pd


# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict(df):
    # Load your model file
    filename = "my_model.pickle"
    loaded_model = pickle.load(open(filename, 'rb'))

    # Make two sets of predictions, one for O3 and another for NO2
    df['Time'] = (pd.to_datetime(df['Time']) - pd.to_datetime('2019-03-27 00:00:00')).dt.total_seconds()

    df['Time'] = (df['Time'] - 1.89277222e+06) / 9.95712701e+05
    df['temp'] = (df['temp'] - 3.17178775e+01) / 3.43227297e+00
    df['humidity'] = (df['humidity'] - 8.55640200e+01) / 2.29258803e+01
    df['no2op1'] = (df['no2op1'] - 1.89521750e+02) / 2.06560143e+01
    df['no2op2'] = (df['no2op2'] - 1.93762750e+02) / 1.84634521e+01
    df['o3op1'] = (df['o3op1'] - 1.99374900e+02) / 2.39995740e+01
    df['o3op2'] = (df['o3op2'] - 1.90473950e+02) / 1.93364558e+01

    X_test = np.array(df)

    y_pred = loaded_model.predict(X_test)
    pred_o3 = y_pred[:, 0]
    pred_no2 = y_pred[:, 1]
    pred_o3 = (pred_o3 * 18.81054117) + 30.05171475
    pred_no2 = (pred_no2 * 10.85105448) + 11.88291335
    # Return both sets of predictions
    return (pred_o3, pred_no2)
