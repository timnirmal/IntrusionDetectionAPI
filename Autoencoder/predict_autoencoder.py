import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import random as rn
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split
from datetime import datetime


def mad_score(points):
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)

    if mad != 0 and not np.isnan(mad):
        return 0.6745 * ad / mad
    else:
        return 0.6745 * ad / 0.0001


def calculate_z_score(THRESHOLD, mse):
    z_scores = mad_score(mse)
    # outliers = np.where(z_scores > THRESHOLD)
    outliers = z_scores > THRESHOLD

    return outliers


def predict_autoencoder(df):
    # load the model
    autoencoder = tf.keras.models.load_model('Autoencoder/autoencoder_best_weights.hdf5')

    # predict on the dataset
    predictions = autoencoder.predict(df)

    # calculate the mean squared error
    mse = np.mean(np.power(df - predictions, 2), axis=1)
    print(mse)
    error_df = pd.DataFrame({'Reconstruction_error': mse})
    print(error_df)

    THRESHOLD = 1000
    outliers = calculate_z_score(THRESHOLD, mse)

    return outliers
