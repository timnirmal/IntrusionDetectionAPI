import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import random as rn
import pickle

from Autoencoder.train_autoencoder.predict_auto import calculate_z_score

def predict_autoencoder(df):
    # load model
    autoencoder = tf.keras.models.load_model(r'D:\Projects\IntrusionDetectionAPI\Autoencoder\train_autoencoder\autoencoder.h5')

    # predict
    prediction = autoencoder.predict(df)

    # calculate the loss
    mse = np.mean(np.power(df - prediction, 2), axis=1)
    print(mse)
    error_df = pd.DataFrame({'Reconstruction_error': mse})
    print(error_df)

    THRESHOLD = 1000
    outliers = calculate_z_score(THRESHOLD, mse)

    return outliers
