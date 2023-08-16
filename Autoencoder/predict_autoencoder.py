import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf


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

    # print(df)

    # predict on the dataset
    predictions = autoencoder.predict(df)

    # calculate the mean squared error
    mse = np.mean(np.power(df - predictions, 2), axis=1)
    # print(mse)
    error_df = pd.DataFrame({'Reconstruction_error': mse})
    # print(error_df)

    THRESHOLD = 1000
    outliers = calculate_z_score(THRESHOLD, mse)

    return outliers


def predict_rf(df):
    try:
        # print current working directory
        # print(os.getcwd())
        # print ls
        # print(os.listdir())
        # load the model Autoencoder/models/encoder.pkl
        encoder = pickle.load(open('Autoencoder/models/encoder.pkl', 'rb'))
        # print(encoder)
    except Exception as e:
        print(e)

    try:
        # print current working directory
        # print(os.getcwd())
        # print ls
        # print(os.listdir())
        # load the model Autoencoder/models/encoder.pkl
        autoencoder = tf.keras.models.load_model('Autoencoder/autoencoder_best_weights.hdf5')
        # print(autoencoder)
    except Exception as e:
        print(e)

    try:
        # load random forest model
        rf_model = pickle.load(open('Autoencoder/models/rf_auto.pkl', 'rb'))
        # print(rf_model)
    except Exception as e:
        print(e)

    try:
        # load le
        label_encoder = pickle.load(open('Autoencoder/label_encoder.pkl', 'rb'))
    except Exception as e:
        print(e)


    # Failed to convert a NumPy array to a Tensor (Unsupported object type int).
    # fix
    # df = df.astype('float32')

    # print(autoencoder.summary())
    # get encoder part from autoencoder
    encoder = tf.keras.models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_1').output)
    print("Everything loaded successfully")
    # print(df)
    predictions = encoder.predict(df)

    # # predict on the dataset
    # predictions = encoder.predict(df)
    # print(predictions)

    # predict with rf
    rf_pred = rf_model.predict(predictions)

    # label encoder
    rf_pred = label_encoder.inverse_transform(rf_pred)

    return rf_pred
