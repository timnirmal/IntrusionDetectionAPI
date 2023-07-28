import matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import random as rn

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import train_test_split
from datetime import datetime

from Autoencoder.train_autoencoder.prepare_datasets import process_dataset
from Autoencoder.train_autoencoder.autoencoder import autoenocder_model, load_data, save_data, plot_auto_train, train_autoencoder


def predict_ocsvm(df):
    # load model
    ocsvm_model = pickle.load(open('models/ocsvm_model.pkl', 'rb'))

    # Predict the anomalies
    prediction = ocsvm_model.predict(df)

    # Change the anomalies' values to make it consistent with the true values
    prediction = [1 if i == -1 else 0 for i in prediction]

    return prediction

if __name__ == '__main__':
    prediction = predict_ocsvm(df)