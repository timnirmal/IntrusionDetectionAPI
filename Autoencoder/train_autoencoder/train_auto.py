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

from prepare_datasets import process_dataset
from autoencoder import autoenocder_model, load_data, save_data, plot_auto_train, train_autoencoder


if __name__ == '__main__':
    ########################### PREPARE DATASET ##############################
    data_path = "D:\Projects\IntrusionDetection\old\create_dataset\class_final.csv"
    X_train, y_train, X_validate, y_validate, X_test, y_test = process_dataset(data_path)

    save_data(X_train, y_train, X_validate, y_validate, X_test, y_test)
    X_train, y_train, X_validate, y_validate, X_test, y_test = load_data()

    ########################### AUTO ENCODER ##############################
    input_dim = X_train.shape[1]
    autoencoder = autoenocder_model(input_dim)

    ########################### TRAIN AUTO ENCODER ##############################
    autoencoder, history = train_autoencoder(autoencoder, X_train, X_validate)
    plot_auto_train(history)





