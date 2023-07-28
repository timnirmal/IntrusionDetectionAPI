import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import random as rn

from Autoencoder.train_supervised.prepare_datasets import process_dataset, process_normal_attack_dataset, \
    equal_fraction_of_data

RANDOM_SEED = 42

# setting random seeds for libraries to ensure reproducibility
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if __name__ == '__main__':
    data_path = r"D:\Projects\IntrusionDetection\old\create_dataset\class_final.csv"

    df = process_dataset(data_path)
    # print(df)

    #
    # df_normal, df_attack = process_normal_attack_dataset(df)
    # # print(df_normal)
    # # print(df_attack)

    # equal_fraction_data = equal_fraction_of_data(df)
    #
    # # count label
    # print(equal_fraction_data['Label'].value_counts())

    # remove Label == 0
    df = df[df['Label'] != 0]

    #