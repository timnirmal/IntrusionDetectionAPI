import pickle
import random as rn

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# For training the autoencoder for detect either a anomaly or normal data
def process_dataset(data_path):
    df = pd.read_csv(data_path)
    print("Dataset Loaded")

    # count number of each class
    print(df['Label'].value_counts())
    # 0     2273097
    # 4      231073
    # 10     158930
    # 2      128027
    # 3       10293
    # 7        7938
    # 11       5897
    # 6        5796
    # 5        5499
    # 1        1966
    # 12       1507
    # 14        652
    # 9          36
    # 13         21
    # 8          11

    ######################### converting label to normal or anomaly #############################
    # if Label != 0, then Label = 1
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 0 else 1)

    ######################### prepare datasets #############################

    # remove Label column
    y = df['Label'].values
    X = df.drop('Label', axis=1).values

    ######################## normalize data #############################
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # save scaler
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

    # create df with X and y -> dataframe after normalization
    df = pd.DataFrame(X)
    df['Label'] = y

    # count number of each class
    print(df['Label'].value_counts())
    # 0    2273097
    # 1     557646

    # manual parameters
    RANDOM_SEED = 42
    VALIDATE_SIZE = 0.2

    # setting random seeds for libraries to ensure reproducibility
    np.random.seed(RANDOM_SEED)
    rn.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # randomize df
    df = df.sample(frac=1, random_state=RANDOM_SEED)

    df_normal = df[df['Label'] == 0]
    df_attack = df[df['Label'] == 1]

    TESTING_NORMAL_SAMPLE = int(df_attack.shape[0] * 1.5)

    ################################### last set of normal transactions + all attacks ###################################
    test = df_normal.iloc[:TESTING_NORMAL_SAMPLE]
    test = pd.concat([test, df_attack], axis=0)
    test = test.sample(frac=1, random_state=RANDOM_SEED)

    # rest of the normal transactions
    train = df_normal.iloc[TESTING_NORMAL_SAMPLE:]

    train, validate = train_test_split(train, test_size=VALIDATE_SIZE, random_state=RANDOM_SEED)

    X_train, y_train = train.drop('Label', axis=1), train['Label']
    X_validate, y_validate = validate.drop('Label', axis=1), validate['Label']
    X_test, y_test = test.drop('Label', axis=1), test['Label']

    print(test["Label"].value_counts())
    # 0    836469
    # 1    557646
    print(train["Label"].value_counts())  # 0    1149302
    print(validate["Label"].value_counts())  # 0    287326

    print(f"""Shape of the datasets:
        training (rows, cols) = {X_train.shape}
        validate (rows, cols) = {X_validate.shape}
        holdout  (rows, cols) = {X_test.shape}""")
    #     training (rows, cols) = (1149302, 72)
    #     validate (rows, cols) = (287326, 72)
    #     holdout  (rows, cols) = (1394115, 72)

    X_train.columns = X_train.columns.astype(str)
    X_validate.columns = X_validate.columns.astype(str)

    return X_train, y_train, X_validate, y_validate, X_test, y_test
