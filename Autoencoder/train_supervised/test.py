import random as rn

import numpy as np
import tensorflow as tf

from Autoencoder.train_supervised.prepare_datasets import process_dataset

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
