from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

BATCH_SIZE = 256
EPOCHS = 100


def callback_set(autoencoder):
    # current date and time
    yyyymmddHHMM = datetime.now().strftime('%Y%m%d%H%M')

    # new folder for a new run
    log_subdir = f'{yyyymmddHHMM}_batch{BATCH_SIZE}_layers{len(autoencoder.layers)}'

    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    save_model = tf.keras.callbacks.ModelCheckpoint(
        filepath='autoencoder_best_weights.hdf5',
        save_best_only=True,
        monitor='val_loss',
        verbose=0,
        mode='min'
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        f'logs/{log_subdir}',
        batch_size=BATCH_SIZE,
        update_freq='batch'
    )

    # callbacks argument only takes a list
    cb = [early_stop, save_model, tensorboard]

    return cb


def autoenocder_model(input_dim):
    autoencoder = tf.keras.models.Sequential([
        # deconstruct / encode
        tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(8, activation='elu'),
        tf.keras.layers.Dense(4, activation='elu'),
        tf.keras.layers.Dense(2, activation='elu'),
        # reconstruction / decode
        tf.keras.layers.Dense(4, activation='elu'),
        tf.keras.layers.Dense(8, activation='elu'),
        tf.keras.layers.Dense(16, activation='elu'),
        tf.keras.layers.Dense(input_dim, activation='elu')
    ])

    # https://keras.io/api/models/model_training_apis/
    autoencoder.compile(optimizer="adam", loss="mse", metrics=["acc"])

    # print an overview of our model
    autoencoder.summary()

    return autoencoder


def train_autoencoder(autoencoder, X_train, X_validate):
    auto_history = autoencoder.fit(
        X_train, X_train,
        shuffle=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callback_set(autoencoder),
        validation_data=(X_validate, X_validate)
    )
    autoencoder.save('autoencoder.h5')

    return autoencoder, auto_history


def plot_auto_train(auto_history):
    # plot the training loss
    plt.plot(auto_history.history["loss"], label="Training Loss")
    plt.plot(auto_history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses Over Time")
    plt.legend()
    plt.savefig("autoencoder_loss.png")
    plt.show()
    # plot the training accuracy
    plt.plot(auto_history.history["acc"], label="Training Accuracy")
    plt.plot(auto_history.history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Over Time")
    plt.legend()
    plt.savefig("autoencoder_acc.png")
    plt.show()


def save_data(X_train, y_train, X_validate, y_validate, X_test, y_test):
    # save datasets
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_validate.npy', X_validate)
    np.save('data/y_validate.npy', y_validate)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)


def load_data():
    try:
        # load datasets
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_validate = np.load('data/X_validate.npy')
        y_validate = np.load('data/y_validate.npy')
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
    except:
        data_path = r"D:\Projects\IntrusionDetection\Autoencoder\train_autoencoder\data"
        X_train = np.load(data_path + '\X_train.npy')
        y_train = np.load(data_path + '\y_train.npy')
        X_validate = np.load(data_path + '\X_validate.npy')
        y_validate = np.load(data_path + '\y_validate.npy')
        X_test = np.load(data_path + '\X_test.npy')
        y_test = np.load(data_path + '\y_test.npy')

    return X_train, y_train, X_validate, y_validate, X_test, y_test
