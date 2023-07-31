import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score

from Autoencoder.train_autoencoder.autoencoder import load_data


def plot_2D(X_test):
    # use PCA to reduce the dimensionality of the data
    pca = PCA(n_components=2)
    X_test = pca.fit_transform(X_test)
    # Put the testing dataset and predictions in the same dataframe
    df_test = pd.DataFrame(X_test, columns=['feature1', 'feature2'])
    df_test['y_test'] = y_test
    df_test['prediction'] = prediction
    df_test['customized_prediction'] = customized_prediction
    # Visualize the actual and predicted anomalies
    plt.figure(figsize=(10, 8))
    # Plot the training data with label 'Train data'
    plt.scatter(df_test[df_test['y_test'] == 0]['feature1'], df_test[df_test['y_test'] == 0]['feature2'],
                label='Train data')
    # Plot the testing data with label 'Test data'
    plt.scatter(df_test[df_test['y_test'] == 1]['feature1'], df_test[df_test['y_test'] == 1]['feature2'],
                label='Test data')
    # Plot the predicted anomalies with label 'Predicted anomalies'
    plt.scatter(df_test[df_test['prediction'] == 1]['feature1'], df_test[df_test['prediction'] == 1]['feature2'],
                label='Predicted anomalies')
    # Plot the predicted anomalies with label 'Customized predicted anomalies'
    plt.scatter(df_test[df_test['customized_prediction'] == 1]['feature1'],
                df_test[df_test['customized_prediction'] == 1]['feature2'],
                label='Customized predicted anomalies')
    plt.legend()
    plt.show()


import plotly.graph_objects as go


def plot_3D(X_test):
    # use PCA to reduce the dimensionality of the data to 3 features
    pca = PCA(n_components=3)
    X_test = pca.fit_transform(X_test)
    # Put the testing dataset and predictions in the same dataframe
    df_test = pd.DataFrame(X_test, columns=['feature1', 'feature2', 'feature3'])
    df_test['y_test'] = y_test
    df_test['prediction'] = prediction
    df_test['customized_prediction'] = customized_prediction

    # Create a 3D scatter plot
    fig = go.Figure()

    # Plot the training data
    fig.add_trace(go.Scatter3d(
        x=df_test[df_test['y_test'] == 0]['feature1'],
        y=df_test[df_test['y_test'] == 0]['feature2'],
        z=df_test[df_test['y_test'] == 0]['feature3'],
        mode='markers',
        name='Train data',
        marker=dict(color='blue', size=5),
        showlegend=True
    ))

    # Plot the testing data
    fig.add_trace(go.Scatter3d(
        x=df_test[df_test['y_test'] == 1]['feature1'],
        y=df_test[df_test['y_test'] == 1]['feature2'],
        z=df_test[df_test['y_test'] == 1]['feature3'],
        mode='markers',
        name='Test data',
        marker=dict(color='orange', size=5),
        showlegend=True
    ))

    # Plot the predicted anomalies
    fig.add_trace(go.Scatter3d(
        x=df_test[df_test['prediction'] == 1]['feature1'],
        y=df_test[df_test['prediction'] == 1]['feature2'],
        z=df_test[df_test['prediction'] == 1]['feature3'],
        mode='markers',
        name='Predicted anomalies',
        marker=dict(color='red', size=5),
        showlegend=True
    ))

    # Plot the customized predicted anomalies
    fig.add_trace(go.Scatter3d(
        x=df_test[df_test['customized_prediction'] == 1]['feature1'],
        y=df_test[df_test['customized_prediction'] == 1]['feature2'],
        z=df_test[df_test['customized_prediction'] == 1]['feature3'],
        mode='markers',
        name='Customized predicted anomalies',
        marker=dict(color='green', size=5),
        showlegend=True
    ))

    # Add an empty trace for the legend
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=0),
        showlegend=True,
        name='_'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Feature 3',
        ),
        legend=dict(
            x=0.85,
            y=0.95
        )
    )

    fig.show()
    fig.write_html("3D.html")
    fig.write_image("3D.png")


if __name__ == '__main__':
    # load data
    X_train, y_train, X_validate, y_validate, X_test, y_test = load_data()

    print(f"X_test: {X_test.shape}", f"y_test: {y_test.shape}")
    # label count in X_test
    unique, counts = np.unique(y_test, return_counts=True)
    print("label count in X_test", dict(zip(unique, counts)))

    TRAIN_SIZE = int(len(X_test) * 0.1)

    # keep only 0.01% of the data
    X_train = X_test[:TRAIN_SIZE]
    y_train = y_test[:TRAIN_SIZE]
    X_test = X_test[TRAIN_SIZE:TRAIN_SIZE + int(len(X_test) * 0.02)]
    y_test = y_test[TRAIN_SIZE:TRAIN_SIZE + int(len(y_test) * 0.02)]

    print(f"X_train: {X_train.shape}", f"y_train: {y_train.shape}")
    # label count in X_train
    unique, counts = np.unique(y_train, return_counts=True)
    print("label count in X_train", dict(zip(unique, counts)))

    print(f"X_test: {X_test.shape}", f"y_test: {y_test.shape}")
    # label count in X_test
    unique, counts = np.unique(y_test, return_counts=True)
    print("label count in X_test", dict(zip(unique, counts)))
    #
    # # train model
    # ocsvm_model = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto', verbose=True, shrinking=False)
    # ocsvm_model.fit(X_train)
    #
    # # save model
    # pickle.dump(ocsvm_model, open('models/ocsvm_model.pkl', 'wb'))

    # load model
    ocsvm_model = pickle.load(open('models/ocsvm_model.pkl', 'rb'))

    # Predict the anomalies
    prediction = ocsvm_model.predict(X_test)

    # Change the anomalies' values to make it consistent with the true values
    prediction = [1 if i == -1 else 0 for i in prediction]
    # Check the model performance
    print(classification_report(y_test, prediction))

    # Get the scores for the testing dataset
    score = ocsvm_model.score_samples(X_test)
    # Check the score for 2% of outliers
    score_threshold = np.percentile(score, 4)
    print(f'The customized score threshold for {4}% of outliers is {score_threshold:.2f}')
    # Check the model performance at 2% threshold
    customized_prediction = [1 if i < score_threshold else 0 for i in score]
    print(classification_report(y_test, customized_prediction))
    accuracy = accuracy_score(y_test, customized_prediction)

    # # Get the scores for the testing dataset
    # score = ocsvm_model.score_samples(X_test)
    # # Check the score for 2% of outliers
    # score_threshold = np.percentile(score, 2)
    # print(f'The customized score threshold for 2% of outliers is {score_threshold:.2f}')
    # # Check the model performance at 2% threshold
    # customized_prediction = [1 if i < score_threshold else 0 for i in score]
    # print(classification_report(y_test, customized_prediction))

    # plot_2D(X_test)
    # plot_3D(X_test)

# D:\Projects\IntrusionDetection\venv\Scripts\python.exe D:\Projects\IntrusionDetection\Autoencoder\train_ocsvm\train_ocsvm.py
# X_test: (1394115, 72) y_test: (1394115,)
# label count in X_test {0: 836469, 1: 557646}
# X_train: (139411, 72) y_train: (139411,)
# label count in X_train {0: 83507, 1: 55904}
# X_test: (27882, 72) y_test: (27882,)
# label count in X_test {0: 16787, 1: 11095}
# [LibSVM]...............
# Warning: using -h 0 may be faster
# *.
# Warning: using -h 0 may be faster
# *.
# Warning: using -h 0 may be faster
# *
# optimization finished, #iter = 15866
# obj = 88495778.758217, rho = 12792.100953
# nSV = 13943, nBSV = 13938
#               precision    recall  f1-score   support
#
#            0       0.61      0.91      0.73     16787
#            1       0.49      0.13      0.20     11095
#
#     accuracy                           0.60     27882
#    macro avg       0.55      0.52      0.47     27882
# weighted avg       0.57      0.60      0.52     27882

# D:\Projects\IntrusionDetection\venv\Scripts\python.exe D:\Projects\IntrusionDetection\Autoencoder\train_ocsvm\train_ocsvm.py
# X_test: (1394115, 72) y_test: (1394115,)
# label count in X_test {0: 836469, 1: 557646}
# X_train: (139411, 72) y_train: (139411,)
# label count in X_train {0: 83507, 1: 55904}
# X_test: (27882, 72) y_test: (27882,)
# label count in X_test {0: 16787, 1: 11095}
#               precision    recall  f1-score   support
#
#            0       0.61      0.91      0.73     16787
#            1       0.49      0.13      0.20     11095
#
#     accuracy                           0.60     27882
#    macro avg       0.55      0.52      0.47     27882
# weighted avg       0.57      0.60      0.52     27882
#
# The customized score threshold for 1% of outliers is 12595.78
#               precision    recall  f1-score   support
#
#            0       0.60      0.99      0.75     16787
#            1       0.49      0.01      0.02     11095
#
#     accuracy                           0.60     27882
#    macro avg       0.55      0.50      0.39     27882
# weighted avg       0.56      0.60      0.46     27882
#
# Accuracy: 0.6019654257226885
# --------------------------------------------------
# The customized score threshold for 2% of outliers is 12646.20
#               precision    recall  f1-score   support
#
#            0       0.61      0.99      0.75     16787
#            1       0.64      0.03      0.06     11095
#
#     accuracy                           0.61     27882
#    macro avg       0.62      0.51      0.41     27882
# weighted avg       0.62      0.61      0.48     27882
#
# Accuracy: 0.6077397604189082
# --------------------------------------------------
# The customized score threshold for 3% of outliers is 12660.92
#               precision    recall  f1-score   support
#
#            0       0.61      0.99      0.76     16787
#            1       0.73      0.05      0.10     11095
#
#     accuracy                           0.62     27882
#    macro avg       0.67      0.52      0.43     27882
# weighted avg       0.66      0.62      0.50     27882
#
# Accuracy: 0.6157377519546661
# --------------------------------------------------
# The customized score threshold for 4% of outliers is 12678.63
#               precision    recall  f1-score   support
#
#            0       0.61      0.98      0.75     16787
#            1       0.69      0.07      0.13     11095
#
#     accuracy                           0.62     27882
#    macro avg       0.65      0.52      0.44     27882
# weighted avg       0.64      0.62      0.50     27882
#
# Accuracy: 0.6172082347033929
# --------------------------------------------------
# The customized score threshold for 5% of outliers is 12740.95
#               precision    recall  f1-score   support
#
#            0       0.61      0.97      0.75     16787
#            1       0.64      0.08      0.14     11095
#
#     accuracy                           0.62     27882
#    macro avg       0.63      0.53      0.45     27882
# weighted avg       0.62      0.62      0.51     27882
#
# Accuracy: 0.6158812136862492
# --------------------------------------------------
# The customized score threshold for 6% of outliers is 12771.11
#               precision    recall  f1-score   support
#
#            0       0.62      0.96      0.75     16787
#            1       0.61      0.09      0.16     11095
#
#     accuracy                           0.62     27882
#    macro avg       0.62      0.53      0.46     27882
# weighted avg       0.62      0.62      0.52     27882
#
# Accuracy: 0.6158094828204577
# --------------------------------------------------
# The customized score threshold for 7% of outliers is 12782.55
#               precision    recall  f1-score   support
#
#            0       0.61      0.95      0.75     16787
#            1       0.57      0.10      0.17     11095
#
#     accuracy                           0.61     27882
#    macro avg       0.59      0.52      0.46     27882
# weighted avg       0.60      0.61      0.52     27882
#
# Accuracy: 0.6115414963058604
# --------------------------------------------------
# The customized score threshold for 8% of outliers is 12786.99
#               precision    recall  f1-score   support
#
#            0       0.61      0.94      0.74     16787
#            1       0.53      0.11      0.18     11095
#
#     accuracy                           0.61     27882
#    macro avg       0.57      0.52      0.46     27882
# weighted avg       0.58      0.61      0.52     27882
#
# Accuracy: 0.6068431245965139
# --------------------------------------------------
# The customized score threshold for 9% of outliers is 12790.25
#               precision    recall  f1-score   support
#
#            0       0.62      0.93      0.74     16787
#            1       0.54      0.12      0.20     11095
#
#     accuracy                           0.61     27882
#    macro avg       0.58      0.53      0.47     27882
# weighted avg       0.58      0.61      0.52     27882
#
# Accuracy: 0.6085287999426153
# --------------------------------------------------
# Best threshold: 4
# Best accuracy: 0.6172082347033929
#
# Process finished with exit code 0
