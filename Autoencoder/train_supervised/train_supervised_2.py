import pickle
import random as rn
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from Autoencoder.train_supervised.prepare_datasets import process_dataset

RANDOM_SEED = 42

# setting random seeds for libraries to ensure reproducibility
np.random.seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

from loguru import logger

logger.add(f"file_{time.strftime('%Y-%m-%d')}.log", rotation="500 MB")

if __name__ == '__main__':
    data_path = r"D:\Projects\IntrusionDetection\old\create_dataset\class_final.csv"

    df = process_dataset(data_path)
    print(df)
    logger.info(df)

    #
    # df_normal, df_attack = process_normal_attack_dataset(df)
    # print(df_normal)
    # print(df_attack)

    # remove Label == 0
    df = df[df['Label'] != 0]

    # from each label get 10000 data
    df = df.groupby('Label').head(10000)
    print(df)
    logger.info(df)

    # count label
    print(df['Label'].value_counts())
    logger.info(df['Label'].value_counts())

    X = df.drop('Label', axis=1).values
    y = df['Label'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
    print(class_weights_dict)
    logger.info(class_weights_dict)

    # Create a list of models to evaluate
    models = [
        ('Random Forest', RandomForestClassifier(class_weight=class_weights_dict)),
        ('Gradient Boosting', GradientBoostingClassifier()),
        ('SVM', SVC(class_weight=class_weights_dict)),
        ('Logistic Regression', LogisticRegression(class_weight=class_weights_dict)),
        ('k-NN', KNeighborsClassifier())
    ]

    # Evaluate models using cross-validation and select the best model based on accuracy
    best_model = None
    best_model_name = None
    best_accuracy = 0.0

    for name, model in models:
        pipeline = Pipeline([
            # ('scaler', StandardScaler()),  # Standardize features (optional but often recommended)
            ('pca', PCA(n_components=0.95)),  # Dimensionality reduction with PCA (keep 95% of variance)
            (name, model)  # Model
        ])

        # Train the model on the full training set
        pipeline.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = pipeline.predict(X_test)

        # Evaluate the model on the test set
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} - Test accuracy: {accuracy:.4f}")
        logger.info(f"{name} - Test accuracy: {accuracy:.4f}")

        # save model with name
        pickle.dump(pipeline, open(f'{name}.pkl', 'wb'))

        if accuracy > best_accuracy:
            best_model = pipeline
            best_model_name = name
            best_accuracy = accuracy

    # save model with name
    pickle.dump(best_model, open('best_model.pkl', 'wb'))

    # Make predictions on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the best model on the test set
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")

    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1-score: {f1:.4f}")

    # print classification report
    print(classification_report(y_test, y_pred))
    logger.info(classification_report(y_test, y_pred))

    # confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    logger.info(cm)

    # plot confusion matrix
    import seaborn as sns

    sns.heatmap(cm, annot=True)
    plt.savefig('confusion_matrix.png')
    plt.show()

#
# # Assuming your dataframe is named 'df', and the target variable is labeled 'target'.
# X = df.drop('target', axis=1)
# y = df['target']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Create a list of models to evaluate
# models = [
#     ('Random Forest', RandomForestClassifier()),
#     ('Gradient Boosting', GradientBoostingClassifier()),
#     ('SVM', SVC()),
#     ('Logistic Regression', LogisticRegression()),
#     ('k-NN', KNeighborsClassifier())
# ]
#
# # Evaluate models using cross-validation and select the best model based on accuracy
# best_model = None
# best_model_name = None
# best_accuracy = 0.0
#
# for name, model in models:
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),  # Standardize features (optional but often recommended)
#         ('pca', PCA(n_components=0.95)),  # Dimensionality reduction with PCA (keep 95% of variance)
#         (name, model)  # Model
#     ])
#
#     # Perform cross-validation
#     scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
#     mean_accuracy = np.mean(scores)
#     print(f"{name} - Cross-validation accuracy: {mean_accuracy:.4f}")
#
#     if mean_accuracy > best_accuracy:
#         best_model = pipeline
#         best_model_name = name
#         best_accuracy = mean_accuracy
#
# # Train the best model on the full training set
# best_model.fit(X_train, y_train)
#
# # Make predictions on the test set
# y_pred = best_model.predict(X_test)
#
# # Evaluate the best model on the test set
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
#
# print(f"\nBest Model: {best_model_name}")
# print(f"Test Accuracy: {accuracy:.4f}")
# print(f"Test Precision: {precision:.4f}")
# print(f"Test Recall: {recall:.4f}")
# print(f"Test F1-score: {f1:.4f}")
