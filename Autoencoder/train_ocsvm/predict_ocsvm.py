import pickle


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
