import numpy as np
import pandas as pd

from Autoencoder.one_predict.create_data import create_data
from Autoencoder.one_predict.predict_autoencoder import predict_autoencoder

if __name__ == '__main__':
    df = pd.read_csv(r"D:\Projects\IntrusionDetectionAPI\flow.csv")
    # csv_line = "192.168.11.164,172.253.118.188,13001,5228,6,2023-07-25 14:11:38,39720.05844116211,3046.319787752554,50.35239318599263,25.176196592996316,25.176196592996316,1,1,55,66,55.0,55.0,55.0,0.0,66.0,66.0,66.0,0.0,66,55,60.5,5.5,30.25,20,20,20,1,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,0,1,0,0,0,0,0,0,1.0,60.5,256,265,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,55.0,66.0,0,1,1,55,66"
    df = create_data(df)
    result = predict_autoencoder(df)
    outliers = [0 if i == False else 1 for i in result]

    # add to df
    df_auto = pd.DataFrame({"autoencoder": outliers})
    # join df_auto to df
    df = df.join(df_auto)

    print(df)

    # print(outliers)
