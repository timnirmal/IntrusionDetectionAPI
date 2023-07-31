import pandas as pd
from pandas.errors import EmptyDataError

from Autoencoder.one_predict.create_data import create_data
# from Autoencoder.one_predict.predict_autoencoder import predict_autoencoder
#
# df = pd.read_csv("flow.csv")
#
# df = create_data(df)
# result = predict_autoencoder(df)
# outliers = [0 if i == False else 1 for i in result]
#
# # add to df
# df_auto = pd.DataFrame({"autoencoder": outliers})
# # join df_auto to df
# df = df.join(df_auto)
#
# print(df)

try:
    # df_queue = pd.read_csv("flows_queue.csv")
    df_queue = pd.read_csv("flow_anomalies_2.csv")
except EmptyDataError:
    df_queue = pd.DataFrame()
    print("EmptyDataError")

# get the first item in the queue and remove it
df_queue = df_queue.iloc[0]
df_queue = df_queue.drop(df_queue.index[0])
# save to csv
df_queue.to_csv("flow_anomalies_2.csv", index=False)

print(df_queue)