import pandas as pd

from Autoencoder.one_predict.create_data import create_data
from Autoencoder.one_predict.predict_autoencoder import predict_autoencoder

df = pd.read_csv("flow.csv")

df = create_data(df)
result = predict_autoencoder(df)
outliers = [0 if i == False else 1 for i in result]

# add to df
df_auto = pd.DataFrame({"autoencoder": outliers})
# join df_auto to df
df = df.join(df_auto)

print(df)

