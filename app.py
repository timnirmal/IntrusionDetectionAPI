# import os
# from datetime import datetime, timedelta
# from typing import List
#
# import pandas as pd
# import requests
# from bson import ObjectId
# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
#
# print("The app is running")
#
# # show all columns
# pd.set_option('display.max_columns', None)
#
# app = FastAPI()
#
# origins = ["*"]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # class PyObjectId(ObjectId):
# #     @classmethod
# #     def __get_validators__(cls):
# #         yield cls.validate
# #
# #     @classmethod
# #     def validate(cls, v):
# #         if not ObjectId.is_valid(v):
# #             raise ValueError("Invalid objectid")
# #         return ObjectId(v)
# #
# #     @classmethod
# #     def __modify_schema__(cls, field_schema):
# #         field_schema.update(type="string")
#
#
# # create a get function to show string
# def get_string():
#     return "Hello World"
#
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
#
#
# # print csv files add ?amount=10 to get 10 rows
# @app.get("/csv_file")
# def read_csv_file(amount: int = 10):
#     data_path = "flow.csv"
#     try:
#         # read csv file to pandas dataframe
#         df = pd.read_csv(data_path)
#     except Exception as e:
#         # raise Exception("Unable to read csv file") from e
#         df = pd.DataFrame()
#
#     # print(df.tail())
#     # print time
#     print(datetime.now().strftime("%H:%M:%S"))
#
#     # return dataframe as json
#     return df.tail(amount).to_json(orient="records")
#
#
#
#
#
#
# # uvicorn main:app --reload
from datetime import datetime

import pandas as pd
import psutil
from fastapi import FastAPI, WebSocket
from random import choice, randint
import asyncio

from Autoencoder.one_predict.create_data import create_data
from Autoencoder.one_predict.predict_autoencoder import predict_autoencoder

# clear flows_queue.csv
with open("flows_queue.csv", "w") as f:
    pass

app = FastAPI()

CHANNELS = ["A", "B", "C"]


def generate_data():
    return {
        "channel": choice(CHANNELS),
        "data": randint(1, 10)
    }


def read_csv_file():
    data_path = "flow.csv"
    try:
        # read csv file to pandas dataframe
        df = pd.read_csv(data_path)
    except Exception as e:
        # raise Exception("Unable to read csv file") from e
        df = pd.DataFrame()
        print("Unable to read csv file")

    # try:
    #     df_queue = pd.read_csv("flows_queue.csv")
    # except Exception as e:
    #     df_queue = pd.DataFrame()
    #
    # # if dataframe is not empty
    # if not df.empty:
    #     # compare with flows_queue.csv and get the difference
    #     df_diff = df[~df.isin(df_queue)].dropna()
    #     # concat to flows_queue.csv
    #     df_queue = pd.concat([df_queue, df_diff])
    #     # save to flows_queue.csv
    #     df_queue.to_csv("flows_queue.csv")
    #
    #     return df_diff.to_json(orient="records")
    #
    # else:
    #     # return empty dataframe
    #     return pd.DataFrame().to_json(orient="records")
    # print("sending..", df.tail(10))

    return df.to_json(orient="records")


def find_anomalities():
    data_path = "flow.csv"
    try:
        # read csv file to pandas dataframe
        df = pd.read_csv(data_path)
        df = create_data(df)
        result = predict_autoencoder(df)
        outliers = [0 if i == False else 1 for i in result]

        # add to df
        df_auto = pd.DataFrame({"autoencoder": outliers})
        # join df_auto to df
        df = df.join(df_auto)

        # save to csv
        df.to_csv("flow_anomalies.csv", index=False)

        # count number of anomalies where autoencoder = 1
        count = df[df["autoencoder"] == 1].shape[0]
        print("count", count)

    except Exception as e:
        # raise Exception("Unable to read csv file") from e
        count = 0
        print("Unable to read csv file")

    return count


def get_interfaces():
    import netifaces
    addrs = psutil.net_if_addrs()
    # return addrs.keys() as json
    return list(addrs.keys())


@app.websocket("/sample")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print(data)

        await websocket.send_json(
            read_csv_file()
        )
        await asyncio.sleep(0.5)


@app.websocket("/anomalities")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json(
            find_anomalities()
        )
        await asyncio.sleep(0.5)


@app.websocket("/interfaces")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json(
            get_interfaces()
        )
        await asyncio.sleep(0.5)
