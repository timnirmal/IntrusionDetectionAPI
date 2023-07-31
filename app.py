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
import os
from datetime import datetime

import pandas as pd
import psutil
from fastapi import FastAPI, WebSocket
from random import choice, randint
import asyncio
import glob
from loguru import logger

from pandas.errors import EmptyDataError

from Autoencoder.one_predict.create_data import create_data
from Autoencoder.one_predict.predict_autoencoder import predict_autoencoder

# delete flow_anomalies.csv if it exists
if os.path.exists("flow_anomalies.csv"):
    os.remove("flow_anomalies.csv")

# delete flow_queue.csv if it exists
if os.path.exists("flow_queue.csv"):
    os.remove("flow_queue.csv")

# delete flow_queue.csv if it exists
if os.path.exists("flow_send.csv"):
    os.remove("flow_send.csv")

# # clear flows_queue.csv
# with open("flows_queue.csv", "w") as f:
#     pass

app = FastAPI()

CHANNELS = ["A", "B", "C"]

logger.add("logs.log")


def generate_data():
    return {
        "channel": choice(CHANNELS),
        "data": randint(1, 10)
    }


def read_csv_file(interface):
    data_path = "flow_data/" + interface + "_flow.csv"
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


def read_csv_files():
    # get all csv files in flow_data folder
    csv_files = glob.glob("flow_data/*.csv")
    csv_file_format = "flow_data/interface_flow.csv"
    try:
        df = pd.DataFrame()
        for csv_file in csv_files:
            try:
                # read csv file to pandas dataframe
                df_temp = pd.read_csv(csv_file)
                # get interface name by removing flow_data/ and _flow.csv
                interface = csv_file
                print(interface)
                # add interface name to column
                df_temp["interface"] = interface
                # concat to df
                df = pd.concat([df, df_temp])
            except Exception as e:
                # print("Unable to read csv file", e)
                pass
    except Exception as e:
        # raise Exception("Unable to read csv file") from e
        df = pd.DataFrame()
        print("Unable to read csv file", e)

    return df.to_json(orient="records")


def find_anomalities():
    csv_files = glob.glob("flow_data/*.csv")
    try:
        df = pd.DataFrame()
        for csv_file in csv_files:
            try:
                # read csv file to pandas dataframe
                df_temp = pd.read_csv(csv_file)
                interface = csv_file
                print(interface)
                # add interface name to column
                df_temp["interface"] = interface
                # concat to df
                df = pd.concat([df, df_temp])
            except Exception as e:
                # print("Unable to read csv file", e)
                pass
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
        an_count = df[df["autoencoder"] == 1].shape[0]
        non_an_count = df[df["autoencoder"] == 0].shape[0]

        anomalies = df[df["autoencoder"] == 1]
        # save to csv
        anomalies.to_csv("flow_queue.csv", index=False)

        print("count", an_count)

    except Exception as e:
        # raise Exception("Unable to read csv file") from e
        count = 0
        print("Unable to read csv file")

    if an_count > 0:
        logger.info({"anomalies": an_count, "non_anomalies": non_an_count})
    return {"anomalies": an_count, "non_anomalies": non_an_count}


def push_anomalies():
    print("pushing anomalies")
    try:
        # if flow_anomalies.csv exists
        if os.path.exists("flow_queue.csv"):
            try:
                df_queue = pd.read_csv("flow_queue.csv")
            except EmptyDataError:
                df_queue = pd.DataFrame()
        else:
            df_queue = pd.DataFrame()

        if os.path.exists("flow_send.csv"):
            try:
                df_send = pd.read_csv("flow_send.csv")
                # df_send = df_send.iloc[0]
                # df_send = df_send.drop(df_send.index[0])
            except EmptyDataError:
                df_send = pd.DataFrame()
        else:
            df_send = pd.DataFrame()

        # if df_send is not empty then compare with df_anom and get the difference
        if not df_queue.empty:
            logger.info(str(df_send.shape) + str(df_queue.shape))
            # compare the shapes and get the first difference
            queue_rows = df_queue.shape[0]
            if not df_send.empty:
                send_rows = df_send.shape[0]
            else:
                send_rows = 0
            logger.info("send_rows")
            logger.info(send_rows)
            logger.info("queue_rows")
            logger.info(queue_rows)
            if send_rows < queue_rows:
                # df_diff_index = queue_rows - send_rows
                df_diff_index = send_rows + 1
                logger.info("df_diff_index")
                logger.info(df_diff_index)
                # get the df_diff_index row from df_queue
                df_diff = df_queue[:df_diff_index]
                logger.info(df_diff)
                df_diff = df_diff.tail(1)
                # df_diff.to_csv("flow_send_1.csv")
                columns_to_keep = ["Flow Duration","Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","Fwd Header Length","Bwd Header Length","Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance","FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count","CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size","Fwd Avg Bulk Rate","Bwd Avg Bulk Rate","Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min","autoencoder"]
                print(len(columns_to_keep))
                # keep only the columns in columns_to_keep
                df_diff = df_diff[columns_to_keep]
                df_diff.to_csv("flow_send_2.csv", index=False)
                # load flow_send_2.csv ignore index
                df_diff = pd.read_csv("flow_send_2.csv", index_col=0)
                df_diff.to_csv("flow_send_3.csv")
                logger.info("df_diff")
                logger.info(df_diff)
                # save to flows_queue.csv
                df_diff_conc = pd.concat([df_send, df_diff])
                df_diff_conc.to_csv("flow_send.csv", index=False)

        # elif df_send.empty and not df_queue.empty:
        #     logger.info("df_send is empty")
        #     # keep only the first row
        #     df_diff = df_queue.head(1)
        #     print("df_diff", df_diff)
        #     logger.info("df_diff", type(df_diff))
        #     logger.info("df_diff", df_diff.shape)
        #     logger.info("df_diff", df_diff)
        #     if not df_diff.empty:
        #         logger.info("df_diff", df_diff)
        #         # save to flows_queue.csv
        #         df_diff.to_csv("flow_send.csv")
        else:
            df_diff = pd.DataFrame()

    except Exception as e:
        df_diff = pd.DataFrame()
        print("Unable to read csv file")

    return df_diff.to_json(orient="records")


def get_interfaces():
    import netifaces
    addrs = psutil.net_if_addrs()
    # return addrs.keys() as json
    return list(addrs.keys())


@app.websocket("/retrive_data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # get interface from url
        # interface = websocket.query_params.get("interface")
        # print("interface is", interface)
        await websocket.send_json(
            read_csv_files()
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


@app.websocket("/anomaly_push")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("pushing anomalies")
    while True:
        await websocket.send_json(
            push_anomalies()
        )
        await asyncio.sleep(2)

print("starting server")
