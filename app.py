import asyncio
import os

from fastapi import FastAPI, WebSocket
from loguru import logger

from app_functions import read_csv_files, get_anomalies, push_anomalies, get_interfaces, \
    get_anomaly_count

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

logger.add("logs.log")


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
            get_anomaly_count()
        )
        await asyncio.sleep(0.5)


@app.websocket("/view-anomalies")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        await websocket.send_json(
            get_anomalies()
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

# uvicorn main:app --reload
