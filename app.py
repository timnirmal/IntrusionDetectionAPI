import asyncio
import os

from fastapi import FastAPI, WebSocket
from loguru import logger

from app_functions import read_csv_files, find_anomalities, get_anomalies, push_anomalies, get_interfaces, \
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


# [
#     "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
#     "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
#     "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
#     "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
#     "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
#     "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
#     "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
#     "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
#     "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
#     "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
#     "Fwd Avg Bulk Rate", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
#     "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
#     "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
#     "Idle Max", "Idle Min"
# ]


# [
#     src_ip,
#     dst_ip,
#     src_port,
#     dst_port,
#     protocol,
#     timestamp,
#     flow_duration,
#     flow_byts_s,
#     flow_pkts_s,
#     fwd_pkts_s,
#     bwd_pkts_s,
#     tot_fwd_pkts,
#     tot_bwd_pkts,
#     totlen_fwd_pkts,
#     totlen_bwd_pkts,
#     fwd_pkt_len_max,fwd_pkt_len_min,fwd_pkt_len_mean,fwd_pkt_len_std,bwd_pkt_len_max,bwd_pkt_len_min,bwd_pkt_len_mean,bwd_pkt_len_std,pkt_len_max,pkt_len_min,pkt_len_mean,pkt_len_std,pkt_len_var,fwd_header_len,bwd_header_len,fwd_seg_size_min,fwd_act_data_pkts,flow_iat_mean,flow_iat_max,flow_iat_min,flow_iat_std,fwd_iat_tot,fwd_iat_max,fwd_iat_min,fwd_iat_mean,fwd_iat_std,bwd_iat_tot,bwd_iat_max,bwd_iat_min,bwd_iat_mean,bwd_iat_std,fwd_psh_flags,bwd_psh_flags,fwd_urg_flags,bwd_urg_flags,fin_flag_cnt,syn_flag_cnt,rst_flag_cnt,psh_flag_cnt,ack_flag_cnt,urg_flag_cnt,ece_flag_cnt,down_up_ratio,pkt_size_avg,init_fwd_win_byts,init_bwd_win_byts,active_max,active_min,active_mean,active_std,idle_max,idle_min,idle_mean,idle_std,fwd_byts_b_avg,fwd_pkts_b_avg,bwd_byts_b_avg,bwd_pkts_b_avg,fwd_blk_rate_avg,bwd_blk_rate_avg,fwd_seg_size_avg,bwd_seg_size_avg,cwe_flag_count,subflow_fwd_pkts,subflow_bwd_pkts,subflow_fwd_byts,subflow_bwd_byts,interface
# ]


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
