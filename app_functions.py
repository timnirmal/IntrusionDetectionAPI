import glob
import json
import os

import pandas as pd
import psutil
from loguru import logger

from Autoencoder.create_data import create_data
from Autoencoder.predict_autoencoder import predict_autoencoder, predict_rf

column_names_to_keep = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
                        "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
                        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
                        "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
                        "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
                        "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
                        "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
                        "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
                        "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
                        "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
                        "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
                        "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
                        "Fwd Avg Bulk Rate", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
                        "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
                        "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean",
                        "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]

original_columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp', "Flow Duration",
                    "Total Fwd Packets",
                    "Total Backward Packets",
                    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
                    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
                    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
                    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
                    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
                    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
                    "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
                    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
                    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
                    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
                    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length",
                    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
                    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
                    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
                    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
                    ]


def predict_for_df(df):
    df, df_scaled = create_data(df)
    result = predict_autoencoder(df_scaled)

    outliers = [0 if i == False else 1 for i in result]

    # add to df
    df_auto = pd.DataFrame({"autoencoder": outliers})

    # get autoencoder column and add to df
    df = df.join(df_auto)

    return df


def predict_for_rf(df):
    # print(df)
    df, df_scaled = create_data(df)
    print("df_scaled", df_scaled.shape)

    try:
        # load the model
        rf_pred = predict_rf(df_scaled)
    except Exception as e:
        print(e)

    # add to df
    df_auto = pd.DataFrame({"rf": rf_pred})

    # get autoencoder column and add to df
    df = df.join(df_auto)

    return df


def read_csv_files():
    # get all csv files in flow_data folder
    csv_files = glob.glob("flow_data/*.csv")
    try:
        df = pd.DataFrame()
        for csv_file in csv_files:
            try:
                # read csv file to pandas dataframe
                df_temp = pd.read_csv(csv_file)
                # get interface name by removing flow_data/ and _flow.csv
                interface = csv_file
                # print(interface)
                # add interface name to column
                df_temp["interface"] = interface
                # concat to df
                df = pd.concat([df, df_temp])
                # if csv_file exists in backup folder
                if os.path.exists("flow_data/backup/" + csv_file.split("\\")[1]):
                    # save to backup folder with a
                    df_temp.to_csv("flow_data/backup/" + csv_file.split("\\")[1], mode="a", index=False, header=False)
                else:
                    # save to backup folder
                    df_temp.to_csv("flow_data/backup/" + csv_file.split("\\")[1], index=False)
                # clear csv_file data
                empty_df = pd.DataFrame()
                empty_df.to_csv(csv_file, index=False)
            except Exception as e:
                # print("Unable to read csv file", e)
                pass

        # if df first column is "Unnamed: 0" drop the row and add the header
        if df.columns[0] == "Unnamed: 0":
            df = df.iloc[1:]
            df.columns = original_columns

        # if df is not empty
        if not df.empty:
            # predict anomalies
            try:
                df = predict_for_df(df)
            except Exception as e:
                # print("Unable to predict anomalies", e)
                df = pd.DataFrame()
        else:
            # return empty dataframe
            return pd.DataFrame().to_json(orient="records")

        # if interface_flow.csv exists
        if os.path.exists("interface_flow.csv"):
            try:
                # read interface_flow.csv
                df_backup = pd.read_csv("interface_flow.csv")
            except Exception as e:
                df_backup = pd.DataFrame()

            # if df_backup is not empty
            if not df_backup.empty:
                # save a dataframe to csv file
                df.to_csv("interface_flow.csv", mode="a", index=False, header=False)
                # print("not empty appended", df.shape)
            else:
                # save a dataframe to csv file
                df.to_csv("interface_flow.csv", mode="a", index=False)
                # print("empty appended", df.shape)
        else:
            # save a dataframe to csv file
            df.to_csv("interface_flow.csv", index=False)
            # print("not exists", df.shape)

    except Exception as e:
        print("Unable to read csv file", e)
        df = pd.DataFrame()

    # get all csvs in flow_data/backup folder to one df
    df_backup = read_backup_data()

    # if "autoencoder" column exists drop it
    if "autoencoder" in df.columns:
        df_backup = df_backup.drop(columns=["autoencoder"])

    df_backup = df_backup.tail(1000)

    print(df_backup)

    return df_backup.to_json(orient="records")


def read_backup_data():
    try:
        csv_files_backup = glob.glob("flow_data/backup/*.csv")
        df_backup = pd.DataFrame()
        for csv_file in csv_files_backup:
            try:
                # read csv file to pandas dataframe
                df_temp = pd.read_csv(csv_file)
            except Exception as e:
                # print("Unable to read csv file", e)
                df_temp = pd.DataFrame()
            # concat to df
            df_backup = pd.concat([df_backup, df_temp])
    except Exception as e:
        print("Unable to read csv file", e)
        df_backup = pd.DataFrame()
    return df_backup


def find_anomalities():
    try:
        df = read_backup_data()

        # if df is not empty
        if not df.empty:
            # predict anomalies
            try:
                df = predict_for_df(df)
                # count number of anomalies where autoencoder = 1
                an_count = df[df["autoencoder"] == 1].shape[0]
                non_an_count = df[df["autoencoder"] == 0].shape[0]

                print("count", an_count)

                if an_count > 0:
                    logger.info({"anomalies": an_count, "non_anomalies": non_an_count})
                return {"anomalies": an_count, "non_anomalies": non_an_count}
            except Exception as e:
                # print("Unable to predict anomalies", e)
                df = pd.DataFrame()

    except Exception as e:
        # print("Unable to predict anomalies", e)
        # print("Unable to read csv file")
        print(e)
        return {"anomalies": 0, "non_anomalies": 0}


def get_anomalies():
    try:
        df = read_backup_data()

        # if df is not empty
        if not df.empty:
            # predict anomalies
            try:
                df = predict_for_rf(df)
                non_BENIGN = df[df["rf"] != "BENIGN"]
                BENIGN = df[df["rf"] == "BENIGN"]

                # if 0 < non_BEIGN < 1000 add to BENIGN.tail(500)
                if non_BENIGN.shape[0] > 0:
                    BENIGN = pd.concat([BENIGN.tail(1000), non_BENIGN.tail(1000)])
                else:
                    BENIGN = pd.concat([BENIGN, non_BENIGN.tail(1000)])

                BENIGN = BENIGN.tail(1000)
                return BENIGN.to_json(orient="records")
            except Exception as e:
                print("Unable to predict anomalies", e)
                df = pd.DataFrame()
                return df.to_json(orient="records")

    except Exception as e:
        # print("Unable to predict anomalies", e)
        print("Unable to read csv file")
        print(e)
        df = pd.DataFrame()
        return df.to_json(orient="records")


def get_anomaly_count():
    try:
        data = get_anomalies()
        data_json = json.loads(data)
        df = pd.DataFrame(data_json)
        # count number of anomalies where autoencoder = 1
        an_count = df[df["rf"] != "BENIGN"].shape[0]
        non_an_count = df[df["rf"] == "BENIGN"].shape[0]
        print({"anomalies": an_count, "non_anomalies": non_an_count})
        return {"anomalies": an_count, "non_anomalies": non_an_count}
    except Exception as e:
        # print("Unable to predict anomalies", e)
        # print("Unable to read csv file")
        print(e)
        print({"anomalies": 0, "non_anomalies": 0})
        return {"anomalies": 0, "non_anomalies": 0}


def push_anomalies():
    print("pushing anomalies")
    try:
        try:
            data = get_anomalies()
            data_json = json.loads(data)
            df = pd.DataFrame(data_json)
            # count number of anomalies where autoencoder = 1
            anom = df[df["rf"] != "BENIGN"]
            print(anom)
        except Exception as e:
            print("Unable to predict anomalies", e)
            anom = pd.DataFrame()

        if not anom.empty:
            # read data.txt
            with open("data.txt", "r") as f:
                data = f.read()
                # data to int
                data_index = int(data)

            # get the data_index row from anom
            anom = anom.iloc[data_index]

            # write data_index + 1 to data.txt
            with open("data.txt", "w") as f:
                f.write(str(data_index + 1))

            return anom.to_json(orient="records")

    except Exception as e:
        print("Unable to read csv file")


def get_interfaces():
    addrs = psutil.net_if_addrs()
    return list(addrs.keys())
