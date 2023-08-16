import glob
import json
import os

import pandas as pd
import psutil
from loguru import logger
from pandas.errors import EmptyDataError

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
    # print("predicting for df 0000000000000000000000000000000000000000")
    # print(df)
    df, df_scaled = create_data(df)
    # print("df_scaled", df_scaled.shape)
    result = predict_autoencoder(df_scaled)

    outliers = [0 if i == False else 1 for i in result]

    # add to df
    df_auto = pd.DataFrame({"autoencoder": outliers})

    # get autoencoder column and add to df
    df = df.join(df_auto)

    # join df_auto to df
    df_scaled = df_scaled.join(df_auto)

    return df


def predict_for_rf(df):
    # print(df)
    df, df_scaled = create_data(df)
    print("df_scaled", df_scaled.shape)

    #     df, df_scaled = create_data(df)
    # print("df_scaled", df_scaled.shape)
    # result = predict_autoencoder(df_scaled)
    #
    # outliers = [0 if i == False else 1 for i in result]
    #
    # # add to df
    # df_auto = pd.DataFrame({"autoencoder": outliers})
    #
    # # get autoencoder column and add to df
    # df = df.join(df_auto)

    # print("Data Scaled")
    # print("Data Scaled")
    # print("Data Scaled")
    # print("Data Scaled")
    # print("Data Scaled")
    # print("Data Scaled")
    # print(df)

    try:
        # load the model
        rf_pred = predict_rf(df_scaled)
    except Exception as e:
        print(e)

    # print("rf_pred")
    # print("rf_pred")
    # print("rf_pred")
    # print("rf_pred")
    # print("rf_pred")
    # print("rf_pred")
    # print("rf_pred")
    # print(rf_pred)

    # add to df
    df_auto = pd.DataFrame({"rf": rf_pred})

    # get autoencoder column and add to df
    df = df.join(df_auto)

    # print(df)

    return df


def predict_for_df_anom(df):
    # print(df)
    df, df_scaled = create_data(df)

    # print("df_scaled", df_scaled.shape)
    result = predict_autoencoder(df_scaled)

    outliers = [0 if i == False else 1 for i in result]

    # add to df
    df_auto = pd.DataFrame({"autoencoder": outliers})

    # get autoencoder column and add to df
    df = df.join(df_auto)

    # keep only autoencoder == 1
    anom = df[df["autoencoder"] == 1]

    return anom


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

        # print(df)

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
        # raise Exception("Unable to read csv file") from e
        df = pd.DataFrame()
        # print("Unable to read csv file", e)

    # if interface_flow.csv exists read it
    if os.path.exists("flow_data/backup/WiFi_flow.csv"):
        try:
            df = pd.read_csv("flow_data/backup/WiFi_flow.csv")
        except Exception as e:
            df = pd.DataFrame()

    # if "autoencoder" column exists drop it
    if "autoencoder" in df.columns:
        df = df.drop(columns=["autoencoder"])

    # print df shape
    # print("df shape_rcf", df.shape)

    df = df.tail(1000)

    return df.to_json(orient="records")


def find_anomalities():
    # print("##################################################################################")
    try:
        # if interface_flow.csv exists read it
        if os.path.exists("flow_data/backup/WiFi_flow.csv"):
            try:
                # read interface_flow and interface_flow_backup get interface_flow + interface_flow_backup[:x] = 2000
                df = pd.read_csv("flow_data/backup/WiFi_flow.csv")
            except Exception as e:
                df = pd.DataFrame()
                print(e)

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
            else:
                # return empty dataframe
                return {"anomalies": 0, "non_anomalies": 0}
    except Exception as e:
        # print("Unable to predict anomalies", e)
        # print("Unable to read csv file")
        print(e)
        return {"anomalies": 0, "non_anomalies": 0}


def get_anomalies():
    print("##################################################################################")
    try:
        # if interface_flow.csv exists read it
        if os.path.exists("flow_data/backup/WiFi_flow.csv"):
            try:
                # read interface_flow and interface_flow_backup get interface_flow + interface_flow_backup[:x] = 2000
                df = pd.read_csv("flow_data/backup/WiFi_flow.csv")
            except Exception as e:
                df = pd.DataFrame()
                print(e)

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
            # else:
            #     # return dataframe
            #     return df.to_json(orient="records")
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
        # if csv file exists
        if os.path.exists("flow_data/backup/WiFi_flow.csv"):
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
                columns_to_keep = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                                   "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                                   "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
                                   "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
                                   "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
                                   "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
                                   "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
                                   "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
                                   "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
                                   "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
                                   "Max Packet Length", "Packet Length Mean", "Packet Length Std",
                                   "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
                                   "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
                                   "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
                                   "Avg Bwd Segment Size", "Fwd Avg Bulk Rate", "Bwd Avg Bulk Rate",
                                   "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
                                   "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
                                   "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
                                   "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
                                   "autoencoder"]
                # print(len(columns_to_keep))
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
    addrs = psutil.net_if_addrs()
    # return addrs.keys() as json
    return list(addrs.keys())
