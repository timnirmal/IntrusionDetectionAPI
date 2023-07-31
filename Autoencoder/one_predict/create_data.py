import pickle

import pandas as pd


# def create_data(csv_line):
def create_data(df):
    # column_line = "src_ip,dst_ip,src_port,dst_port,protocol,timestamp,flow_duration,flow_byts_s,flow_pkts_s,fwd_pkts_s,bwd_pkts_s,tot_fwd_pkts,tot_bwd_pkts,totlen_fwd_pkts,totlen_bwd_pkts,fwd_pkt_len_max,fwd_pkt_len_min,fwd_pkt_len_mean,fwd_pkt_len_std,bwd_pkt_len_max,bwd_pkt_len_min,bwd_pkt_len_mean,bwd_pkt_len_std,pkt_len_max,pkt_len_min,pkt_len_mean,pkt_len_std,pkt_len_var,fwd_header_len,bwd_header_len,fwd_seg_size_min,fwd_act_data_pkts,flow_iat_mean,flow_iat_max,flow_iat_min,flow_iat_std,fwd_iat_tot,fwd_iat_max,fwd_iat_min,fwd_iat_mean,fwd_iat_std,bwd_iat_tot,bwd_iat_max,bwd_iat_min,bwd_iat_mean,bwd_iat_std,fwd_psh_flags,bwd_psh_flags,fwd_urg_flags,bwd_urg_flags,fin_flag_cnt,syn_flag_cnt,rst_flag_cnt,psh_flag_cnt,ack_flag_cnt,urg_flag_cnt,ece_flag_cnt,down_up_ratio,pkt_size_avg,init_fwd_win_byts,init_bwd_win_byts,active_max,active_min,active_mean,active_std,idle_max,idle_min,idle_mean,idle_std,fwd_byts_b_avg,fwd_pkts_b_avg,bwd_byts_b_avg,bwd_pkts_b_avg,fwd_blk_rate_avg,bwd_blk_rate_avg,fwd_seg_size_avg,bwd_seg_size_avg,cwe_flag_count,subflow_fwd_pkts,subflow_bwd_pkts,subflow_fwd_byts,subflow_bwd_byts"
    #
    # # create a dataframe with the column names and the csv line as the first row
    # df = pd.DataFrame([csv_line.split(",")], columns=column_line.split(","))

    # Flow Duration - flow_duration
    # Total Fwd Packets - tot_fwd_pkts
    # Total Backward Packets - tot_bwd_pkts
    # Total Length of Fwd Packets - totlen_fwd_pkts
    # Total Length of Bwd Packets - totlen_bwd_pkts
    # Fwd Packet Length Max - fwd_pkt_len_max
    # Fwd Packet Length Min - fwd_pkt_len_min
    # Fwd Packet Length Mean - fwd_pkt_len_mean
    # Fwd Packet Length Std - fwd_pkt_len_std
    # Bwd Packet Length Max - bwd_pkt_len_max
    # Bwd Packet Length Min - bwd_pkt_len_min
    # Bwd Packet Length Mean - bwd_pkt_len_mean
    # Bwd Packet Length Std - bwd_pkt_len_std
    # Flow Bytes/s - flow_byts_s
    # Flow Packets/s - flow_pkts_s
    # Flow IAT Mean - flow_iat_mean
    # Flow IAT Std - flow_iat_std
    # Flow IAT Max - flow_iat_max
    # Flow IAT Min - flow_iat_min
    # Fwd IAT Total - fwd_iat_tot
    # Fwd IAT Mean - fwd_iat_mean
    # Fwd IAT Std - fwd_iat_std
    # Fwd IAT Max - fwd_iat_max
    # Fwd IAT Min - fwd_iat_min
    # Bwd IAT Total - bwd_iat_tot
    # Bwd IAT Mean - bwd_iat_mean
    # Bwd IAT Std - bwd_iat_std
    # Bwd IAT Max - bwd_iat_max
    # Bwd IAT Min - bwd_iat_min
    # Fwd PSH Flags - fwd_psh_flags
    # Bwd PSH Flags - bwd_psh_flags
    # Fwd URG Flags - fwd_urg_flags
    # Bwd URG Flags - bwd_urg_flags
    # Fwd Header Length - fwd_header_len
    # Bwd Header Length - bwd_header_len
    # Fwd Packets/s - fwd_pkts_s
    # Bwd Packets/s - bwd_pkts_s
    # Min Packet Length - pkt_len_min
    # Max Packet Length - pkt_len_max
    # Packet Length Mean - pkt_len_mean
    # Packet Length Std - pkt_len_std
    # Packet Length Variance - pkt_len_var
    # FIN Flag Count - fin_flag_cnt
    # SYN Flag Count - syn_flag_cnt
    # RST Flag Count - rst_flag_cnt
    # PSH Flag Count - psh_flag_cnt
    # ACK Flag Count - ack_flag_cnt
    # URG Flag Count - urg_flag_cnt
    # CWE Flag Count - cwe_flag_count
    # ECE Flag Count - ece_flag_cnt
    # Down/Up Ratio - down_up_ratio
    # Average Packet Size - pkt_size_avg
    # Avg Fwd Segment Size - fwd_seg_size_avg
    # Avg Bwd Segment Size - bwd_seg_size_avg
    # Fwd Avg Bulk Rate - fwd_blk_rate_avg
    # Bwd Avg Bulk Rate - bwd_blk_rate_avg
    # Subflow Fwd Packets - subflow_fwd_pkts
    # Subflow Fwd Bytes - subflow_fwd_byts
    # Subflow Bwd Packets - subflow_bwd_pkts
    # Subflow Bwd Bytes - subflow_bwd_byts
    # Init_Win_bytes_forward - init_fwd_win_byts
    # Init_Win_bytes_backward - init_bwd_win_byts
    # act_data_pkt_fwd - fwd_act_data_pkts
    # min_seg_size_forward - fwd_seg_size_min
    # Active Mean - active_mean
    # Active Std - active_std
    # Active Max - active_max
    # Active Min - active_min
    # Idle Mean - idle_mean
    # Idle Std - idle_std
    # Idle Max - idle_max
    # Idle Min - idle_min

    # rename the rights as left in df
    df = df.rename(columns={
        "flow_duration": "Flow Duration", "tot_fwd_pkts": "Total Fwd Packets", "tot_bwd_pkts": "Total Backward Packets",
        "totlen_fwd_pkts": "Total Length of Fwd Packets", "totlen_bwd_pkts": "Total Length of Bwd Packets",
        "fwd_pkt_len_max": "Fwd Packet Length Max", "fwd_pkt_len_min": "Fwd Packet Length Min",
        "fwd_pkt_len_mean": "Fwd Packet Length Mean", "fwd_pkt_len_std": "Fwd Packet Length Std",
        "bwd_pkt_len_max": "Bwd Packet Length Max", "bwd_pkt_len_min": "Bwd Packet Length Min",
        "bwd_pkt_len_mean": "Bwd Packet Length Mean", "bwd_pkt_len_std": "Bwd Packet Length Std",
        "flow_byts_s": "Flow Bytes/s", "flow_pkts_s": "Flow Packets/s", "flow_iat_mean": "Flow IAT Mean",
        "flow_iat_std": "Flow IAT Std", "flow_iat_max": "Flow IAT Max", "flow_iat_min": "Flow IAT Min",
        "fwd_iat_tot": "Fwd IAT Total", "fwd_iat_mean": "Fwd IAT Mean", "fwd_iat_std": "Fwd IAT Std",
        "fwd_iat_max": "Fwd IAT Max", "fwd_iat_min": "Fwd IAT Min", "bwd_iat_tot": "Bwd IAT Total",
        "bwd_iat_mean": "Bwd IAT Mean", "bwd_iat_std": "Bwd IAT Std", "bwd_iat_max": "Bwd IAT Max",
        "bwd_iat_min": "Bwd IAT Min", "fwd_psh_flags": "Fwd PSH Flags", "bwd_psh_flags": "Bwd PSH Flags",
        "fwd_urg_flags": "Fwd URG Flags", "bwd_urg_flags": "Bwd URG Flags", "fwd_header_len": "Fwd Header Length",
        "bwd_header_len": "Bwd Header Length", "fwd_pkts_s": "Fwd Packets/s", "bwd_pkts_s": "Bwd Packets/s",
        "pkt_len_min": "Min Packet Length", "pkt_len_max": "Max Packet Length", "pkt_len_mean": "Packet Length Mean",
        "pkt_len_std": "Packet Length Std", "pkt_len_var": "Packet Length Variance", "fin_flag_cnt": "FIN Flag Count",
        "syn_flag_cnt": "SYN Flag Count", "rst_flag_cnt": "RST Flag Count", "psh_flag_cnt": "PSH Flag Count",
        "ack_flag_cnt": "ACK Flag Count", "urg_flag_cnt": "URG Flag Count", "cwe_flag_count": "CWE Flag Count",
        "ece_flag_cnt": "ECE Flag Count", "down_up_ratio": "Down/Up Ratio", "pkt_size_avg": "Average Packet Size",
        "fwd_seg_size_avg": "Avg Fwd Segment Size", "bwd_seg_size_avg": "Avg Bwd Segment Size",
        "fwd_blk_rate_avg": "Fwd Avg Bulk Rate", "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
        "subflow_fwd_pkts": "Subflow Fwd Packets", "subflow_fwd_byts": "Subflow Fwd Bytes",
        "subflow_bwd_pkts": "Subflow Bwd Packets", "subflow_bwd_byts": "Subflow Bwd Bytes",
        "init_fwd_win_byts": "Init_Win_bytes_forward", "init_bwd_win_byts": "Init_Win_bytes_backward",
        "fwd_act_data_pkts": "act_data_pkt_fwd", "fwd_seg_size_min": "min_seg_size_forward",
        "active_mean": "Active Mean",
        "active_std": "Active Std", "active_max": "Active Max", "active_min": "Active Min", "idle_mean": "Idle Mean",
        "idle_std": "Idle Std", "idle_max": "Idle Max", "idle_min": "Idle Min"
    })

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

    # count column names
    print(len(column_names_to_keep))

    # load scaler
    # scaler = pickle.load(open('../train_autoencoder/scaler.pkl', 'rb'))
    scaler = pickle.load(open(r'D:\Projects\IntrusionDetectionAPI\Autoencoder\train_autoencoder\scaler.pkl', 'rb'))

    # scale the data
    df = scaler.transform(df[column_names_to_keep])

    # convert back to dataframe
    df = pd.DataFrame(df, columns=column_names_to_keep)

    return df

#
# # load the X_test '../train_autoencoder/data/X_test.npy'
# X_test = np.load('../train_autoencoder/data/X_test.npy')
# y_test = np.load('../train_autoencoder/data/y_test.npy')
# # create df with X_test and y_test
# df = pd.DataFrame(X_test)
# print(df.columns)
# df['Label'] = y_test
#
# # count label
# print(df['Label'].value_counts())
#
# # # keep only the normal data
# # df = df[df['Label'] == 1]
# # y_test = df['Label']
# df = df.drop('Label', axis=1)
#


# # print(outliers)
#
# # if outliers == False then 0 else 1
# outliers = [0 if i == False else 1 for i in outliers]
# # print(outliers)

#
# # by mse find the threshold to classify the data
# from sklearn.metrics import mean_squared_error, precision_recall_curve, f1_score
#
# # Calculate MSE values
# mse_values = mean_squared_error(y_test, prediction, multioutput='raw_values')
#
# # Define a range of threshold values (adjust the range as needed)
# threshold_range = np.linspace(min(mse_values), max(mse_values), num=100)
#
# best_threshold = None
# best_f1_score = 0.0
#
# # Evaluate classification performance for each threshold
# for threshold in threshold_range:
#     y_pred_class = mse_values > threshold
#     f1 = f1_score(y_test, y_pred_class)
#
#     if f1 > best_f1_score:
#         best_f1_score = f1
#         best_threshold = threshold
#
# print("Best Threshold:", best_threshold)
# print("Best F1-score:", best_f1_score)
#
