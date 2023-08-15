# # from app import push_anomalies, find_anomalities
# #
# # find_anomalities()
# #
# # for i in range(10):
# #     push_anomalies()
# import glob
# import os
#
# import pandas as pd
#
# from app import predict_for_df
#
# # get all csv files in flow_data folder
# csv_files = glob.glob("flow_data/*.csv")
# csv_files_backup = glob.glob("flow_data/backup/*.csv")
# csv_file_format = "flow_data/interface_flow.csv"
# try:
#     df = pd.DataFrame()
#     for csv_file in csv_files:
#         try:
#             # read csv file to pandas dataframe
#             df_temp = pd.read_csv(csv_file)
#             # get interface name by removing flow_data/ and _flow.csv
#             interface = csv_file
#             print(interface)
#             # add interface name to column
#             df_temp["interface"] = interface
#             # concat to df
#             df = pd.concat([df, df_temp])
#         except Exception as e:
#             # print("Unable to read csv file", e)
#             pass
#
#     print("Before: ", df.shape)
#     # predict anomalies
#     df = predict_for_df(df)
#     print("after: ", df.shape)
#
#     # if interface_flow.csv exists
#     if os.path.exists("interface_flow.csv"):
#         try:
#             # read interface_flow.csv
#             df_backup = pd.read_csv("interface_flow.csv")
#         except Exception as e:
#             df_backup = pd.DataFrame()
#
#         # if df_backup is not empty
#         if not df_backup.empty:
#             # save a dataframe to csv file
#             df.to_csv("interface_flow.csv", mode="a", index=False, header=False)
#         else:
#             # save a dataframe to csv file
#             df.to_csv("interface_flow.csv", mode="a", index=False)
#     else:
#         # save a dataframe to csv file
#         df.to_csv("interface_flow.csv", mode="a", index=False)
#
# except Exception as e:
#     # raise Exception("Unable to read csv file") from e
#     df = pd.DataFrame()
#     print("Unable to read csv file", e)
#
# print(df.to_json(orient="records"))
import pandas as pd

from app import get_interfaces, find_anomalities, read_csv_files

# find_anomalities()
read_csv_files()


# df = pd.read_csv("flow_data/WiFi_flow.csv")
#
# print(df.shape)
# print(df.columns)
# print(df)


# a = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp', 'Flow Duration', 'Flow Bytes/s',
#      'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s', 'Total Fwd Packets', 'Total Backward Packets',
#      'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
#      'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
#      'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Max Packet Length', 'Min Packet Length', 'Packet Length Mean',
#      'Packet Length Std', 'Packet Length Variance', 'Fwd Header Length', 'Bwd Header Length', 'min_seg_size_forward',
#      'act_data_pkt_fwd', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min', 'Flow IAT Std', 'Fwd IAT Total',
#      'Fwd IAT Max', 'Fwd IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Bwd IAT Total', 'Bwd IAT Max', 'Bwd IAT Min',
#      'Bwd IAT Mean', 'Bwd IAT Std', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
#      'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
#      'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
#      'Active Max', 'Active Min', 'Active Mean', 'Active Std', 'Idle Max', 'Idle Min', 'Idle Mean', 'Idle Std',
#      'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'Fwd Avg Bulk Rate', 'Bwd Avg Bulk Rate',
#      'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'CWE Flag Count', 'Subflow Fwd Packets', 'Subflow Bwd Packets',
#      'Subflow Fwd Bytes', 'Subflow Bwd Bytes', 'interface'
#      ]
#
# b = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp', "Flow Duration", "Total Fwd Packets",
#      "Total Backward Packets",
#      "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
#      "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
#      "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
#      "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
#      "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
#      "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
#      "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
#      "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
#      "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
#      "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
#      "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length",
#      "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
#      "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
#      "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
#      "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
#      "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
#      ]
#
# # comapre a with b
# print(set(a) - set(b))
# print(set(b) - set(a))
#
# print(len(a))
# print(len(b))