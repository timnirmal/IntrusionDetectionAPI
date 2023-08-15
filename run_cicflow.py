import os
import threading
import time

import pandas as pd
import psutil

from cicflowmeter.sniffer import create_sniffer


def read_flow_csv(file_path):
    while True:
        if not os.path.exists(file_path):
            time.sleep(1)
            continue

        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                print(df)  # Replace this with whatever you want to do with the data
        except pd.errors.EmptyDataError:
            print(f"{file_path} is empty. \t time = ", time.strftime("%H:%M:%S", time.localtime()))

        time.sleep(5)  # Adjust the interval for reading the file here


def get_interfaces():
    addrs = psutil.net_if_addrs()
    return addrs.keys()


def run_sniffer(interface, output_file):
    sniffer = create_sniffer(
        input_file=None,
        input_interface=interface,
        output_mode="flow",
        output_file=output_file,
        url_model=None
    )
    sniffer.start()
    try:
        sniffer.join()
    except KeyboardInterrupt:
        sniffer.stop()


if __name__ == '__main__':
    interfaces = get_interfaces()
    # # keep only WiFi and Ethernet interfaces
    # interfaces = [interface for interface in interfaces if interface in ['Ethernet', 'WiFi']]
    # print("Interfaces: ", interfaces)
    # (['Ethernet', 'Local Area Connection* 1', 'Local Area Connection* 4', 'WiFi', 'Bluetooth Network Connection 2', 'Loopback Pseudo-Interface 1'])

    # show list of interfaces
    print("Interfaces: ", get_interfaces())

    # print start time
    print("Start time: ", time.strftime("%H:%M:%S", time.localtime()))

    # Delete the flow.csv files if they exist
    for interface in interfaces:
        try:
            file_path = f"flow_data/{interface}_flow.csv"
            file_path = file_path.replace("*", "")
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            print(f"Unable to delete {interface}_flow.csv")

    # Delete the flow.csv files if they exist
    for interface in interfaces:
        try:
            file_path = f"flow_data/backup/{interface}_flow.csv"
            file_path = file_path.replace("*", "")
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            print(f"Unable to delete {interface}_flow.csv")

    # delete flow_anomalies.csv if it exists
    if os.path.exists("interface_flow.csv"):
        os.remove("interface_flow.csv")

    # Create and start sniffers for each interface
    sniffer_threads = []
    for interface in interfaces:
        try:
            file_path = f"flow_data/{interface}_flow.csv"
            file_path = file_path.replace("*", "")
            sniffer_thread = threading.Thread(target=run_sniffer, args=(interface, file_path))
            sniffer_thread.start()
            sniffer_threads.append(sniffer_thread)
        except OSError:
            print(f"Unable to create sniffer for {interface}")

    try:
        # Start the reader thread for each flow.csv file
        reader_threads = []
        for interface in interfaces:
            file_path = f"{interface}_flow.csv"
            file_path = file_path.replace("*", "")
            reader_thread = threading.Thread(target=read_flow_csv, args=(file_path,))
            reader_thread.start()
            reader_threads.append(reader_thread)

        # Wait for all reader threads to finish
        for reader_thread in reader_threads:
            reader_thread.join()

    except KeyboardInterrupt:
        # Stop all the sniffer threads if there is a KeyboardInterrupt (Ctrl+C)
        for sniffer_thread in sniffer_threads:
            sniffer_thread.stop()

    finally:
        # Join all the sniffer threads
        for sniffer_thread in sniffer_threads:
            sniffer_thread.join()
