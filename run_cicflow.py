import psutil

from cicflowmeter.sniffer import create_sniffer
import threading
import time
import pandas as pd

def read_flow_csv():
    while True:
        # print("Reading flow.csv")
        if not os.path.exists("flow.csv"):
            time.sleep(1)  # Wait for the file to be created by the sniffer
            continue

        try:
            df = pd.read_csv("flow.csv")
            if not df.empty:
                print(df)  # You can replace this with whatever you want to do with the data
        except pd.errors.EmptyDataError:
            print("flow.csv is empty. \t time = ", time.strftime("%H:%M:%S", time.localtime()))
            # pass  # Handle the case when the file is empty

        time.sleep(5)  # Adjust the interval for reading the file here


def get_interfaces():
    import netifaces
    addrs = psutil.net_if_addrs()
    return addrs.keys()


if __name__ == '__main__':
    interfaces = get_interfaces()
    # show list of interfaces
    print("Interfaces: ", get_interfaces())

    # print start time
    print("Start time: ", time.strftime("%H:%M:%S", time.localtime()))

    # delete the flow.csv file
    import os
    if os.path.exists("flow.csv"):
        os.remove("flow.csv")

    sniffer = create_sniffer(
        input_file=None,
        input_interface="WiFi",
        output_mode="flow",
        output_file="flow.csv",
        url_model=None
    )
    sniffer.start()

    try:
        reader_thread = threading.Thread(target=read_flow_csv)
        reader_thread.start()
        sniffer.join()
    except KeyboardInterrupt:
        sniffer.stop()
    finally:
        sniffer.join()
