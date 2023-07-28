import subprocess

import psutil as psutil

if __name__ == '__main__':
    print("Starting...")

    # uvicorn app:app --reload
    # process_fastapi = subprocess.Popen("uvicorn app:app --reload", shell=True, stderr=subprocess.PIPE)
    # process_fastapi = subprocess.Popen("python main.py", shell=True, stderr=subprocess.PIPE)

    # Start run_cicflow.py in a background process
    process_cicflow = subprocess.Popen("python run_cicflow.py", shell=True, stderr=subprocess.PIPE)

    # Start main.py streamlit in a background process
    # process_streamlit = subprocess.Popen("streamlit run main.py", shell=True, stderr=subprocess.PIPE)

    # Wait for the subprocesses to finish (optional)
    process_cicflow.wait()
    # process_streamlit.wait()
    # process_fastapi.wait()

    # Print standard error output (if any)
    print("run_cicflow.py errors:", process_cicflow.stderr.read())
    # print("main.py streamlit errors:", process_streamlit.stderr.read())
    # print("app.py fastapi errors:", process_fastapi.stderr.read())
