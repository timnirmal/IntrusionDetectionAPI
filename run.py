import subprocess

if __name__ == '__main__':
    print("Starting...")

    # Start a background process
    process_fastapi = subprocess.Popen("uvicorn app:app --reload", shell=True, stderr=subprocess.PIPE)
    process_cicflow = subprocess.Popen("python run_cicflow.py", shell=True, stderr=subprocess.PIPE)

    # Wait for the subprocesses to finish (optional)
    process_cicflow.wait()
    process_fastapi.wait()

    # Print standard error output (if any)
    print("run_cicflow.py errors:", process_cicflow.stderr.read())
    print("app.py fastapi errors:", process_fastapi.stderr.read())
