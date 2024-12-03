import subprocess
import time
import os
import sys

def run_frontend():
    """Function to start the frontend using Streamlit."""
    frontend_command = ["streamlit", "run", "Frontend/user_interface.py", "--server.runOnSave=true"]
    
    try:
        subprocess.run(frontend_command)
    except Exception as e:
        print(f"Error starting frontend: {e}")

def run_backend():
    """Function to start the backend using uvicorn."""
    backend_command = ["uvicorn", "Backend.main:app", "--reload"]
    
    try:
        subprocess.run(backend_command)
    except Exception as e:
        print(f"Error starting backend: {e}")

if __name__ == "__main__":
    print("Starting Backend and Frontend...")

    # Start backend and frontend using subprocess in a way that restarts correctly
    try:
        # Run frontend in a separate process and ensure it reloads properly
        frontend_process = subprocess.Popen(["streamlit", "run", "Frontend/user_interface.py", "--server.runOnSave=true"])

        # Run the backend in the same way
        backend_process = subprocess.Popen(["uvicorn", "Backend.main:app", "--reload"])

        # Monitor for any issues and wait for the processes to finish
        frontend_process.wait()
        backend_process.wait()

    except KeyboardInterrupt:
        print("Shutting down both processes...")

        # Gracefully terminate the frontend and backend when exiting
        frontend_process.terminate()
        backend_process.terminate()

        sys.exit(0)
