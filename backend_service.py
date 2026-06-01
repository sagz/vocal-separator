import xmlrpc.server
import xmlrpc.client
import threading
import subprocess
import os
import sys
import uuid
import json
import time
import pickle

PORT = 8123

class UVRBackendService:
    def __init__(self):
        self.workers = {}
        self.status = {}
        self.lock = threading.Lock()
        
    def start_task(self, task_config_bytes, env_profile="default"):
        task_id = str(uuid.uuid4())
        
        env_python = sys.executable
        if env_profile != "default":
            env_dir = os.path.join("/app/envs", env_profile)
            env_python = os.path.join(env_dir, "bin", "python")
            if not os.path.exists(env_python):
                os.makedirs(os.path.join("/app/envs"), exist_ok=True)
                subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", env_dir], check=True)
                
        # Write config to a temp file
        config_path = f"/tmp/{task_id}_config.pkl"
        with open(config_path, "wb") as f:
            f.write(task_config_bytes.data)
            
        # Initialize status
        with self.lock:
            self.status[task_id] = {"status": "starting", "progress": 0, "log": ""}
            
        # Start worker
        p = subprocess.Popen(
            [env_python, "/app/worker.py", config_path, task_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/app"
        )
        
        with self.lock:
            self.workers[task_id] = p
            self.status[task_id]["status"] = "running"
            
        return task_id
        
    def get_status(self, task_id):
        with self.lock:
            if task_id not in self.status:
                return {"status": "error", "error": "Task not found"}
            
            p = self.workers.get(task_id)
            if p:
                ret = p.poll()
                if ret is not None:
                    if self.status[task_id]["status"] == "running":
                        self.status[task_id]["status"] = "completed" if ret == 0 else "failed"
                        out, err = p.communicate()
                        self.status[task_id]["log"] += f"\nWorker exited with code {ret}\nSTDOUT: {out}\nSTDERR: {err}"
                        
            # Read status updates written by worker if any
            status_file = f"/tmp/{task_id}_status.json"
            if os.path.exists(status_file):
                try:
                    with open(status_file, "r") as f:
                        lines = f.readlines()
                    # clear the file
                    with open(status_file, "w") as f:
                        pass
                    
                    for line in lines:
                        if not line.strip(): continue
                        worker_st = json.loads(line)
                        if "progress" in worker_st:
                            self.status[task_id]["progress"] = worker_st["progress"]
                        if "log" in worker_st:
                            self.status[task_id]["log"] += worker_st["log"]
                except Exception as e:
                    pass
                    
            resp = dict(self.status[task_id])
            # Clear log after reading to avoid transferring huge strings
            self.status[task_id]["log"] = ""
            return resp

    def cancel_task(self, task_id):
        with self.lock:
            if task_id in self.workers:
                p = self.workers[task_id]
                p.terminate()
                self.status[task_id]["status"] = "terminated"
                return True
        return False

def start_server():
    server = xmlrpc.server.SimpleXMLRPCServer(("127.0.0.1", PORT), allow_none=True)
    server.register_instance(UVRBackendService())
    print(f"UVR Backend listening on port {PORT}...")
    server.serve_forever()

if __name__ == "__main__":
    start_server()
