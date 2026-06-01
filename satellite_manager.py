import os
import sys
import json
import subprocess
import threading
import logging
import shutil
import venv
import queue
import importlib.metadata
from typing import Dict, List, Optional, Callable, Any

logger = logging.getLogger("SatelliteManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class SatelliteEnvironmentManager:
    """
    Manages isolated satellite environments for model execution to prevent
    dependency conflicts and version mismatches.
    """
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.base_dir = os.path.join(os.getcwd(), "satellite_envs")
        else:
            self.base_dir = base_dir
            
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
            
        self._active_processes: List[subprocess.Popen] = []
        self._setup_threads: Dict[str, threading.Thread] = {}
        
        import atexit
        atexit.register(self.cleanup_processes)

    def get_env_dir(self, env_name: str) -> str:
        return os.path.join(self.base_dir, env_name)

    def is_env_ready(self, env_name: str) -> bool:
        env_dir = self.get_env_dir(env_name)
        if not os.path.exists(os.path.join(env_dir, "ready.marker")):
            return False
            
        python_exe = os.path.join(env_dir, "bin", "python") if os.name != 'nt' else os.path.join(env_dir, "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            return False
            
        try:
            subprocess.check_call([python_exe, "-c", "import sys; sys.exit(0)"], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, OSError):
            return False

    def detect_environment_mismatch(self, required_packages: Dict[str, str]) -> bool:
        if not required_packages:
            return False
        for pkg, req_version in required_packages.items():
            try:
                current_version = importlib.metadata.version(pkg)
                if current_version != req_version:
                    return True
            except importlib.metadata.PackageNotFoundError:
                return True
        return False

    def setup_environment(self, env_name: str, requirements: List[str], on_complete: Optional[Callable[[bool, str], None]] = None):
        if env_name in self._setup_threads and self._setup_threads[env_name].is_alive():
            logger.info(f"Environment {env_name} is already being setup.")
            return

        def task():
            env_dir = self.get_env_dir(env_name)
            try:
                if os.path.exists(env_dir) and not self.is_env_ready(env_name):
                    logger.info(f"Repairing corrupted environment: {env_name}")
                    shutil.rmtree(env_dir)
                    
                if not os.path.exists(env_dir):
                    logger.info(f"Creating virtual environment: {env_name}")
                    venv.create(env_dir, with_pip=True)
                
                pip_exe = os.path.join(env_dir, "bin", "pip") if os.name != 'nt' else os.path.join(env_dir, "Scripts", "pip.exe")
                
                if requirements:
                    logger.info(f"Installing requirements for {env_name}...")
                    subprocess.check_call([pip_exe, "install", "--quiet"] + requirements)
                
                with open(os.path.join(env_dir, "ready.marker"), "w") as f:
                    f.write("ready")
                
                logger.info(f"Environment {env_name} is ready.")
                if on_complete:
                    on_complete(True, "")
            except Exception as e:
                logger.error(f"Failed to setup environment {env_name}: {e}")
                if on_complete:
                    on_complete(False, str(e))

        t = threading.Thread(target=task, daemon=True)
        self._setup_threads[env_name] = t
        t.start()

    def pre_warm(self, env_name: str, requirements: List[str]):
        logger.info(f"Pre-warming environment: {env_name}")
        self.setup_environment(env_name, requirements)

    def run_rpc(self, env_name: str, script_path: str, args: List[str],
                on_progress: Callable[[Any], None],
                on_result: Callable[[Any], None],
                on_error: Callable[[str], None]):
        
        if not self.is_env_ready(env_name):
            on_error(f"Environment {env_name} is not ready or is corrupted.")
            return

        env_dir = self.get_env_dir(env_name)
        python_exe = os.path.join(env_dir, "bin", "python") if os.name != 'nt' else os.path.join(env_dir, "Scripts", "python.exe")

        def process_task():
            cmd = [python_exe, script_path] + args
            try:
                proc = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    stdin=subprocess.PIPE, 
                    text=True
                )
                self._active_processes.append(proc)
                
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        if msg.get("jsonrpc") == "2.0":
                            method = msg.get("method")
                            params = msg.get("params", {})
                            if method == "progress":
                                on_progress(params)
                            elif method == "result":
                                on_result(params)
                            elif method == "log":
                                logger.info(f"[{env_name}] {params.get('message', '')}")
                    except json.JSONDecodeError:
                        logger.debug(f"[{env_name} STDOUT] {line}")

                stderr_output = proc.stderr.read()
                if stderr_output:
                    logger.error(f"[{env_name} ERROR] {stderr_output}")
                    
                    # Extract human-readable error from stack trace
                    human_readable = stderr_output.strip()
                    if "Traceback (most recent call last):" in human_readable:
                        lines = human_readable.split('\n')
                        # The actual error is usually the last line of the traceback
                        for i in range(len(lines) - 1, -1, -1):
                            if lines[i].strip() and not lines[i].startswith(" ") and ":" in lines[i]:
                                human_readable = lines[i].strip()
                                break
                    
                    on_error(f"Error from satellite environment: {human_readable}")

                proc.wait()
                if proc in self._active_processes:
                    self._active_processes.remove(proc)

            except Exception as e:
                on_error(str(e))

        t = threading.Thread(target=process_task, daemon=True)
        t.start()

    def purge_environment(self, env_name: str):
        env_dir = self.get_env_dir(env_name)
        if os.path.exists(env_dir):
            shutil.rmtree(env_dir)
            logger.info(f"Purged satellite environment: {env_name}")

    def cleanup_processes(self):
        for proc in self._active_processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                try:
                    proc.kill()
                except:
                    pass
        self._active_processes.clear()

# Global singleton
satellite_manager = SatelliteEnvironmentManager()
