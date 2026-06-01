import sys
import pickle
import json
import os
import traceback

class MockGUI:
    def __init__(self, task_id):
        self.task_id = task_id
        self.status_file = f"/tmp/{task_id}_status.json"

    def set_progress_bar(self, step, inference_iterations=0):
        try:
            with open(self.status_file, "a") as f:
                f.write(json.dumps({"progress": step + inference_iterations}) + "\n")
        except:
            pass

    def write_to_console(self, text):
        try:
            with open(self.status_file, "a") as f:
                f.write(json.dumps({"log": text}) + "\n")
        except:
            pass
            
    def cached_source_callback(self, *args, **kwargs):
        return None, None

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "UVR"
        return super().find_class(module, name)

def main():
    if len(sys.argv) < 3:
        print("Usage: python worker.py <config_path> <task_id>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    task_id = sys.argv[2]
    
    with open(config_path, "rb") as f:
        task_config = CustomUnpickler(f).load()
        
    # Inject callables
    gui = MockGUI(task_id)
    
    process_data = task_config['process_data']
    process_data['set_progress_bar'] = gui.set_progress_bar
    process_data['write_to_console'] = gui.write_to_console
    process_data['cached_source_callback'] = gui.cached_source_callback
    process_data['cached_model_source_holder'] = lambda *args, **kwargs: None

    current_model = process_data['model_data']
    
    import separate
    from gui_data.constants import VR_ARCH_TYPE, MDX_ARCH_TYPE, DEMUCS_ARCH_TYPE
    
    try:
        if current_model.process_method == VR_ARCH_TYPE:
            seperator = separate.SeperateVR(current_model, process_data)
        elif current_model.process_method == MDX_ARCH_TYPE:
            if getattr(current_model, 'is_mdx_c', False):
                seperator = separate.SeperateMDXC(current_model, process_data)
            else:
                seperator = separate.SeperateMDX(current_model, process_data)
        elif current_model.process_method == DEMUCS_ARCH_TYPE:
            seperator = separate.SeperateDemucs(current_model, process_data)
        else:
            raise ValueError(f"Unknown process method: {current_model.process_method}")
            
        seperator.seperate()
        
    except Exception as e:
        gui.write_to_console(f"Worker Exception: {traceback.format_exc()}\n")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
