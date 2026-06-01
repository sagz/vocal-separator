import sys
import json
import os

from separate import SeperateMDX, SeperateMDXC, SeperateVR, SeperateDemucs

class DummyModelData:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

def main():
    if len(sys.argv) < 3:
        print("UVR_CLI_BRIDGE_CONSOLE:Error: Missing required arguments.")
        sys.exit(1)
        
    model_data_file = sys.argv[1]
    process_data_file = sys.argv[2]
    
    with open(model_data_file, 'r') as f:
        model_data_dict = json.load(f)
        
    with open(process_data_file, 'r') as f:
        process_data_dict = json.load(f)
        
    def dummy_progress_bar(val):
        print(f"UVR_CLI_BRIDGE_PROGRESS:{val}", flush=True)

    def dummy_write_to_console(msg):
        print(f"UVR_CLI_BRIDGE_CONSOLE:{msg}", flush=True)

    def dummy_process_iteration():
        pass
        
    def dummy_cached_source_callback(process_method, model_name=None):
        return None, None
        
    def dummy_cached_model_source_holder(process_method, sources, model_name=None):
        pass

    process_data_dict['set_progress_bar'] = dummy_progress_bar
    process_data_dict['write_to_console'] = dummy_write_to_console
    process_data_dict['process_iteration'] = dummy_process_iteration
    process_data_dict['cached_source_callback'] = dummy_cached_source_callback
    process_data_dict['cached_model_source_holder'] = dummy_cached_model_source_holder

    model_data = DummyModelData(model_data_dict)
    
    arch_type = getattr(model_data, 'process_method', None)
    try:
        if arch_type == 'VR Arc': 
            seperator = SeperateVR(model_data, process_data_dict)
        elif arch_type == 'MDX-Net': 
            seperator = SeperateMDXC(model_data, process_data_dict) if getattr(model_data, 'is_mdx_c', False) else SeperateMDX(model_data, process_data_dict)
        elif arch_type == 'Demucs': 
            seperator = SeperateDemucs(model_data, process_data_dict)
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")
            
        seperator.seperate()
        print("UVR_CLI_BRIDGE_DONE", flush=True)
    except Exception as e:
        import traceback
        print(f"UVR_CLI_BRIDGE_CONSOLE:Error in external process: {str(e)}", flush=True)
        print(f"UVR_CLI_BRIDGE_CONSOLE:{traceback.format_exc()}", flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
