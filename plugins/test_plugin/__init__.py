import os

PLUGIN_NAME = 'Test Plugin'

CONFIG = [
    {"name": "Window Size", "id": "window_size", "type": "dropdown", "options": ["320", "512", "1024"], "default": "512"},
]

def get_models(app):
    return ["TestModel_A", "TestModel_B"]

def get_model_data(app):
    # Depending on selected model, we mock it
    class MockModel:
        process_method = PLUGIN_NAME
        model_basename = app.plugin_model_var[PLUGIN_NAME].get()
        demucs_4_stem_added_count = 0
        pre_proc_model_activated = False
        is_secondary_model_activated = False
        model_status = True
    return MockModel()

class Seperator:
    def __init__(self, model_data, process_data):
        self.model_data = model_data
        self.process_data = process_data
    
    def seperate(self):
        print(f"Test plugin separating with model {self.model_data.model_basename}!")
        self.process_data['set_progress_bar'](0.5)
        self.process_data['write_to_console']("Test Plugin Progress")
        self.process_data['set_progress_bar'](1.0)
