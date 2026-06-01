import os
import sys
import importlib.util

PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plugins')

def load_plugins():
    plugins = {}
    if not os.path.exists(PLUGIN_DIR):
        os.makedirs(PLUGIN_DIR)
        return plugins
        
    for item in os.listdir(PLUGIN_DIR):
        plugin_path = os.path.join(PLUGIN_DIR, item)
        if os.path.isdir(plugin_path):
            init_file = os.path.join(plugin_path, '__init__.py')
            if os.path.isfile(init_file):
                spec = importlib.util.spec_from_file_location(item, init_file)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[item] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, 'PLUGIN_NAME'):
                    plugins[mod.PLUGIN_NAME] = mod
    return plugins

PLUGINS = load_plugins()
