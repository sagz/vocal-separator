import os
import importlib
import inspect
from core.separation import SeperateAttributes
import core.constants

class PluginManager:
    _plugins = {}
    
    @classmethod
    def register(cls, name, plugin_class):
        cls._plugins[name] = plugin_class
        
        # Inject into UI process methods
        methods = list(core.constants.PROCESS_METHODS)
        if name not in methods and name not in (core.constants.VR_ARCH_TYPE, core.constants.VR_ARCH_PM, core.constants.MDX_ARCH_TYPE, core.constants.DEMUCS_ARCH_TYPE):
            # Insert before ENSEMBLE_MODE so it's with the other models
            ensemble_idx = methods.index(core.constants.ENSEMBLE_MODE) if core.constants.ENSEMBLE_MODE in methods else len(methods)
            methods.insert(ensemble_idx, name)
            core.constants.PROCESS_METHODS = tuple(methods)
        
    @classmethod
    def get_plugin(cls, name):
        return cls._plugins.get(name)
        
    @classmethod
    def get_all_plugins(cls):
        return cls._plugins.copy()

    @classmethod
    def load_plugins(cls, plugin_dir="/app/plugins"):
        if not os.path.isdir(plugin_dir):
            return
        for filename in os.listdir(plugin_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = f"plugins.{filename[:-3]}"
                try:
                    module = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, SeperateAttributes) and obj is not SeperateAttributes:
                            # It's a plugin! 
                            if hasattr(obj, 'plugin_name'):
                                cls.register(obj.plugin_name, obj)
                except Exception as e:
                    print(f"Failed to load plugin {filename}: {e}")

class ModelConfig:
    """Framework-agnostic data structure for plugin parameters."""
    def __init__(self, **kwargs):
        self.process_method = kwargs.get("process_method", "")
        self.model_path = kwargs.get("model_path", "")
        self.model_name = kwargs.get("model_name", "")
        self.model_basename = kwargs.get("model_basename", "")
        self.is_pitch_change = kwargs.get("is_pitch_change", False)
        self.semitone_shift = kwargs.get("semitone_shift", 0)
        self.is_match_frequency_pitch = kwargs.get("is_match_frequency_pitch", False)
        self.overlap = kwargs.get("overlap", 0.25)
        self.overlap_mdx = kwargs.get("overlap_mdx", "Default")
        self.overlap_mdx23 = kwargs.get("overlap_mdx23", 8)
        self.is_mdx_combine_stems = kwargs.get("is_mdx_combine_stems", False)
        self.is_mdx_c = kwargs.get("is_mdx_c", False)
        self.mdx_c_configs = kwargs.get("mdx_c_configs", None)
        self.mdxnet_stem_select = kwargs.get("mdxnet_stem_select", "Vocals")
        self.mixer_path = kwargs.get("mixer_path", "")
        self.model_samplerate = kwargs.get("model_samplerate", 44100)
        self.model_capacity = kwargs.get("model_capacity", (32, 128))
        self.is_vr_51_model = kwargs.get("is_vr_51_model", False)
        self.is_pre_proc_model = kwargs.get("is_pre_proc_model", False)
        self.is_secondary_model_activated = kwargs.get("is_secondary_model_activated", False)
        self.is_secondary_model = kwargs.get("is_secondary_model", False)
        self.wav_type_set = kwargs.get("wav_type_set", "PCM_16")
        self.mp3_bit_set = kwargs.get("mp3_bit_set", "320k")
        # Allow any other attributes requested
        for k, v in kwargs.items():
            setattr(self, k, v)
