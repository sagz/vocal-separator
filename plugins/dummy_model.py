from core.separation import SeperateAttributes
from core.constants import *

class SeperateDummy(SeperateAttributes):
    plugin_name = "DummyModel"
    
    def seperate(self):
        # A dummy implementation for testing
        print("Processing with DummyModel...")
        # Write some fake output to satisfy any checks
        pass
