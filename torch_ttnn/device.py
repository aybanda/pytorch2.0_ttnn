class DispatchCoreType:
    ETH = "ETH"
    WORKER = "WORKER"

class DispatchCoreAxis:
    ROW = "ROW"
    COL = "COL"

class DispatchCoreConfig:
    def __init__(self, core_type, core_axis):
        self.core_type = core_type
        self.core_axis = core_axis

def open_device(device_id=0, dispatch_core_config=None, l1_small_size=None):
    class DummyDevice:
        def enable_program_cache(self):
            pass
        def disable_and_clear_program_cache(self):
            pass
    return DummyDevice()

def open_mesh_device(mesh_shape, dispatch_core_config=None, l1_small_size=None):
    class DummyDevice:
        def enable_program_cache(self):
            pass
        def disable_and_clear_program_cache(self):
            pass
    return DummyDevice()

def close_device(device):
    pass

def close_mesh_device(device):
    pass

def synchronize_device(device):
    pass

def SetDefaultDevice(device):
    pass

def is_grayskull(device):
    return False

def is_wormhole_b0(device):
    return False 