from torch_ttnn import device as ttnn_device

def get_dispatch_core_type():
    # Instead of conditionally returning WORKER or ETH, here we always return ETH
    # Without setting this property, we get less cores availble on N300 than on N150, which might lead to inconsistent and sub-sufficient results
    return ttnn_device.DispatchCoreType.ETH


def get_dispatch_core_axis():
    # Should be ttnn.DispatchCoreAxis.COL for Blackhole
    return ttnn_device.DispatchCoreAxis.ROW


def get_dispatch_core_config():
    dispatch_core_type = get_dispatch_core_type()
    dispatch_core_axis = get_dispatch_core_axis()
    dispatch_core_config = ttnn_device.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
    return dispatch_core_config 