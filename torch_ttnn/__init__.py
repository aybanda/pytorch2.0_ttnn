# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# The inferface of this backend

from .backend import ttnn_backend as backend
from .backend import TorchTtnnOption
import torch as _torch

_torch._dynamo.backends.registry.register_backend(name="ttnn", compiler_fn=backend)

# To wrap the ttnn ops
from .passes.lowering import target_wrappers

try:
    import torch_ttnn as ttnn
except ImportError as e:
    print(
        "ttnn is not installed. Run `python3 -m pip install -r requirements.txt` or `python3 -m pip install -r requirements-dev.txt` if you are developing the compiler"
    )
    raise e

from . import device
from .device import DispatchCoreType, DispatchCoreAxis, DispatchCoreConfig, open_device, open_mesh_device, close_device, close_mesh_device, synchronize_device, SetDefaultDevice
from . import experimental

from .utils import TtnnDevice as Device, TtnnRowMajorLayout as ROW_MAJOR_LAYOUT, TtnnTileLayout as TILE_LAYOUT, TtnnUint32 as uint32, TtnnInt32 as int32, TtnnBfloat16 as bfloat16, TtnnDramMemoryConfig as DRAM_MEMORY_CONFIG, TtnnL1MemoryConfig as L1_MEMORY_CONFIG

# Dynamically expose C++ extension ops (e.g., add, mul, etc.) as ttnn.add, ttnn.mul, etc.
import importlib

def __getattr__(name):
    try:
        ttnn_ext = importlib.import_module("ttnn_device_extension")
        return getattr(ttnn_ext, name)
    except (ImportError, AttributeError) as e:
        raise AttributeError(f"module 'torch_ttnn' has no attribute '{name}' and could not find it in ttnn_device_extension") from e
