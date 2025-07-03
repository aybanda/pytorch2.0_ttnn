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
