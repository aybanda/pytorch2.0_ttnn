# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import re
import numpy as np


def GraphCleanup(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    return gm


def get_shape(gm: torch.fx.GraphModule, node_or_shape):
    """
    Get the shape of a node or shape itself.

    Args:
        gm (torch.fx.GraphModule): The GraphModule containing the node.
        node_or_shape: The node or shape to get the shape of. Can be an int, float, torch.Size, list, tuple, torch.fx.node.Node, or torch.fx.proxy.Proxy.

    Returns:
        torch.Size or None: The shape of the node or shape itself, or None if it cannot be determined.
    """
    if isinstance(node_or_shape, torch.fx.proxy.Proxy):
        node_or_shape = node_or_shape.node

    if isinstance(node_or_shape, (int, float)):
        return torch.Size()
    if isinstance(node_or_shape, (torch.Size, list, tuple)):
        return node_or_shape
    if isinstance(node_or_shape, torch.fx.node.Node):
        if (val := node_or_shape.meta.get("val", None)) is not None:
            return val.size()

        if node_or_shape.op == "get_attr":
            if gm is None:
                return None
            val = getattr(gm, node_or_shape.target)
            if isinstance(val, torch.Tensor):
                return val.size()
            if isinstance(val, (int, float)):
                return torch.Size()

    return None


def get_arg(node, index, name, default=None):
    if hasattr(node, "args") and len(node.args) > index:
        return node.args[index]
    if hasattr(node, "kwargs") and name in node.kwargs:
        return node.kwargs[name]
    return default


def get_dtype(node):
    if isinstance(node, torch.fx.node.Node):
        if (val := node.meta.get("val", None)) is not None:
            return val.dtype
    return None


def get_opname(node):
    if str(node.target).startswith("aten."):
        return str(node.target)
    elif hasattr(node.target, "__name__"):
        return node.target.__name__
    elif isinstance(node.op, str):
        return node.target
    else:
        raise


def users_have_getitem(node):
    for user in list(node.users.keys()):
        if user.op != "output" and user.op != "placeholder" and user.target.__name__ == "getitem":
            return user
    return None


def is_operation(node):
    return node.op not in ["placeholder", "output"]


def get_meta_val_attr(node, attr: str):
    """
    Example usage in condition statement:
    # Important to wrap var assignment in parenthesis because of order of operations
    if (var := get_meta_val_attr(node, "attr")) and str(attr) == "some_val":
        ...
    """
    if "val" in node.meta:
        return getattr(node.meta["val"], attr, False)
    else:
        return False


# Certain ops don't support certain shapes and will emit a valid_page_size error
# RuntimeError: TT_FATAL @ ../tt_metal/impl/buffers/buffer.cpp:38: valid_page_size
# For valid non-interleaved buffers page size 2048 must equal buffer size X. For interleaved-buffers page size should be divisible by buffer size
def HasValidPageSize(shape, strict=False):
    if len(shape) >= 2 and shape[-1] > 0:
        return shape[-1] % 32 == 0 or (not strict and shape[-1] < 32)
    return False


# Ttnn globals added with torch.fx._register_custom_builtin
class TtnnDevice:
    def __repr__(self):
        return f"ttnn_Specified_Device"


class TtnnRowMajorLayout:
    def __repr__(self):
        return f"ttnn_ROW_MAJOR_LAYOUT"


class TtnnTileLayout:
    def __repr__(self):
        return f"ttnn_TILE_LAYOUT"


class TtnnInt32:
    def __repr__(self):
        return f"ttnn_int32"


class TtnnUint32:
    def __repr__(self):
        return f"ttnn_uint32"


class TtnnBfloat16:
    def __repr__(self):
        return f"ttnn_bfloat16"


class TtnnDramMemoryConfig:
    def __repr__(self):
        return f"ttnn_DRAM_MEMORY_CONFIG"


class TtnnL1MemoryConfig:
    def __repr__(self):
        return f"ttnn_L1_MEMORY_CONFIG"


# repr_str -> (object_wrapper, object)
__custom_objects_registry = {}


# Note: key MUST be unique!
def get_add_custom_object_in_graph(key: str, obj):
    """
    Register a custom object in the FX graph with a unique string representation.

    This function creates a wrapper object that represents the given object in the FX graph.
    The wrapper's __repr__ returns the provided key, ensuring the object can be properly
    serialized and deserialized within the graph.

    Args:
        key (str): A unique string identifier for the object in the graph.
        obj: The actual object to be wrapped and registered.

    Returns:
        WrapperObj: A wrapper object that represents the original object in the FX graph.

    Side effects:
        - Registers the key as a custom builtin in torch.fx.graph
        - Adds the wrapper and original object to __custom_objects_registry
    """
    if key in __custom_objects_registry:
        return __custom_objects_registry[key][0]

    class WrapperObj:
        def __repr__(self):
            return key

    torch.fx.graph._register_custom_builtin(key, "", obj)
    __custom_objects_registry[key] = (WrapperObj(), obj)
    return __custom_objects_registry[key][0]


def get_emplace_custom_object_in_graph(object_type, *args, **kwargs):
    """
    Create an instance of object_type and register it in the FX graph with a unique name.

    This function creates a unique string identifier based on the object's class name and parameters (args, kwargs),
    then instantiates the object and registers it in the graph using get_add_custom_object_in_graph.

    Args:
        object_type: The class to instantiate
        *args: Positional arguments to pass to the object's constructor
        **kwargs: Keyword arguments to pass to the object's constructor

    Returns:
        WrapperObj: A wrapper object that represents the instantiated object in the FX graph.
        You should use this object when passing to ttnn ops.

    Example:
        >>> program_config = get_emplace_custom_object_in_graph(
        >>>     ttnn.SDPAProgramConfig,
        >>>     compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        >>>     q_chunk_size=default_q_chunk_size,
        >>>     k_chunk_size=default_k_chunk_size,
        >>>     exp_approx_mode=False,
        >>> )
        >>> res_node = g.call_function(ttnn.transformer.scaled_dot_product_attention, args, kwargs={"program_config": program_config})

    Side effects:
        - Registers the key as a custom builtin in torch.fx.graph
        - Adds the wrapper and original object to __custom_objects_registry
    """

    def sanitize(s):
        # Replace non-alphanumeric characters with underscores
        s = re.sub(r"[^0-9a-zA-Z_]", "_", str(s))
        # Remove consecutive underscores
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    # This representation of an object MUST be unique
    def build_repr():
        repr_str = f"{object_type.__name__}"
        for arg in args:
            repr_str += f"_{sanitize(arg)}"
        for k, v in kwargs.items():
            repr_str += f"_{sanitize(k)}_{sanitize(v)}"
        return repr_str

    key = build_repr()

    return get_add_custom_object_in_graph(key, object_type(*args, **kwargs))


def comp_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden)
    calculated = torch.Tensor(calculated)

    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return True, 1.0

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        return False, 0.0

    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        return False, 0.0

    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, 1.0

    return cal_pcc >= pcc, cal_pcc


def construct_pcc_assert_message(message, expected_pytorch_result, actual_pytorch_result):
    messages = []
    messages.append(message)
    messages = [str(m) for m in messages]
    return "\n".join(messages)


def assert_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.999):
    assert list(expected_pytorch_result.shape) == list(
        actual_pytorch_result.shape
    ), f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}"
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    assert pcc_passed, construct_pcc_assert_message(pcc_message, expected_pytorch_result, actual_pytorch_result)
    return pcc_passed, pcc_message


def get_dispatch_core_type():
    import torch_ttnn as ttnn
    return ttnn.device.DispatchCoreType.ETH


def get_dispatch_core_axis():
    import torch_ttnn as ttnn
    return ttnn.DispatchCoreAxis.ROW


def get_dispatch_core_config():
    dispatch_core_type = get_dispatch_core_type()
    dispatch_core_axis = get_dispatch_core_axis()
    import torch_ttnn as ttnn
    dispatch_core_config = ttnn.DispatchCoreConfig(dispatch_core_type, dispatch_core_axis)
    return dispatch_core_config
