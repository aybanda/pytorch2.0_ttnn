import torch
import numpy as np

def comp_pcc(golden, calculated, pcc=0.99):
    golden = torch.Tensor(golden)
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

def check_with_pcc(expected_pytorch_result, actual_pytorch_result, pcc=0.999):
    if expected_pytorch_result.shape != actual_pytorch_result.shape:
        return (
            False,
            f"list(expected_pytorch_result.shape)={list(expected_pytorch_result.shape)} vs list(actual_pytorch_result.shape)={list(actual_pytorch_result.shape)}",
        )
    pcc_passed, pcc_message = comp_pcc(expected_pytorch_result, actual_pytorch_result, pcc)
    return pcc_passed, construct_pcc_assert_message(pcc_message, expected_pytorch_result, actual_pytorch_result) 