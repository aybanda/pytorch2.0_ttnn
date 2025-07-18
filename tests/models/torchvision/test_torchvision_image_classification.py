# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torchvision import models, transforms
from PIL import Image
import torch
import requests
import pytest
from tests.utils import ModelTester, repeat_inputs


class ThisTester(ModelTester):
    # pass model_info instead of model_name
    def __init__(self, model_info, mode, batch_size=1):
        if mode not in ["train", "eval"]:
            raise ValueError(f"Current mode is not supported: {mode}")
        self.model_info = model_info
        self.mode = mode
        self.model = self._load_model()
        self.batch_size = batch_size
        self.inputs = self._load_inputs(self.batch_size)

    def _load_model(self):
        model_name, weights_name = self.model_info
        self.weights = getattr(models, weights_name).DEFAULT
        model = models.get_model(model_name, weights=self.weights).to(torch.bfloat16)
        return model

    def _load_inputs(self, batch_size):
        preprocess = self.weights.transforms()
        # Load and preprocess the image
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0).to(torch.bfloat16)
        batch_t = repeat_inputs(batch_t, batch_size)
        return batch_t


model_info_and_mode_list = [
    [("googlenet", "GoogLeNet_Weights"), "eval"],
    pytest.param([("densenet121", "DenseNet121_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("densenet161", "DenseNet161_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param(
        [("densenet169", "DenseNet169_Weights"), "eval"],
        marks=pytest.mark.converted_end_to_end,
    ),
    pytest.param(
        [("densenet201", "DenseNet201_Weights"), "eval"],
        marks=pytest.mark.compilation_xfail(reason="Fail with more run_once"),
    ),
    pytest.param([("mobilenet_v2", "MobileNet_V2_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param(
        [("mobilenet_v3_small", "MobileNet_V3_Small_Weights"), "eval"], marks=pytest.mark.converted_end_to_end
    ),
    pytest.param(
        [("mobilenet_v3_large", "MobileNet_V3_Large_Weights"), "eval"], marks=pytest.mark.converted_end_to_end
    ),
    pytest.param([("resnet18", "ResNet18_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("resnet34", "ResNet34_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("resnet50", "ResNet50_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("resnet101", "ResNet101_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("resnet152", "ResNet152_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("resnext50_32x4d", "ResNeXt50_32X4D_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("resnext101_32x8d", "ResNeXt101_32X8D_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("resnext101_64x4d", "ResNeXt101_64X4D_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("vgg11", "VGG11_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("vgg11_bn", "VGG11_BN_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("vgg13", "VGG13_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("vgg13_bn", "VGG13_BN_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("vgg16", "VGG16_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("vgg16_bn", "VGG16_BN_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("vgg19", "VGG19_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("vgg19_bn", "VGG19_BN_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    [("vit_b_16", "ViT_B_16_Weights"), "eval"],
    [("vit_b_32", "ViT_B_32_Weights"), "eval"],
    [("vit_l_16", "ViT_L_16_Weights"), "eval"],
    [("vit_l_32", "ViT_L_32_Weights"), "eval"],
    [("vit_h_14", "ViT_H_14_Weights"), "eval"],
    pytest.param([("wide_resnet50_2", "Wide_ResNet50_2_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("wide_resnet101_2", "Wide_ResNet101_2_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_y_400mf", "RegNet_Y_400MF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_y_800mf", "RegNet_Y_800MF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_y_1_6gf", "RegNet_Y_1_6GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_y_3_2gf", "RegNet_Y_3_2GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_y_8gf", "RegNet_Y_8GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_y_16gf", "RegNet_Y_16GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_y_32gf", "RegNet_Y_32GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    [("regnet_y_128gf", "RegNet_Y_128GF_Weights"), "eval"],
    pytest.param([("regnet_x_400mf", "RegNet_X_400MF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_x_800mf", "RegNet_X_800MF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_x_1_6gf", "RegNet_X_1_6GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_x_3_2gf", "RegNet_X_3_2GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_x_8gf", "RegNet_X_8GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_x_16gf", "RegNet_X_16GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    pytest.param([("regnet_x_32gf", "RegNet_X_32GF_Weights"), "eval"], marks=pytest.mark.converted_end_to_end),
    [("swin_t", "Swin_T_Weights"), "eval"],
    [("swin_s", "Swin_S_Weights"), "eval"],
    [("swin_b", "Swin_B_Weights"), "eval"],
    [("swin_v2_t", "Swin_V2_T_Weights"), "eval"],
    [("swin_v2_s", "Swin_V2_S_Weights"), "eval"],
    [("swin_v2_b", "Swin_V2_B_Weights"), "eval"],
]


@pytest.mark.parametrize("model_info_and_mode", model_info_and_mode_list)
def test_torchvision_image_classification(record_property, model_info_and_mode):
    model_info = model_info_and_mode[0]
    mode = model_info_and_mode[1]
    model_name, _ = model_info
    record_property("model_name", model_name)
    record_property("mode", mode)

    tester = ThisTester(model_info, mode)
    results = tester.test_model()
    if mode == "eval":
        # Print the top 5 predictions
        _, indices = torch.topk(results, 5)
        print(f"Model: {model_name} | Top 5 predictions: {indices[0].tolist()}")

    record_property("torch_ttnn", (tester, results))
