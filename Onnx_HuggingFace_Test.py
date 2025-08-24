# import torch
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# import os
#
# # --- 第1步：加载预训练的 PyTorch 模型 ---
# print("正在从 Hugging Face 下载并加载 PyTorch ResNet-18 模型...")
#
# model_id = "microsoft/resnet-18"
# image_processor = AutoImageProcessor.from_pretrained(model_id)
# model = AutoModelForImageClassification.from_pretrained(model_id)
# model.eval()
#
# print("PyTorch 模型加载成功！")
#
# # --- 第2步：将 PyTorch 模型导出为 ONNX ---
# onnx_model_path = "resnet18.onnx"
#
# if not os.path.exists(onnx_model_path):
#     print(f"\n正在将模型导出为 ONNX 格式，保存至: {onnx_model_path}")
#
#     # --- 【修改点在这里】 ---
#     # 对于ResNet，我们直接使用标准的224x224尺寸来创建虚拟输入
#     # 不再从 image_processor.size 中读取，避免 KeyError
#     dummy_input = torch.randn(1, 3, 224, 224)
#
#     # 执行导出
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_model_path,
#         export_params=True,
#         opset_version=12,
#         do_constant_folding=True,
#         input_names=['input'],
#         output_names=['output'],
#         dynamic_axes={'input': {0: 'batch_size'},
#                       'output': {0: 'batch_size'}}
#     )
#
#     print("模型成功导出为 ONNX！")
# else:
#     print(f"\nONNX 模型文件 '{onnx_model_path}' 已存在。")
#
# if os.path.exists(onnx_model_path):
#     print(f"\n成功！最终目标 '{onnx_model_path}' 已准备就绪。")

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

# --- 第1步：加载预训练的 Vision Transformer (ViT) 模型 ---
print("正在从 Hugging Face 下载并加载 ViT 模型...")

model_id = "google/vit-base-patch16-224"

image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)
model.eval()

print("ViT 模型和图像处理器加载成功！")

# --- 第2步：将模型导出为 ONNX ---
onnx_model_path = "vit_b_16_224.onnx"

if not os.path.exists(onnx_model_path):
    print(f"\n正在将模型导出为 ONNX 格式，保存至: {onnx_model_path}")

    dummy_input = torch.randn(1, 3, 224, 224)

    # 执行导出
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=14,  # <-- 【关键修改点】从 12 提高到 14
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

    print("模型成功导出为 ONNX！")
else:
    print(f"\nONNX 模型文件 '{onnx_model_path}' 已存在。")

if os.path.exists(onnx_model_path):
    print(f"\n成功！模型 '{onnx_model_path}' 已准备就绪。")