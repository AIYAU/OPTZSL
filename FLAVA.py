import torch
from transformers import AutoProcessor, FlavaModel
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import pandas as pd
import os
from glob import glob
from PIL import Image
import prompt

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 FLAVA 模型和处理器
model_name = r"D:\Small sample\molmo\FLAVA"  # 使用 FLAVA 模型
processor = AutoProcessor.from_pretrained(model_name)
model = FlavaModel.from_pretrained(model_name).to(device)

# 通过变量来获取prompt，例如
use_prompt = prompt.PP_prompt  # dict
result = []

for key in use_prompt.keys():
    # 获取每个类的第一段描述
    custom_text = use_prompt[key]

    def flava_classify(image: torch.Tensor, classes: list, model: FlavaModel = model) -> int:
        text_inputs = processor(text=custom_text, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # 获取图像特征和文本特征
            image_embeds = model.get_image_features(pixel_values=image)
            text_embeds = model.get_text_features(input_ids=text_inputs['input_ids'],
                                                  attention_mask=text_inputs['attention_mask'])

            # 取出主要的图像嵌入和文本嵌入
            image_embeds_main = image_embeds[:, 0, :]
            text_embeds_main = text_embeds[:, 0, :]

            # 计算相似度（对比图像和文本特征）
            similarity = torch.matmul(image_embeds_main, text_embeds_main.T).softmax(dim=-1).cpu().numpy()

        # 返回概率最大的类别索引
        return similarity.argmax()

    def make_dataset(path: str):
        image_paths = glob(os.path.join(path, '*.jpg')) + glob(os.path.join(path, '*.png'))
        images = [Image.open(p).convert("RGB") for p in image_paths]  # 转换为 RGB 格式
        images_preprocessed = [processor(images=[img], return_tensors="pt")['pixel_values'].to(device) for img in images]
        return images_preprocessed

    # 设置图像数据集路径
    DATA_PATH_APPLES = r'D:\Small sample\molmo\dataset'
    images_preprocessed = make_dataset(DATA_PATH_APPLES)

    # 类别标签
    fixed_labels = ["Alternaria", "Blossom", "Brown", "Grey", "Healthy", "Mosaic", "Mildew", "Rust", "Scab"]

    # 初始化真实标签和预测标签
    num_images_per_class = [103, 27, 93, 92, 134, 201, 351, 197, 147]
    y_true = [i for i, count in enumerate(num_images_per_class) for _ in range(count)]
    y_pred = []

    # 分类图像
    for i, image_pre in enumerate(images_preprocessed):
        pred = flava_classify(image_pre, fixed_labels)
        y_pred.append(pred)

    # 转换为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算F1分数和准确性
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = np.mean(y_true == y_pred)

    # 计算Sensitivity (Recall)
    sensitivity = recall_score(y_true, y_pred, average=None)

    # 计算Specificity
    specificity = []
    for i in range(len(fixed_labels)):
        tn = np.sum(np.delete(np.delete(conf_matrix, i, axis=0), i, axis=1))  # True Negatives
        fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]  # False Positives
        specificity.append(tn / (tn + fp))

    # 输出加权F1得分、准确率、Sensitivity (Recall)和Specificity
    print(f"加权F1得分: {f1}")
    print(f"准确率: {accuracy}")
    print(f"Sensitivity (Recall): {sensitivity}")
    print(f"Specificity: {specificity}")
print(result)

