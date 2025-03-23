import os
import numpy as np
import torch
import clip
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from PIL import Image
from glob import glob
from tabulate import tabulate
import pandas as pd
from IPython.display import Markdown, display
import prompt

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


# 定义图像分类函数
def clip_classify(image: torch.Tensor, classes: list, model: clip.model.CLIP = model) -> int:
    text = clip.tokenize(classes).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 打印每个类别的预测概率
    class_prob_list = [(c, round(p, 3)) for c, p in zip(classes, probs.tolist()[0])]
    class_prob_list = sorted(class_prob_list, key=lambda x: x[1], reverse=True)

    print(tabulate(class_prob_list, headers=["Class", "Probability"]))
    return probs.argmax()


# 数据集准备和预处理
def make_dataset(path: str):
    image_paths = glob(os.path.join(path, '*.jpg')) + glob(os.path.join(path, '*.png'))
    image_names = [os.path.basename(p) for p in image_paths]
    images = [Image.open(p) for p in image_paths]
    images_preprocessed = [preprocess(Image.open(p)).unsqueeze(0).to(device) for p in image_paths]
    return image_names, images, images_preprocessed


# 设置图像数据集路径
DATA_PATH_APPLES = r'D:\Small sample\molmo\dataset'
image_names, images, images_preprocessed = make_dataset(DATA_PATH_APPLES)

# Classes = []
# #通过变量来获取prompt例如
# use_prompt = prompt.claude_prompt # dict
#
# for key in use_prompt.keys():
#     temp_ls = ''
#     # 顺序拼接
#     # for i in range(3): # 拼接n个句子
#     #     temp_ls = temp_ls + str(use_prompt[key][i])
#     #
#     # Classes.append(temp_ls)
#     # 随机拼接1-10两个句子
#     # temp_ls = temp_ls + str(use_prompt[key][np.random.randint(0, 10)])
#     # temp_ls = temp_ls + str(use_prompt[key][np.random.randint(0, 10)])
#     # Classes.append(temp_ls)
#     # 指定索引拼接两个句子
#     temp_ls = temp_ls + str(use_prompt[key][10])
#     # temp_ls = temp_ls + str(use_prompt[key][1])
#     Classes.append(temp_ls)

Classes = []
#通过变量来获取prompt例如
use_prompt = prompt.PP_prompt # dict

result=[]

for key in use_prompt.keys():
    # 获取每个类的第一段描述
    Classes = use_prompt[key]
    # 初始化真实标签和预测标签
    num_images_per_class = [103, 27, 93, 92, 134, 201, 351, 197, 147]
    y_true = [i for i, count in enumerate(num_images_per_class) for _ in range(count)]
    y_pred = []

    # 分类图像
    for i, image_pre in enumerate(images_preprocessed):
        print(f"分类 {image_names[i]}...")
        pred = clip_classify(image_pre, Classes)
        y_pred.append(pred)

    # 转换为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # 计算F1分数和准确性
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = np.mean(y_true == y_pred)
    result.append(f1)
    result.append(accuracy)

    # 计算Sensitivity (Recall) 和 Specificity
    sensitivity = recall_score(y_true, y_pred, average=None)
    specificity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

    print(f"混淆矩阵:\n{conf_matrix}")
    print(f"加权F1得分: {f1}")
    print(f"准确率: {accuracy}")
    print(f"Sensitivity (Recall): {sensitivity}")
    print(f"Specificity: {specificity}")

    # 保存评估结果为CSV
    metrics_df = pd.DataFrame({
        'Class': Classes,
        'F1': f1_score(y_true, y_pred, average=None),
        'Sensitivity': sensitivity,
        'Specificity': specificity
    })
    metrics_df.to_csv('evaluation_metrics.csv', index=False)

    fixed_labels = ["Alternaria", "Blossom", "Brown", "Grey", "Healthy", "Mosaic", "Mildew", "Rust", "Scab"]

    # 定义绘制混淆矩阵的函数，使用固定的标签
    def plot_confusion_matrix(conf_matrix, fixed_labels):
        plt.figure(figsize=(10, 7))
        plt.imshow(conf_matrix, cmap='Blues', interpolation='none')
        plt.title("Confusion Matrix")
        plt.colorbar()

        # 使用固定的类名
        tick_marks = np.arange(len(fixed_labels))
        plt.xticks(tick_marks, fixed_labels, rotation=45, ha="right")
        plt.yticks(tick_marks, fixed_labels)

        # 添加标签
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        # 在每个格子中添加具体的值
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                         ha="center", va="center", color="black")

        plt.tight_layout()
        plt.show()

    plot_confusion_matrix(conf_matrix, fixed_labels)

print(result)



