import os
import numpy as np
import torch
from transformers import BlipProcessor, BlipModel
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
from PIL import Image
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pickle
import prompt

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载BLIP模型的视觉部分和预处理器
processor = BlipProcessor.from_pretrained(r"D:\Small sample\molmo\blip_model")
model = BlipModel.from_pretrained(r"D:\Small sample\molmo\blip_model").vision_model.to(device)

# 类别名称和自定义文本描述
Classes = [
    "Alternaria leaf spot",
    "Blossom blight leaf",
    "Brown spot leaf",
    "Grey spot leaf",
    "Healthy apple leaf",
    "Mosaic apple leaf",
    "Powdery mildew leaf",
    "Rust apple leaf",
    "Scab apple leaf"
]

custom_text = []
#通过变量来获取prompt例如
use_prompt = prompt.PP_prompt # dict

result=[]
for key in use_prompt.keys():
    # 获取每个类的第一段描述
    custom_text = use_prompt[key]
# 使用视觉特征和文本特征进行融合分类
    class CombinedClassifier(nn.Module):
        def __init__(self, num_classes):
            super(CombinedClassifier, self).__init__()
            self.fc1 = nn.Linear(768 + 768, 512)  # 结合视觉特征和文本特征，假设视觉特征和文本特征都是768维
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, features):
            x = self.relu(self.fc1(features))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x


    def extract_features(image_paths, model, processor, custom_text):
        vision_features = []
        text_features = []

        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                vision_feat = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
            vision_features.append(vision_feat)

        text_inputs = processor.tokenizer(custom_text, return_tensors="pt", padding=True).input_ids.to(device)
        with torch.no_grad():
            text_feat = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
            text_features = [text_feat] * len(image_paths)  # 每张图片使用相同的文本特征

        return np.array(vision_features), np.array(text_features)


    def save_precomputed_features(image_paths, vision_features, text_features, path='precomputed_features.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(
                {'image_paths': image_paths, 'vision_features': vision_features, 'text_features': text_features}, f)


    def load_precomputed_features(path='precomputed_features.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['image_paths'], data['vision_features'], data['text_features']


    # 加载预计算特征（如果存在），否则提取并保存
    feature_file = 'precomputed_features.pkl'
    if os.path.exists(feature_file):
        image_paths, vision_features, text_features = load_precomputed_features(feature_file)
    else:
        DATA_PATH_APPLES = r'D:\Small sample\molmo\dataset'
        image_paths = glob(os.path.join(DATA_PATH_APPLES, '*.jpg')) + glob(os.path.join(DATA_PATH_APPLES, '*.png'))
        vision_features, text_features = extract_features(image_paths, model, processor, custom_text)
        save_precomputed_features(image_paths, vision_features, text_features)

    # 创建分类器
    classifier = CombinedClassifier(num_classes=len(Classes)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练和评估分类器（无需每次重新提取特征）
    num_images_per_class = [103, 27, 93, 92, 134, 201, 351, 197, 147]
    y_true = [i for i, count in enumerate(num_images_per_class) for _ in range(count)]

    vision_features = torch.tensor(vision_features).squeeze().to(device)
    text_features = torch.tensor(text_features).squeeze().to(device)
    combined_features = torch.cat((vision_features, text_features), dim=1)

    for epoch in range(5):
        classifier.train()
        optimizer.zero_grad()
        outputs = classifier(combined_features)
        loss = criterion(outputs, torch.tensor(y_true).to(device))
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/5], Loss: {loss.item()}")

    # 评估模型
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(combined_features)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

    # 计算混淆矩阵和其他评估指标
    conf_matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, average=None)

    print(f"混淆矩阵:\n{conf_matrix}")
    print(f"加权F1得分: {f1}")
    print(f"准确率: {accuracy}")
    print(f"Sensitivity (Recall): {sensitivity}")

    # 保存评估结果为CSV
    metrics_df = pd.DataFrame({
        'Class': Classes,
        'F1': f1_score(y_true, y_pred, average=None),
        'Sensitivity': sensitivity
    })
    metrics_df.to_csv('evaluation_metrics_blip_combined_finetuned.csv', index=False)


    # 绘制混淆矩阵
    def plot_confusion_matrix(conf_matrix, fixed_labels):
        plt.figure(figsize=(10, 7))
        plt.imshow(conf_matrix, cmap='Blues', interpolation='none')
        plt.title("Confusion Matrix")
        plt.colorbar()

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


    plot_confusion_matrix(conf_matrix, Classes)

print(result)


