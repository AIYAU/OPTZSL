 # GigaGAN
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LeakyReLU, BatchNormalization, Reshape, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from PIL import Image
from glob import glob
import re
import pandas as pd

# 设置数据集路径和类别名称
data_dir = r'D:\Small sample\molmo\dataset'  # 请将此路径修改为您本地数据集的路径
categories = ["Alternaria", "Blossom", "Brown", "Grey", "Healthy", "Mosaic", "Mildew", "Rust", "Scab"]
num_classes = len(categories)


# 数据集准备和预处理
def make_dataset(path: str):
    image_paths = glob(os.path.join(path, '*.jpg')) + glob(os.path.join(path, '*.png'))
    image_names = [os.path.basename(p) for p in image_paths]
    images = [np.array(Image.open(p).resize((128, 128)).convert('RGB')) / 255.0 for p in image_paths]  # 归一化至0-1
    # 根据图像名称推断类别
    labels = []
    for name in image_names:
        matched = False
        for idx, category in enumerate(categories):
            if re.search(category, name, re.IGNORECASE):
                labels.append(idx)
                matched = True
                break
        if not matched:
            labels.append(-1)  # 未知类别
    return image_names, np.stack(images), labels


# 设置图像数据集路径
image_names, images, labels = make_dataset(data_dir)

# 过滤掉未知类别的数据
valid_indices = [i for i, label in enumerate(labels) if label != -1]
image_names = [image_names[i] for i in valid_indices]
images = images[valid_indices]
labels = np.array([labels[i] for i in valid_indices])

# 定义一些超参数
img_height, img_width, channels = 128, 128, 3
latent_dim = 100
learning_rate = 0.0002
epochs = 10
batch_size = 32


# 生成器模型
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(img_height * img_width * channels, activation='tanh'))
    model.add(Reshape((img_height, img_width, channels)))
    return model


# 判别器模型
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=(img_height, img_width, channels), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))  # 输出类别数量的概率
    return model


# 构建生成对抗网络（GAN）
generator = build_generator()
discriminator = build_discriminator()

# 编译判别器
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, beta_1=0.5)

discriminator.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 构建并编译生成对抗模型（不训练判别器）
random_input = Input(shape=(latent_dim,))
generated_image = generator(random_input)
discriminator.trainable = False
validity = discriminator(generated_image)
gan = Model(random_input, validity)
gan.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

# 训练判别器进行分类
num_batches = len(images) // batch_size
for epoch in range(epochs):
    for batch in range(num_batches):
        # 获取真实图像及其标签
        idx = np.random.randint(0, len(images), batch_size)
        real_images, real_labels = images[idx], labels[idx]
        # 生成随机噪声和相应的生成图像
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_images = generator.predict(noise)
        fake_labels = np.random.randint(0, num_classes, batch_size)

        # 训练判别器
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.random.randint(0, num_classes, batch_size)  # 生成器希望生成的图像可以欺骗判别器
        g_loss = gan.train_on_batch(noise, valid_y)

        # 打印进度
        print(
            f"Epoch {epoch + 1}/{epochs}, Batch {batch + 1}/{num_batches} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss}]")

# 使用判别器进行分类预测
y_true = labels
y_pred = []
for i in range(len(images)):
    img = images[i].reshape((1, img_height, img_width, channels))
    pred = discriminator.predict(img)
    y_pred.append(np.argmax(pred))

# 转换为NumPy数组
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 计算混淆矩阵和评估指标
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
sensitivity = recall_score(y_true, y_pred, average=None)

# 打印评估结果
print(f"混淆矩阵:\n{conf_matrix}")
print(f"加权F1得分: {f1}")
print(f"准确率: {accuracy}")
print(f"Sensitivity (Recall): {sensitivity}")

# 检查 categories 和 sensitivity 的长度
print(f"Categories 长度: {len(categories)}")
print(f"Sensitivity 长度: {len(sensitivity)}")

# 如果长度不一致，则可能需要处理未分类的类别
if len(categories) == len(sensitivity):
    # 保存评估结果为CSV文件
    metrics_df = pd.DataFrame({
        'Class': categories,
        'Sensitivity': sensitivity
    })
    metrics_df.to_csv('evaluation_metrics.csv', index=False)
else:
    print("Error: Categories and Sensitivity length mismatch!")


# 保存评估结果为CSV文件
metrics_df = pd.DataFrame({
    'Class': categories,
    'Sensitivity': sensitivity
})
metrics_df.to_csv('evaluation_metrics.csv', index=False)


# 定义绘制混淆矩阵的函数
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


plot_confusion_matrix(conf_matrix, categories)

