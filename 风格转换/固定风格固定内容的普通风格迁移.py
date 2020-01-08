# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2020-01-08 10:07

import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import read_image, imshow, gram_matrix
import matplotlib.pyplot as plt
import torchvision.models as models
from my_models import VGG
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入图像
width = 512
style_img = read_image('images/picasso.jpg', target_width=width).to(device)
content_img = read_image('images/dancing.jpg', target_width=width).to(device)

# 显示风格图像与内容图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
imshow(style_img, title='Style Image')
plt.subplot(1, 2, 2)
imshow(content_img, title='Content Image')
plt.show()

# 加载模型
vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

# 计算特征
# 特征为VGG16的relu_1_2、relu_2_2、relu_3_3和relu_4_3
style_features = vgg16(style_img)
content_features = vgg16(content_img)
print([x.shape for x in content_features])


style_grams = [gram_matrix(x) for x in style_features]
print([x.shape for x in style_grams])

# 训练
input_img = content_img.clone()
optimizer = optim.LBFGS([input_img.requires_grad_()])
style_weight = 1e6
content_weight = 1
epochs = 300
epoch = [0]
while epoch[0] <= epochs:
    def f():
        optimizer.zero_grad()
        features = vgg16(input_img)
        # 根据生成图像和内容图像在relu_3_3输出的特征图的均方误差作为内容损失。
        content_loss = F.mse_loss(features[2], content_features[2]) * content_weight
        # 根据生成图像和内容图像在relu_1_2、relu_2_2、relu_3_3、relu_4_3输出的特征图的均方误差作为风格损失。
        style_loss = 0
        grams = [gram_matrix(x) for x in features]
        for a, b in zip(grams, style_grams):
            style_loss += F.mse_loss(a, b) * style_weight
        loss = content_loss + style_loss
        if epoch[0] % 10 == 0:
            print('Step {}: Style Loss: {:4f} Content Loss: {:4f}'.format(
                epoch[0], style_loss.item(), content_loss.item()))
        epoch[0] += 1
        loss.backward()
        return loss
    optimizer.step(f)

# 可视化风格图片
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
imshow(style_img, title='Style Image')
plt.subplot(1, 3, 2)
imshow(content_img, title='Content Image')
plt.subplot(1, 3, 3)
imshow(input_img, title='Output Image')
plt.show()
