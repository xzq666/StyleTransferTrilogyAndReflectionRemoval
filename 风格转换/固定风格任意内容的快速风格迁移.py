# -*- coding:utf-8 -*-
# Author: xzq
# Date: 2020-01-08 15:05

import torch
from utils import read_image, imshow, gram_matrix
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from my_models import VGG
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)

# 载入风格图像
style_path = "images/mosaic.jpg"
style_img = read_image(style_path).to(device)
imshow(style_img, title='Style Image')
plt.show()

# 加载自定义vgg16模型
vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

# 载入COCO数据集
batch_size = 4
width = 256
data_transform = transforms.Compose([
    transforms.Resize(width),
    transforms.CenterCrop(width),
    transforms.ToTensor(),
    tensor_normalizer,
])
root = 'COCO/train2014/'
ann_file = 'COCO/annotations/instances_train2014.json'
dataset = torchvision.datasets.CocoDetection(root, ann_file, transform=data_transform)
# dataset = torchvision.datasets.ImageFolder('../COCO/', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(dataset)

# 计算风格图像的Gram矩阵
style_features = vgg16(style_img)
style_grams = [gram_matrix(x) for x in style_features]
style_grams = [x.detach() for x in style_grams]
print([x.shape for x in style_grams])
