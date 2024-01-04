# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split,Dataset,Subset
from torchinfo import summary
from tqdm import tqdm
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import os
import re

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

"""# Create Resnet structure"""

# get BasicBlock which layers < 50(18, 34)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample #Residual Block結構的輸入與輸出通道數量不一致，這時候我們就需要在短路連接的時候調整通道數量

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# get BottleBlock which layers >= 50
class Bottleneck(nn.Module):
    expansion = 4 # the factor of the last layer of BottleBlock and the first layer of it

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*4)
        self.downsample = downsample
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=7):
        super(ResNet, self).__init__()
        self.in_channel = 64 # 64for utk_teacher_model.pth / 8 for others .pth

        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):

        # Initial convolution and batch normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # First set of ResNet blocks
        feature1 = self.layer1(x)

        # Second set of ResNet blocks
        feature2 = self.layer2(feature1)

        # Third set of ResNet blocks
        feature3 = self.layer3(feature2)

        # Fourth set of ResNet blocks
        feature4 = self.layer4(feature3)

        # Global Average Pooling
        out = self.avgpool(feature4)
        out = out.view(out.size(0), -1) # Flatten the tensor

        # Fully connected layer for classification
        out = self.fc(out)

        return out, [feature1, feature2, feature3, feature4]

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


"""## Load model"""
MyModel = resnet34(num_classes=7)  # commment out this line if loading trained MyModel model
state_dict = torch.load('C:\\Users\\user\\Desktop\\EAI project\\django based WEB\\age_prediction_for_several_1227\\weights\\utk_teacher_model.pth', map_location ='cpu')  # loading trained MyModel model

MyModel.load_state_dict(state_dict)
MyModel = MyModel.to(device)



"""# Face Detection Model (MTCNN)"""
"""## Setup for test"""

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import torch
from torch.utils.data import DataLoader
from collections import Counter

import shutil

# MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cuda')

# Transformation as used in your training, now including a grayscale conversion
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # or any other size used during training
    transforms.Grayscale(),  # Convert the image to grayscale
    transforms.ToTensor(),
])

# Define age ranges based on your training data
age_ranges = ["1-2", "3-9", "10-20", "21-27", "28-45", "46-65", "66-116"]

def display_image_with_ages(original_image, boxes, age_estimates):
    plt.imshow(np.array(original_image))
    ax = plt.gca()

    for box, age in zip(boxes, age_estimates):
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red', linewidth=1))
        plt.text(box[0], box[1], age, color='white', fontsize=10, backgroundcolor='red')

    plt.axis('off')
    output_folder = 'C:\\Users\\user\\Desktop\\EAI project\\django based WEB\\web\\myproject\\myproject\\static'
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f'{output_folder}/output.png')


def save_ad_image(ad):
    if 0 <= ad <= 7:
        ad_filename = f'ad_{ad}.jpg'  # Format the ad file name with leading zero
        ad_source_path = f'C:\\Users\\user\\Desktop\\EAI project\\django based WEB\\ads\\{ad_filename}'  # Replace with your actual path to adS folder
        output_folder = 'C:\\Users\\user\\Desktop\\EAI project\\django based WEB\\web\\myproject\\myproject\\static'

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Destination path for the ad image in the static folder
        ad_dest_path = f'{output_folder}/ad.png'

        # Copy the corresponding ad image to the static folder
        shutil.copyfile(ad_source_path, ad_dest_path)

        # Return the name of the saved ad image
        return 'ad.png'
    else:
        return None  


def show_ad(most_age):
  if(most_age == '1-2'):
    return 0
  elif(most_age == '3-9'):
    return 1
  elif(most_age == '10-20'):
    return 2
  elif(most_age == '21-27'):
    return 3
  elif(most_age == '28-45'):
    return 4
  elif(most_age == '46-65'):
    return 5
  elif(most_age == '66-116'):
    return 6
  else:
    return 7

def predict_ages(image_path):
    # Load and process the image
    img = Image.open(image_path)
    img = img.convert('RGB')

    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        age_estimates = []

        # Initialize a counter for each age range
        age_range_count = Counter({age_range: 0 for age_range in age_ranges})

        draw = ImageDraw.Draw(img)

        for box in boxes:
            # Crop each face and apply the transformation
            face = img.crop(box)
            face = transform(face).unsqueeze(0).to(device)  # Add batch dimension and move to the device

            with torch.no_grad():
                output = MyModel(face) # Get the output from the model

                # If the output is a tuple, extract the tensor containing the age predictions
                if isinstance(output, tuple):
                    output = output[0]  # Assuming the first element of the tuple is the desired output

                predicted_age_range_idx = torch.argmax(output, dim=1).item()  # Get the predicted age range index
                predicted_age_range = age_ranges[predicted_age_range_idx]
                age_estimates.append(predicted_age_range)

                # Increment the count for the predicted age range
                age_range_count[predicted_age_range] += 1

            # Draw bounding box
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)

        # print("Got ages:" + str(age_estimates))
        print(str(age_range_count)+ "\n")

        # Calculate how many people in various age range
        most_common_ages = age_range_count.most_common(2)

        print("The most numerous range:", most_common_ages[0])


        ad = show_ad(most_common_ages[0][0])
        print(ad)
        save_ad_image(ad)

        # Display the results
        display_image_with_ages(img, boxes, age_estimates)

    else:

        plt.imshow(np.array(img))
        ax = plt.gca()

        plt.text(150,150,'No faces detected', color='white', fontsize=12, backgroundcolor='red')
        plt.axis('off')
        output_folder = 'C:\\Users\\user\\Desktop\\EAI project\\django based WEB\\myproject\\myproject\\static'

        plt.savefig(f'{output_folder}/output.png')
        save_ad_image(7)


"""# Test our flow with images"""

import os
path = "C:\\Users\\user\\Desktop\\EAI project\\django based WEB\\input\\input.jpg"
predict_ages(path)
