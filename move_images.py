"""
@author: viet
The most useless script in the entire universe
"""

import os
import torch
from src.loader import load_images
import torchvision
from matplotlib import pyplot as plt
from PIL import Image


img_path = 'data/images_128_128_20000.pth'
lmaopth = 'olddata/'
images = torch.load(img_path)
image = images[3].numpy()
plt.imshow(image.transpose(1,2,0))

for idx in range(5000):
    im = Image.fromarray(images[idx].numpy().transpose(1,2,0))
    im.save('olddata/img{}.jpg'.format(idx))
    #torchvision.utils.save_image(images[idx], lmaopth + 'img{}.jpg'.format(idx))