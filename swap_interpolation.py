"""
@author: Marie
Does interpolation by swapping the attributes,  then runs a classifier and gives out accuracy.
Similar to the interpolation.py, but we don't want to ruin the original authors' code.
"""

import os
import argparse
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.image
import matplotlib.pyplot as plt
from PIL import Image
from src.logger import create_logger
from src.loader import load_images, DataSampler
from src.utils import bool_flag

# parse parameters
parser = argparse.ArgumentParser(description='Attributes swapping')
parser.add_argument("--model_path", type=str, default="",
                    help="Trained model path")
parser.add_argument("--n_images", type=int, default=10,
                    help="Number of images to modify")
parser.add_argument("--offset", type=int, default=0,
                    help="First image index")
parser.add_argument("--n_interpolations", type=int, default=10,
                    help="Number of interpolations per image")
parser.add_argument("--alpha_min", type=float, default=2,
                    help="Min interpolation value")
parser.add_argument("--alpha_max", type=float, default=2,
                    help="Max interpolation value")
parser.add_argument("--plot_size", type=int, default=5,
                    help="Size of images in the grid")
parser.add_argument("--row_wise", type=bool_flag, default=True,
                    help="Represent image interpolations horizontally")
parser.add_argument("--output_path", type=str, default="output.png",
                    help="Output path")
params = parser.parse_args()

# check parameters
assert os.path.isfile(params.model_path)
assert params.n_images >= 1 and params.n_interpolations >= 2

# create logger / load trained model
logger = create_logger(None)
ae = torch.load(params.model_path).eval()
# restore main parameters
params.debug = True
params.batch_size = 32
params.v_flip = False
params.h_flip = False
params.img_sz = ae.img_sz
params.attr = ae.attr
params.n_attr = ae.n_attr


if not (len(params.attr) == 1 and params.n_attr == 2):
    raise Exception("The model must use a single boolean attribute only.")

data, attributes = load_images(params)
test_data = DataSampler(data[2], attributes[2], params)

def get_interpolations(ae, images, attributes, params):
    """
    Reconstruct images / create interpolations
    """
    assert len(images) == len(attributes)
    
    # Create latent layer
    enc_outputs = ae.encode(images)
    # interpolation values
    # original image / reconstructed image / interpolations
    outputs = []
    #outputs.append(images)
    swapped_attributes = []
    for i in attributes:
        if i[0] == 1.0:
            swapped_attributes.append([-1., 2.])
        else:
            swapped_attributes.append([2., -1.])
    swapped_attributes = torch.tensor(swapped_attributes).cuda()

    # Add normal reconstructed, then messed up part
    #outputs.append(ae.decode(enc_outputs, attributes)[-1])
    outputs.append(ae.decode(enc_outputs, swapped_attributes)[-1])

    # return stacked images
    return torch.cat([x.unsqueeze(1) for x in outputs], 1).data.cpu()

interpolations = []

for k in range(0, params.n_images, 100):
    i = params.offset + k
    j = params.offset + min(params.n_images, k + 100)
    images, attributes = test_data.eval_batch(i, j)
    interpolations.append(get_interpolations(ae, images, attributes, params))


lmaopth = 'newdata/'
interpolations = torch.cat(interpolations, 0)

def get_grid(images, row_wise, plot_size=5):
    """
    Create a grid with all images.
    """
    n_images, n_columns, img_fm, img_sz, _ = images.size()
    if not row_wise:
        images = images.transpose(0, 1).contiguous()
    images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
    images.add_(1).div_(2.0)
    return make_grid(images, nrow=(n_columns if row_wise else n_images))


# generate the grid / save it to a PNG file
grid = get_grid(interpolations, params.row_wise, params.plot_size)
#matplotlib.image.imsave(params.output_path, grid.numpy().transpose((1, 2, 0)))
del grid

for idx in range(interpolations.size()[0]):
    torchvision.utils.save_image(interpolations[idx][0], lmaopth + 'img{}.jpg'.format(idx))