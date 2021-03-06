#!/usr/bin/env python
import os
import matplotlib.image as mpimg
import cv2
import numpy as np
import torch
from joblib import Parallel, delayed


N_IMAGES =  120000 
IMG_SIZE = 128
IMG_PATH = 'images_%i_%i.pth' % (IMG_SIZE, IMG_SIZE)
ATTR_PATH = 'attributes.pth'
all_images = []
print('something')
def preprocess_images():

    if os.path.isfile(IMG_PATH):
        print("%s exists, nothing to do." % IMG_PATH)
        return
    global all_images
    print("Reading images from img_align_celeba/ ...")
    #raw_images = []
    
    def single_processing(i): 
        if i % 10000 == 0:
            print(i)
        image = mpimg.imread('img_align_celeba/%06i.jpg' % i)[20:-20]

        # Image processing
        assert image.shape == (178, 178, 3)
        if IMG_SIZE < 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        elif IMG_SIZE > 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        global all_images
        return image 
        
    all_images = Parallel(n_jobs=12)(delayed(single_processing)(i) for i in range(1, N_IMAGES + 1))
    '''
    for i in range(1, N_IMAGES + 1):
        if i % 10000 == 0:
            print(i)
        image = mpimg.imread('img_align_celeba/%06i.jpg' % i)[20:-20]

        # Image processing
        assert image.shape == (178, 178, 3)
        if IMG_SIZE < 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        elif IMG_SIZE > 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        all_images.append(image)

        #raw_images.append(mpimg.imread('img_align_celeba/%06i.jpg' % i)[20:-20])
    '''
    #if len(raw_images) != N_IMAGES:
        #raise Exception("Found %i images. Expected %i" % (len(raw_images), N_IMAGES))
    if len(all_images) != N_IMAGES:
        raise Exception("Found %i images, Expeccted %i" % (len(all_images), N_IMAGES))
    '''
    print("Resizing images ...")
    all_images = []
    for i, image in enumerate(raw_images):
        if i % 10000 == 0:
            print(i)
        assert image.shape == (178, 178, 3)
        if IMG_SIZE < 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        elif IMG_SIZE > 178:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        all_images.append(image)
    '''
    '''
    data = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
    data = torch.from_numpy(data)
    assert data.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(data[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(data, IMG_PATH)'''
    all_images = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
    all_images = torch.from_numpy(all_images)
    assert all_images.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(all_images[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(all_images, IMG_PATH)
 

def preprocess_attributes():

    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return
    
    with open('list_attr_celeba.txt', 'r') as f:
        attr_lines = [next(f) for x in range(N_IMAGES + 2)] # because of the first two lines lmfao
    #attr_lines = [line.rstrip() for line in open('list_attr_celeba.txt', 'r')]
    print(len(attr_lines))
    assert len(attr_lines) == N_IMAGES + 2

    attr_keys = attr_lines[1].split()
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in attr_keys}

    for i, line in enumerate(attr_lines[2:]):
        image_id = i + 1
        split = line.split()
        assert len(split) == 41
        assert split[0] == ('%06i.jpg' % image_id)
        assert all(x in ['-1', '1'] for x in split[1:])
        for j, value in enumerate(split[1:]):
            attributes[attr_keys[j]][i] = value == '1'

    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, ATTR_PATH)


preprocess_images()
preprocess_attributes()

with open('logs_pre.txt', 'w+') as f:
    f.write("Preprocessed {} images of size {}".format(N_IMAGES, IMG_SIZE))