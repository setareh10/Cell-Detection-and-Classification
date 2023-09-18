# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:35:44 2023

@author: setar
"""

import numpy as np
from cdc_classes import (SimpleBlobDetection)
from cdc_utils import get_integer
import matplotlib.pyplot as plt


images = np.load('test_data.npy')

image_num = get_integer("Enter the image number (Range: 0-31): ", 0, 31)
column_num = get_integer("Enter the column number (Range: 0-3): ", 0, 3)
row_num = get_integer("Enter the row number (Range: 0-8): ", 0, 8)




options = []
for background_correction in range(0, 2):
    for mask_droplet in range(0, 2):
        options.append([background_correction, mask_droplet])
        
print('------------------------------------------------------------------------')

for background_correction,  mask_droplet in  options:
   
    print(f'The result of classification for background_correction set to {background_correction}, and mask_droplet to {mask_droplet}')

    blob_detection = SimpleBlobDetection(images, image_num, row_num, column_num, mask_droplet, background_correction)

    im_with_keypoints, classifications  = blob_detection.simple_blob_detection_method()

    print('------------------------------------------------------------------------')

# plt.close('all')

