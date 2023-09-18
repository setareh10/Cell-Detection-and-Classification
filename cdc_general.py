# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:41:04 2023

@author: Setareh
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lightcast_utils import preprocessing


imgs = np.load('test_data.npy')

plt.figure(figsize=(12,6))
plt.imshow(imgs[31,], cmap=plt.cm.gray)

background_correction = 0
image_list = []



if background_correction:
    
    imgs_preprocessed = np.load('preprocessed_images_bkcorrected.npy')
    file_name_preprocessed = 'preprocessed_images_bkcorrected.npy'
    file_name_synthetic = 'temp_synthetic_bkcorrected.pkl'
    file_name_cell = 'temp_cell_bkcorrected.pkl'
    
else:
    
    imgs_preprocessed = np.load('preprocessed_images_bkuncorrected.npy')
    file_name_preprocessed = 'preprocessed_images_bkuncorrected.npy'
    file_name_synthetic = 'temp_synthetic_bkuncorrected.pkl'
    file_name_cell = 'temp_cell_bkuncorrected.pkl'
    

    


for i in range(imgs.shape[0]):
    
    im = imgs[i]
    preprocced_image = preprocessing(im, background_correction)
    image_list.append(preprocced_image)
    
# image_array = np.array(image_list, dtype=object)
image_array = np.stack(image_list, axis=0)
# Save the stacked images as a single .npy file
np.save(file_name_preprocessed, image_array)


## Plot (and save) preprocessed images

for i in range(imgs.shape[0]):
    
    im = imgs[i]
    
    # filename = f'C:\\Users\\setar\\Desktop\\lightcast_exercise\\preprocessed images\\preprocessed_image_{i}.jpg' 

    preprocced_image = preprocessing(im, background_correction)
    
    plt.figure(figsize=(12,6))
    plt.imshow(preprocced_image, cmap=plt.cm.gray)
    # plt.savefig(filename)

plt.close('all')


# Create templates for synthetic/dark and living/bright cells


plt.figure(figsize=(12,6))
plt.imshow(imgs_preprocessed[0,1536:1546,548:558], cmap=plt.cm.gray)
plt.close('all')


temp_dark_12 = imgs_preprocessed[0,102:118, 155:175]#imgs_preprocessed[0,106:117, 161:171]
temp_dark_12 = imgs_preprocessed[1,256:275, 1256:1274]#imgs_preprocessed[1,262:271, 1260:1271]#imgs_preprocessed[1,256:275, 1256:1274]

temp_dark_2e = imgs_preprocessed[2,975:988,250:262]#imgs_preprocessed[2,977:987,252:263]#imgs_preprocessed[2,975:988,250:262]
temp_dark_2w = imgs_preprocessed[2,623:636,475:486]#imgs_preprocessed[2,625:635,475:484]#imgs_preprocessed[2,623:636,475:486]
temp_dark_2n = imgs_preprocessed[4,577:588,199:211]#imgs_preprocessed[4,577:586,200:210]#imgs_preprocessed[4,577:588,199:211]
temp_dark_2s = imgs_preprocessed[5,1204:1215,173:186]#imgs_preprocessed[5,1206:1215,175:184]#imgs_preprocessed[5,1204:1215,173:186]

temp_dark_31 = imgs_preprocessed[0,1532:1547,543:560]#imgs_preprocessed[0,1536:1546,548:558]#imgs_preprocessed[0,1532:1547,543:560]

temp_synthetic = [temp_dark_12, temp_dark_12, temp_dark_2e, temp_dark_2w,
                  temp_dark_2n, temp_dark_2s, temp_dark_31]

temp_bright_11 = imgs_preprocessed[1,281:300,513:533]
temp_bright_12 = imgs_preprocessed[1,1318:1335,554:575]
temp_bright_13 = imgs_preprocessed[3,807:828,520:544]

temp_bright_2e = imgs_preprocessed[3,802:817,231:246]
temp_bright_2s = imgs_preprocessed[7,1030:1043, 525:541]
temp_bright_2n = imgs_preprocessed[19,743:754,1243:1257]
temp_bright_2w = imgs_preprocessed[14, 72:82,135:147]

temp_bright_31 = imgs_preprocessed[7,226:235,855:866]

temp_bright_41 = imgs_preprocessed[3,765:788,913:938]


temp_cell = [temp_bright_11, temp_bright_12, temp_bright_13, temp_bright_2e,
              temp_bright_2s, temp_bright_2n, temp_bright_2w, temp_bright_31,
              temp_bright_41]



## Save temples as pickle files
with open(file_name_synthetic, 'wb') as f:
    pickle.dump(temp_synthetic, f)
    
    
with open(file_name_cell, 'wb') as f:
    pickle.dump(temp_cell, f)
    

