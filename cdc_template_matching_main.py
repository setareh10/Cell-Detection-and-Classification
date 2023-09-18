# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 19:33:04 2023

@author: setareh
"""


import pickle
import numpy as np
from cdc_classes import (MatchTemplte)
from cdc_utils import get_integer
import matplotlib.pyplot as plt


images = np.load('test_data.npy')

  

    

# Get user input for the type of cell to detect
while True:
    cell_type = input("Which type of cell would you like to detect? (Living/Synthetic): ").strip().lower()
    if cell_type.strip() in ['living', 'synthetic']:
        break
    print("Not valid")

while True:

    background_correction = int(input("Would you like to perform background correction? Enter 1 for 'Yes' or 0 for 'No': "))
    if background_correction in [0, 1]:
        break
    print("Not valid")


while True:
    mask_droplet = int(input("Would you like to mask the droplet? Enter 1 for 'Yes' or 0 for 'No': "))
    if mask_droplet in [0, 1]:
        break
    print("Not valid")
    
    
# Validate the input
if cell_type in ['living', 'synthetic']:
    
    image_num = get_integer("Enter the image number (Range: 0-31): ", 0, 31)
    column_num = get_integer("Enter the column number (Range: 0-3): ", 0, 3)
    row_num = get_integer("Enter the row number (Range: 0-8): ", 0, 8)

    if cell_type == 'living':
        
        print(f'You have selected to detect Living Cells in image {image_num}, at (row, col)=({row_num},{column_num}).')
        
        if background_correction==1:
            with open('temp_cell_bkcorrected.pkl', 'rb') as f:
                templates_cell = pickle.load(f)
                
        else:
            with open('temp_cell_bkuncorrected.pkl', 'rb') as f:
                templates_cell = pickle.load(f)
                
            
            
                    
    # Get and validate the mode
        while True:
            mode = input("Specify the detection mode please (single/multi): ").strip().lower()
            if mode in ['single', 'multi']:
                break
            else:
                print("Invalid selection. Please enter either 'single' or 'multi'.")

        class_name = 'living'
        templates = templates_cell
        # Your code for detecting living cells goes here

    elif cell_type == 'synthetic':
        print(f'You have selected to detect Synthetic Cells in image {image_num}, at (row, col)=({row_num},{column_num}).')

        if background_correction==1:
            with open('temp_synthetic_bkcorrected.pkl', 'rb') as f:
                templates_synthetic = pickle.load(f)
                
        else:
            with open('temp_synthetic_bkuncorrected.pkl', 'rb') as f:
                templates_synthetic = pickle.load(f)
        
        templates = templates_synthetic
        mode = 'single'
        class_name = 'synthetic'
        # Your code for detecting synthetic cells goes here

else:
    print("Invalid selection. Please enter either 'Living' or 'Synthetic'.")


match_template = MatchTemplte(templates, mode)
bounding_boxes = match_template.plot_detected_blobs(images, image_num, row_num, column_num, class_name, mask_droplet, background_correction)




# plt.close('all')
            


