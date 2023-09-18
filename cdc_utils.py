# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 12:53:07 2023

@author: setar
"""
import numpy as np
from skimage.exposure import rescale_intensity

from skimage import  img_as_ubyte
from skimage.filters import threshold_otsu
from skimage import color, measure, draw
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.morphology import disk, dilation
from skimage.restoration import denoise_bilateral



def get_integer(prompt, min_value, max_value):
    while True:
        try:
            value = int(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Invalid input. Please enter a number between {min_value} and {max_value}.")
        # Catches ValueError exceptions, which occur when the user inputs an invalid value that cannot be converted to an integer.
        except ValueError:
            print(f"That's not a valid number. Please enter a number between {min_value} and {max_value}.")




def correct_background(image):
    
    # remove small dark spots (i.e. “pepper”) and connect small bright cracks
    closed = closing(image, disk(10))
    foreground = image - closed
    # foreground = foreground - np.min(foreground)

    return foreground





def preprocessing(image, background_correction):
    
    # Converts the image to an 8-bit unsigned integer format (0, 255)
    image = img_as_ubyte(image)
    # Reduces noise while preserving edges in the image
    denoised_image = denoise_bilateral(image, sigma_color=0.1)
    
    # normalised_image = normalise_image(denoised_image)
    
    if background_correction:
        foreground_image = correct_background(denoised_image)
    else:
        foreground_image = denoised_image
            
    # contrast_image = equalize_adapthist(foreground_image, clip_limit=0.01, nbins=256)
    # Calculates the 1st and 99th percentiles of the pixel intensities
    v_min, v_max = np.percentile(foreground_image, (1, 99))
    contrast_image = rescale_intensity(foreground_image, in_range=(v_min, v_max))
    preprocessed_image = contrast_image
        
    return preprocessed_image
 
    
 
def crop_image(image, row, col, num_rows=9, num_columns=4):
   
    # Crop a specific region from the input image to create a single droplet.
    # row and col specify the starting point for the crop.
    
    height = image.shape[0] // num_rows
    width = image.shape[1] // num_columns
    
    y1 = row * height
    print(f'height = {height}, row = {row}')
    y2 = (row + 1) * height
    x1 = col * width
    x2 = (col + 1) * width

    # Crop the tile from the original image
    tile = image[y1:y2, x1:x2]
    
    return tile
            
 
        


def get_droplet_region(image):
    

    # Convert the image to grayscale if it's a color image
    if len(image.shape) == 3:
        image_gray = img_as_ubyte(color.rgb2gray(image))
    else:
        image_gray = img_as_ubyte(image)

    # Apply thresholding using Otsu method
    thresh = threshold_otsu(image_gray)
    binary = image_gray > thresh
    

    # Remove any connected components (like noise) that are touching the border of the image
    cleared = clear_border(binary)
    

    # Close small holes and gaps in the droplet
    closed = closing(cleared, square(3))
    

    # Labels the connected components in the binary image
    label_image = measure.label(closed)

    # Finds the largest connected component based on the area
    regions = measure.regionprops(label_image)

    max_area = 0
    largest_region = None
    
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            largest_region = region

    # Get the coordinates of the largest region
    rr, cc = draw.polygon(largest_region.coords[:, 0], largest_region.coords[:, 1])

    # Create an empty image to store the largest region
    droplet_image = np.zeros(image_gray.shape, dtype=np.uint8)

    # Fill in the droplet region
    droplet_image[rr, cc] = 255
    
    footprint = disk(6)
    droplet_image_dilated = dilation(droplet_image, footprint)
    
    return droplet_image_dilated, image_gray



def crop_by_mask(mask, original_image):
    # Use the mask to focus on the inner part of the droplet in the original image
    return original_image * (mask // 255)    


